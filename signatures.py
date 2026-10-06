"""
Heybo signature (pre-set) meals — aligned with salad.generate_signature_bowls:

Menu row filtering matches Salad (nutrient, price, allergen, diet, cuisine, prep, flavor,
light/hearty) using the **raw** request before preprocess_heybo_filters, so DietFilters
behave like Salad. Heybo catalog uses heybo.menu_details.sub_category vs BowlType (Salad uses
category). After ingredients load, Salad’s Include keyword filter runs on recipe SKUs.

Bowl build: Include/Exclude/Extra, SKU availability, customization, pricing — Salad parity.
image_id from gogreenfood_menu. Pagination: heybo.display_meal dedupe by recommend_page_id.
"""
from __future__ import annotations

import os
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from .config_loader import get_heybo_config
from .db import (
    get_heybo_available_skus,
    load_heybo_data_from_db,
    load_heybo_previous_signature_bowl_names,
    load_heybo_signature_catalog,
    load_heybo_vendor_sku_to_image_id_map,
)
from .generation import HEYBO_BOWLS_PER_PAGE, PRICE_RELAXATION_SLACKS
from .diet_constants import HEYBO_BOWL_COMPONENT_KEYS, HEYBO_EXTRA_CATEGORIES, NUTRIENT_COLUMNS
from .filters import preprocess_heybo_filters
from .pricing import apply_pricing_tier_split_to_bowl, calculate_heybo_bowl_cost_with_breakdown
from .user_message import build_heybo_signature_user_message

# Base category (Proteins / Warm / Cold) → bowl list for user-requested second portions.
_HEYBO_MAIN_CATEGORY_TO_EXTRA_BOWL_KEY = {v: k for k, v in HEYBO_EXTRA_CATEGORIES.items()}


def _is_truthy_flag(val: Any) -> bool:
    if val is True:
        return True
    if isinstance(val, str) and val.strip().lower() in ("true", "1", "yes", "on"):
        return True
    if isinstance(val, (int, float)) and val == 1:
        return True
    return False


def _sig_debug_enabled(user_input: dict) -> bool:
    # Keep Salad parity: DEBUG=1 in env enables signature debug logs globally.
    env_debug = os.getenv("DEBUG")
    if env_debug is not None:
        return _is_truthy_flag(env_debug)
    # Backward-compatible fallback in case DEBUG_ENABLED is used in some environments.
    env_debug_enabled = os.getenv("DEBUG_ENABLED")
    if env_debug_enabled is not None:
        return _is_truthy_flag(env_debug_enabled)
    # Request-level override if env isn't set.
    return _is_truthy_flag(user_input.get("SignatureDebug") or user_input.get("_debug_signatures"))


def _sig_dbg(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[HEYBO_SIGNATURE_DEBUG] {message}")


def _empty_heybo_bowl_shell(shop_name: Any, bowl_type: str, menu_name: str, sku_code: str) -> Dict[str, Any]:
    bowl: Dict[str, Any] = {
        "Shop Name": shop_name if isinstance(shop_name, list) else [shop_name] if shop_name else ["heybo"],
        "Bowl Type": bowl_type,
        "Bowl Name": menu_name,
        "sku_code": sku_code,
        "customized": False,
    }
    for k in HEYBO_BOWL_COMPONENT_KEYS:
        bowl[k] = []
    return bowl


def _map_category_to_bowl_list(bowl: Dict[str, Any], category: str, ingredient_name: str) -> None:
    if not category or not isinstance(category, str):
        return
    cat = category.strip()
    if cat in HEYBO_BOWL_COMPONENT_KEYS:
        bowl[cat].append(ingredient_name)


def _append_sku_to_bowl(bowl: Dict[str, Any], df: pd.DataFrame, sku: str) -> bool:
    ir = df[df["sku_code"] == sku]
    if ir.empty:
        return False
    r = ir.iloc[0]
    cat = str(r.get("category") or "").strip()
    nm = str(r.get("ingredient_name") or "")
    if cat not in HEYBO_BOWL_COMPONENT_KEYS:
        return False
    _map_category_to_bowl_list(bowl, cat, nm)
    return True


def _sku_list_from_row(ingredients_csv: str) -> List[str]:
    if not ingredients_csv or not isinstance(ingredients_csv, str):
        return []
    return [s.strip() for s in ingredients_csv.split(",") if s.strip()]


def _merge_signature_ingredient_sets(user_input: dict) -> Tuple[Set[str], Set[str], Set[str]]:
    """Salad parity: top-level Include/Exclude/Extra ∪ Ingredients.*."""
    ing = user_input.get("Ingredients") or {}
    inc: Set[str] = {str(x).strip() for x in (user_input.get("Include") or []) if str(x).strip()}
    inc |= {str(x).strip() for x in (ing.get("Include") or []) if str(x).strip()}
    exc: Set[str] = {str(x).strip().lower() for x in (user_input.get("Exclude") or []) if str(x).strip()}
    exc |= {str(x).strip().lower() for x in (ing.get("Exclude") or []) if str(x).strip()}
    ext: Set[str] = {str(x).strip() for x in (user_input.get("Extra") or []) if str(x).strip()}
    ext |= {str(x).strip() for x in (ing.get("Extra") or []) if str(x).strip()}
    return inc, exc, ext


def _signature_specific_bowl_requested(user_input: dict) -> bool:
    bn = user_input.get("BowlName")
    if isinstance(bn, list):
        return any(str(x).strip() for x in bn if x is not None)
    return isinstance(bn, str) and bool(bn.strip())


def _ingredient_marked_for_removal(ingredient_name: str, ingredients_to_remove: List[str]) -> bool:
    nl = ingredient_name.lower()
    for exc in ingredients_to_remove:
        el = str(exc).lower().strip()
        if el == nl or el in nl or nl in el:
            return True
    return False


def _recipe_skus_after_exclude_allergen(
    skus: List[str],
    df: pd.DataFrame,
    ingredients_to_remove: List[str],
    allergen_filters: List[str],
) -> List[str]:
    """Drop recipe lines excluded by name token or allergen (Salad removes lines, not whole bowl)."""
    kept: List[str] = []
    for sku in skus:
        ir = df[df["sku_code"] == sku]
        if ir.empty:
            continue
        r = ir.iloc[0]
        nm = str(r.get("ingredient_name") or "")
        if ingredients_to_remove and _ingredient_marked_for_removal(nm, ingredients_to_remove):
            continue
        if allergen_filters:
            ag = str(r.get("allergens") or "").lower()
            if ag and any(a in ag for a in allergen_filters):
                continue
        kept.append(sku)
    return kept


def _lookup_first_available_sku_by_name(
    df: pd.DataFrame, ingredient_name: str, available: Set[str]
) -> Optional[str]:
    """Exact ingredient_name match (case-insensitive), first SKU that is location-available."""
    rows = df[df["ingredient_name"].fillna("").astype(str).str.lower() == ingredient_name.strip().lower()]
    for _, r in rows.iterrows():
        sku = str(r.get("sku_code") or "").strip()
        if sku and sku in available:
            return sku
    return None


def _extras_max_for_category(heybo_cfg: dict, category: str) -> int:
    lim = heybo_cfg.get("category_limits_extras") or {}
    if category in lim:
        t = lim[category]
        if isinstance(t, (list, tuple)) and len(t) >= 2:
            return max(0, int(t[1]))
    for key, t in lim.items():
        if category.startswith(str(key)) or str(key).startswith(category.split()[0]):
            if isinstance(t, (list, tuple)) and len(t) >= 2:
                return max(0, int(t[1]))
    return 99


def _fold_name_for_match(s: str) -> str:
    """
    Fold for BowlName vs menu_name substring match: strip combining marks (accents)
    so e.g. 'kotai' matches 'KōTai' (ō → o). No extra dependency; uses Unicode NFKD.
    """
    if not s:
        return ""
    nfd = unicodedata.normalize("NFKD", str(s))
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return stripped.casefold()


def _aggregate_from_ingredients(df: pd.DataFrame, skus: List[str]) -> Tuple[Dict[str, float], float, float]:
    nutrients = {k: 0.0 for k in NUTRIENT_COLUMNS}
    weight = 0.0
    co2 = 0.0
    for sku in skus:
        rows = df[df["sku_code"] == sku]
        if rows.empty:
            continue
        r = rows.iloc[0]
        w = float(pd.to_numeric(r.get("serving_amount_per_portion_in_grams"), errors="coerce") or 0)
        weight += w
        c = float(pd.to_numeric(r.get("co2e_values_per_serving"), errors="coerce") or 0)
        co2 += c
        for col in NUTRIENT_COLUMNS:
            if col in r.index:
                nutrients[col] += float(pd.to_numeric(r.get(col), errors="coerce") or 0)
    return nutrients, weight, co2


def _heybo_coerce_price_bound(raw) -> Optional[float]:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _heybo_apply_signature_selling_price_filter(
    out: pd.DataFrame,
    price_filter: dict,
    flag_ui: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Filter signature catalog rows by ``selling_price``.

    • Normal Min/Max band: keep rows in ``[Min, Max]`` when possible.
    • Point target (Min == Max, e.g. $11.90–$11.90): treat as a **budget** —
      return bowls ``<= Max`` ranked nearest to the target (same intent as
      Max-only), not only exact-price hits.
    • If still empty: widen Max with BYB ``PRICE_RELAXATION_SLACKS``, then
      fall back to nearest catalog prices.

    Sets ``_signature_price_relaxed`` / ``_signature_price_relax_mode`` on
    ``flag_ui`` when the strict band is not used.
    """
    if out.empty or "selling_price" not in out.columns:
        return out
    if not isinstance(price_filter, dict):
        return out

    mn = _heybo_coerce_price_bound(price_filter.get("Min"))
    mx = _heybo_coerce_price_bound(price_filter.get("Max"))
    if mn is None and mx is None:
        return out

    prices = pd.to_numeric(out["selling_price"], errors="coerce")
    point_target = (
        mn is not None
        and mx is not None
        and abs(float(mn) - float(mx)) < 1e-6
    )
    if mn is not None and mx is not None:
        target = (float(mn) + float(mx)) / 2.0
    elif mx is not None:
        target = float(mx)
    else:
        target = float(mn)

    def _rank_by_distance(df: pd.DataFrame) -> pd.DataFrame:
        dist = (pd.to_numeric(df["selling_price"], errors="coerce") - target).abs()
        ranked = df.assign(_sig_price_dist=dist).sort_values(
            by=["_sig_price_dist", "selling_price"],
            ascending=[True, False],
            kind="mergesort",
        )
        return ranked.drop(columns=["_sig_price_dist"], errors="ignore")

    def _flag(mode: str, slack: Optional[float] = None) -> None:
        if flag_ui is None:
            return
        flag_ui["_signature_price_relaxed"] = True
        flag_ui["_signature_price_relax_mode"] = mode
        flag_ui["_signature_price_target"] = target
        if slack is not None:
            flag_ui["_signature_price_slack"] = float(slack)

    # Point Min==Max → budget-style matches (under Max), nearest first.
    if point_target and mx is not None:
        under = out.loc[prices.notna() & (prices <= mx)]
        if not under.empty:
            exact = under.loc[
                (pd.to_numeric(under["selling_price"], errors="coerce") - target).abs()
                < 1e-6
            ]
            if exact.empty:
                _flag("under_budget")
            return _rank_by_distance(under)
        for slack in PRICE_RELAXATION_SLACKS:
            widened = out.loc[prices.notna() & (prices <= mx + float(slack))]
            if not widened.empty:
                _flag("widened_budget", slack=float(slack))
                return _rank_by_distance(widened)
        valid = out.loc[prices.notna()]
        if valid.empty:
            return out.iloc[0:0]
        _flag("nearest")
        return _rank_by_distance(valid)

    # Strict band for real ranges / Max-only / Min-only.
    mask = prices.notna()
    if mn is not None:
        mask = mask & (prices >= mn)
    if mx is not None:
        mask = mask & (prices <= mx)
    strict = out.loc[mask]
    if not strict.empty:
        return strict

    # Soft: prefer under Max when present, else nearest to target.
    if mx is not None:
        under = out.loc[prices.notna() & (prices <= mx)]
        if not under.empty:
            _flag("under_max")
            return _rank_by_distance(under)
        for slack in PRICE_RELAXATION_SLACKS:
            widened = out.loc[prices.notna() & (prices <= mx + float(slack))]
            if not widened.empty:
                _flag("widened_budget", slack=float(slack))
                return _rank_by_distance(widened)

    valid = out.loc[prices.notna()]
    if valid.empty:
        return out.iloc[0:0]
    _flag("nearest")
    return _rank_by_distance(valid)


def apply_salad_signature_menu_filters(
    sig_df: pd.DataFrame,
    ui: dict,
    heybo_cfg: Optional[dict] = None,
    ui_meta: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Mirror salad.generate_signature_bowls menu_details filters (salad.py ~22117–22456).
    Pass the **request as the client sent it** (before preprocess_heybo_filters) so DietFilters
    and NutrientFilters match Salad behavior. Heybo differs only in catalog query (sub_category).
    """
    out = sig_df
    if out.empty:
        return out

    flag_ui = ui_meta if isinstance(ui_meta, dict) else ui

    # --- Nutrient filtering (same as Salad: Range.Min / Range.Max) ---
    nutrient_filters = ui.get("NutrientFilters") or []
    if nutrient_filters:
        for nf in nutrient_filters:
            if not isinstance(nf, dict):
                continue
            col = nf.get("Nutrient")
            rng = nf.get("Range", {})
            if col in out.columns:
                if "Min" in rng and rng.get("Min") is not None:
                    out = out[out[col] >= rng["Min"]]
                if "Max" in rng and rng.get("Max") is not None:
                    out = out[out[col] <= rng["Max"]]

    # --- Price filtering (strict band, then likely matches under budget / nearest) ---
    price_filter = ui.get("Price") or {}
    if price_filter:
        out = _heybo_apply_signature_selling_price_filter(
            out, price_filter, flag_ui=flag_ui if isinstance(flag_ui, dict) else None
        )

    # --- Allergen filtering ---
    from .filters import _get_filter_list, ingredient_names_excluded_by_allergen

    allergen_filters = _get_filter_list(ui.get("AllergenFilters"))
    if allergen_filters and "allergens" in out.columns and "ingredient_name" in out.columns:
        excluded_names = ingredient_names_excluded_by_allergen(out, allergen_filters)
        out = out[~out["ingredient_name"].isin(excluded_names)]

    # --- Diet parameter filtering (same token rules as Salad) ---
    diet_filters = ui.get("DietFilters", {})
    if diet_filters and "diet_parameters" in out.columns:
        if isinstance(diet_filters, dict):
            diet_filters = [diet for diet, active in diet_filters.items() if active]
        elif not isinstance(diet_filters, list):
            diet_filters = []
        if diet_filters:

            def has_all_diets(diet_params: Any, filters: List[Any]) -> bool:
                if pd.isna(diet_params):
                    return False
                diets_set = {d.strip().lower() for d in str(diet_params).split(",")}
                return all(str(f).lower() in diets_set for f in filters)

            out = out[out["diet_parameters"].apply(lambda x: has_all_diets(x, diet_filters))]

    # --- Cuisine filtering ---
    cuisine_filters = ui.get("CuisineFilters", {})
    cuisine_filter_logic_used = None
    if cuisine_filters and "cuisine" in out.columns:
        if isinstance(cuisine_filters, dict):
            included_cuisines = [cuisine.strip().lower() for cuisine, include in cuisine_filters.items() if include is True]
            excluded_cuisines = [cuisine.strip().lower() for cuisine, include in cuisine_filters.items() if include is False]
            if included_cuisines or excluded_cuisines:
                if len(included_cuisines) > 1:

                    def matches_cuisine_filter_and(cuisine_value: Any) -> bool:
                        if pd.isna(cuisine_value):
                            return False
                        cuisine_list = [c.strip().lower() for c in str(cuisine_value).split(",") if c.strip()]
                        if included_cuisines:
                            if not all(c in cuisine_list for c in included_cuisines):
                                return False
                        if excluded_cuisines:
                            if any(c in cuisine_list for c in excluded_cuisines):
                                return False
                        return True

                    sig_df_and = out[out["cuisine"].apply(matches_cuisine_filter_and)]
                    if len(sig_df_and) > 0:
                        out = sig_df_and
                        cuisine_filter_logic_used = "AND"
                    else:

                        def matches_cuisine_filter_or(cuisine_value: Any) -> bool:
                            if pd.isna(cuisine_value):
                                return False
                            cuisine_list = [c.strip().lower() for c in str(cuisine_value).split(",") if c.strip()]
                            if included_cuisines:
                                if not any(c in cuisine_list for c in included_cuisines):
                                    return False
                            if excluded_cuisines:
                                if any(c in cuisine_list for c in excluded_cuisines):
                                    return False
                            return True

                        out = out[out["cuisine"].apply(matches_cuisine_filter_or)]
                        cuisine_filter_logic_used = "OR"
                else:

                    def matches_cuisine_filter(cuisine_value: Any) -> bool:
                        if pd.isna(cuisine_value):
                            return False
                        cuisine_list = [c.strip().lower() for c in str(cuisine_value).split(",") if c.strip()]
                        if included_cuisines:
                            if not any(c in cuisine_list for c in included_cuisines):
                                return False
                        if excluded_cuisines:
                            if any(c in cuisine_list for c in excluded_cuisines):
                                return False
                        return True

                    out = out[out["cuisine"].apply(matches_cuisine_filter)]

                if cuisine_filter_logic_used and ui_meta is not None:
                    ui_meta["_cuisine_filter_logic_used"] = cuisine_filter_logic_used

    # --- Preparation method filtering ---
    prep_method_filters = ui.get("PreparationMethod", {})
    prep_method_filter_logic_used = None
    if prep_method_filters and "preparation_method" in out.columns:
        if isinstance(prep_method_filters, dict):
            included_methods = [method.strip() for method, include in prep_method_filters.items() if include is True]
            excluded_methods = [method.strip() for method, include in prep_method_filters.items() if include is False]
            if included_methods or excluded_methods:
                if len(included_methods) > 1:

                    def matches_prep_method_filter_and(prep_method_value: Any) -> bool:
                        if pd.isna(prep_method_value):
                            return False
                        prep_method_list = [m.strip() for m in str(prep_method_value).split(",") if m.strip()]
                        if included_methods:
                            if not all(method in prep_method_list for method in included_methods):
                                return False
                        if excluded_methods:
                            if any(method in prep_method_list for method in excluded_methods):
                                return False
                        return True

                    sig_df_and = out[out["preparation_method"].apply(matches_prep_method_filter_and)]
                    if len(sig_df_and) > 0:
                        out = sig_df_and
                        prep_method_filter_logic_used = "AND"
                    else:

                        def matches_prep_method_filter_or(prep_method_value: Any) -> bool:
                            if pd.isna(prep_method_value):
                                return False
                            prep_method_list = [m.strip() for m in str(prep_method_value).split(",") if m.strip()]
                            if included_methods:
                                if not any(method in prep_method_list for method in included_methods):
                                    return False
                            if excluded_methods:
                                if any(method in prep_method_list for method in excluded_methods):
                                    return False
                            return True

                        out = out[out["preparation_method"].apply(matches_prep_method_filter_or)]
                        prep_method_filter_logic_used = "OR"
                else:

                    def matches_prep_method_filter(prep_method_value: Any) -> bool:
                        if pd.isna(prep_method_value):
                            return False
                        prep_method_list = [m.strip() for m in str(prep_method_value).split(",") if m.strip()]
                        if included_methods:
                            if not any(method in prep_method_list for method in included_methods):
                                return False
                        if excluded_methods:
                            if any(method in prep_method_list for method in excluded_methods):
                                return False
                        return True

                    out = out[out["preparation_method"].apply(matches_prep_method_filter)]

                if prep_method_filter_logic_used and ui_meta is not None:
                    ui_meta["_prep_method_filter_logic_used"] = prep_method_filter_logic_used

    # --- Flavor preferences (thresholds from Heybo config; same AND/OR structure as Salad) ---
    flavor_preferences = ui.get("FlavorPreferences", {})
    flavor_filter_logic_used = None
    flavor_thresholds = (heybo_cfg or {}).get("flavor_thresholds") or {}
    if flavor_preferences and flavor_thresholds:
        flavor_cols = ["sweet", "sour", "salty", "bitter", "spicy", "umami"]
        available_flavor_cols = [col for col in flavor_cols if col in out.columns]
        if available_flavor_cols:
            active_flavor_count = sum(1 for pref in flavor_preferences.values() if pref and str(pref).strip())
            if active_flavor_count > 1:

                def matches_flavor_preferences_and(row: pd.Series) -> bool:
                    has_valid_preferences = False
                    all_flavors_match = True
                    for flavor, intensity in flavor_preferences.items():
                        if not intensity or not str(intensity).strip():
                            continue
                        has_valid_preferences = True
                        flavor_col = str(flavor).lower()
                        if flavor_col not in available_flavor_cols:
                            all_flavors_match = False
                            break
                        flavor_value = pd.to_numeric(row.get(flavor_col), errors="coerce")
                        if pd.isna(flavor_value):
                            all_flavors_match = False
                            break
                        if flavor_col in flavor_thresholds and intensity in flavor_thresholds[flavor_col]:
                            min_intensity, max_intensity = flavor_thresholds[flavor_col][intensity]
                            if not (min_intensity <= flavor_value <= max_intensity):
                                all_flavors_match = False
                                break
                        else:
                            all_flavors_match = False
                            break
                    if has_valid_preferences:
                        return all_flavors_match
                    return True

                sig_df_and = out[out.apply(matches_flavor_preferences_and, axis=1)]
                if len(sig_df_and) > 0:
                    out = sig_df_and
                    flavor_filter_logic_used = "AND"
                else:

                    def matches_flavor_preferences_or(row: pd.Series) -> bool:
                        for flavor, intensity in flavor_preferences.items():
                            if not intensity or not str(intensity).strip():
                                continue
                            flavor_col = str(flavor).lower()
                            if flavor_col not in available_flavor_cols:
                                continue
                            flavor_value = pd.to_numeric(row.get(flavor_col), errors="coerce")
                            if pd.isna(flavor_value):
                                continue
                            if flavor_col in flavor_thresholds and intensity in flavor_thresholds[flavor_col]:
                                min_intensity, max_intensity = flavor_thresholds[flavor_col][intensity]
                                if min_intensity <= flavor_value <= max_intensity:
                                    return True
                        has_valid_preferences = any(pref and str(pref).strip() for pref in flavor_preferences.values())
                        if has_valid_preferences:
                            return False
                        return True

                    out = out[out.apply(matches_flavor_preferences_or, axis=1)]
                    flavor_filter_logic_used = "OR"
            else:

                def matches_flavor_preferences(row: pd.Series) -> bool:
                    for flavor, intensity in flavor_preferences.items():
                        if not intensity or not str(intensity).strip():
                            continue
                        flavor_col = str(flavor).lower()
                        if flavor_col not in available_flavor_cols:
                            continue
                        flavor_value = pd.to_numeric(row.get(flavor_col), errors="coerce")
                        if pd.isna(flavor_value):
                            continue
                        if flavor_col in flavor_thresholds and intensity in flavor_thresholds[flavor_col]:
                            min_intensity, max_intensity = flavor_thresholds[flavor_col][intensity]
                            if min_intensity <= flavor_value <= max_intensity:
                                return True
                    has_valid_preferences = any(pref and str(pref).strip() for pref in flavor_preferences.values())
                    if has_valid_preferences:
                        return False
                    return True

                out = out[out.apply(matches_flavor_preferences, axis=1)]

            if flavor_filter_logic_used and ui_meta is not None:
                ui_meta["_flavor_filter_logic_used"] = flavor_filter_logic_used

    # --- Light/Hearty filtering ---
    light = ui.get("Light", False)
    hearty = ui.get("Hearty", False)
    if not isinstance(light, bool):
        light = False
    if not isinstance(hearty, bool):
        hearty = False
    if (light or hearty) and "light_hearty" in out.columns:
        out = out.copy()
        out["light_hearty"] = pd.to_numeric(out["light_hearty"], errors="coerce")
        if light and not hearty:
            out = out[out["light_hearty"].between(1, 1, inclusive="both")]
        elif hearty and not light:
            out = out[out["light_hearty"].between(2, 2, inclusive="both")]
        elif light and hearty:
            out = out[out["light_hearty"].between(1, 2, inclusive="both")]

    return out


def _apply_heybo_bowl_name_fold_filter(sig_df: pd.DataFrame, ui: dict) -> pd.DataFrame:
    """Heybo-only: accent-insensitive BowlName vs menu_name (kotai ↔ KōTai)."""
    out = sig_df
    if out.empty:
        return out
    bn = ui.get("BowlName")
    names: List[str] = []
    if isinstance(bn, list):
        names = [str(x).strip().lower() for x in bn if str(x).strip()]
    elif isinstance(bn, str) and bn.strip():
        names = [bn.strip().lower()]
    if names and "menu_name" in out.columns:
        mn = out["menu_name"].fillna("").astype(str)
        folded_patterns = [_fold_name_for_match(p) for p in names if _fold_name_for_match(p)]
        if folded_patterns:
            out = out[
                mn.apply(lambda n: any(fp in _fold_name_for_match(n) for fp in folded_patterns))
            ]
    return out


def _filter_signature_df(
    sig_df: pd.DataFrame,
    user_input_raw: dict,
    heybo_cfg: Optional[dict] = None,
    user_input_meta: Optional[dict] = None,
) -> pd.DataFrame:
    """Salad-equivalent menu_details filters on raw request + Heybo BowlName fold."""
    out = sig_df
    if out.empty:
        return out
    meta = user_input_meta if isinstance(user_input_meta, dict) else user_input_raw
    out = apply_salad_signature_menu_filters(
        out, user_input_raw, heybo_cfg=heybo_cfg, ui_meta=meta
    )
    out = _apply_heybo_bowl_name_fold_filter(out, user_input_raw)
    return out


def _apply_salad_include_keyword_filter_sig_df(
    sig_df: pd.DataFrame,
    ui: dict,
    df_heybo: pd.DataFrame,
) -> pd.DataFrame:
    """salad.py: ingredient_keywords filter when Include is set and no specific BowlName."""
    if sig_df.empty:
        return sig_df
    specific_bowl_names = ui.get("BowlName", None)
    if specific_bowl_names is not None and not isinstance(specific_bowl_names, list):
        specific_bowl_names = [specific_bowl_names]
    ingredient_keywords = set(ui.get("Include", []))
    ingredient_keywords |= set((ui.get("Ingredients") or {}).get("Include", []))
    if not ingredient_keywords or specific_bowl_names:
        return sig_df
    ingredient_keywords = {str(kw).lower() for kw in ingredient_keywords if str(kw).strip()}
    if not ingredient_keywords:
        return sig_df

    def bowl_has_keyword(row: pd.Series) -> bool:
        raw_csv = row.get("ingredients_list_with_sku")
        skus = _sku_list_from_row(str(raw_csv) if raw_csv is not None else "")
        names: List[str] = []
        for sku in skus:
            ing_row = df_heybo[df_heybo["sku_code"] == sku]
            if ing_row.empty:
                continue
            names.append(str(ing_row.iloc[0]["ingredient_name"]).lower())
        if not names:
            return False
        return all(any(kw in name for name in names) for kw in ingredient_keywords)

    return sig_df[sig_df.apply(bowl_has_keyword, axis=1)]


def generate_heybo_signature_bowls(user_input: dict) -> dict:
    """
    Returns the same top-level shape as salad.generate_signature_bowls for Signatures=True:
    {"signature_bowls": [...], "message_to_user": "..."}
    """
    user_input_raw = dict(user_input)
    user_input = preprocess_heybo_filters(dict(user_input_raw))
    dbg = _sig_debug_enabled(user_input)
    shop_name = user_input.get("ShopName", ["heybo"])
    bowl_type = (user_input.get("BowlType") or user_input.get("Bowl Type") or "bowl").lower()
    heybo_cfg = get_heybo_config(
        (user_input.get("recommend_page_id") or "").strip(),
        (user_input.get("session_id") or "").strip(),
    )

    sig_df = load_heybo_signature_catalog(user_input)
    _sig_dbg(dbg, f"catalog rows loaded: {len(sig_df)}")
    # Filter on raw Price/Diet/…; stamp relax flags onto preprocessed user_input for messaging.
    sig_df = _filter_signature_df(
        sig_df,
        user_input_raw,
        heybo_cfg=heybo_cfg,
        user_input_meta=user_input,
    )
    _sig_dbg(
        dbg,
        f"after BowlName/Price filtering: {len(sig_df)}"
        + (
            f" (price_relaxed={user_input.get('_signature_price_relax_mode')})"
            if user_input.get("_signature_price_relaxed")
            else ""
        ),
    )
    recommend_page_id = (user_input.get("recommend_page_id") or "").strip()
    if recommend_page_id and not sig_df.empty and "menu_name" in sig_df.columns:
        previous = load_heybo_previous_signature_bowl_names(recommend_page_id)
        if previous:
            before = len(sig_df)
            sig_df = sig_df[~sig_df["menu_name"].isin(previous)]
            _sig_dbg(
                dbg,
                f"after previous-page dedupe: {len(sig_df)} (removed {before - len(sig_df)})",
            )
    if sig_df.empty:
        _sig_dbg(dbg, "no candidates after initial filtering")
        user_input["_heybo_signature_empty_reason"] = "no_candidates"
        return {
            "signature_bowls": [],
            "message_to_user": build_heybo_signature_user_message([], user_input),
        }

    try:
        df_heybo = load_heybo_data_from_db(user_input)
    except Exception as e:
        _sig_dbg(dbg, f"ingredients load failed: {e}")
        return {
            "signature_bowls": [],
            "message_to_user": build_heybo_signature_user_message([], user_input, error=str(e)),
        }

    sig_df = _apply_salad_include_keyword_filter_sig_df(sig_df, user_input_raw, df_heybo)
    _sig_dbg(dbg, f"after Salad Include keyword filter: {len(sig_df)}")
    if sig_df.empty:
        _sig_dbg(dbg, "no candidates after Include keyword filter")
        user_input["_heybo_signature_empty_reason"] = "no_candidates"
        return {
            "signature_bowls": [],
            "message_to_user": build_heybo_signature_user_message([], user_input),
        }

    available = get_heybo_available_skus(user_input["location_id"], user_input["location_type"])
    _sig_dbg(dbg, f"available SKUs for location: {len(available)}")
    include_ings, exclude_ings, extra_ings = _merge_signature_ingredient_sets(user_input)
    specific_bowl = _signature_specific_bowl_requested(user_input)
    allergen_filters = [
        str(a).strip().lower() for a in (user_input.get("AllergenFilters") or []) if str(a).strip()
    ]

    max_bowl_weight = heybo_cfg.get("max_bowl_weight", 1000)

    sku_to_image_id = load_heybo_vendor_sku_to_image_id_map()
    _sig_dbg(dbg, f"vendor image map size: {len(sku_to_image_id)}")

    signature_bowls: List[dict] = []
    processed = 0
    skip_reasons: Dict[str, int] = {
        "missing_row_sku": 0,
        "missing_available_skus": 0,
        "excluded_ingredient": 0,
        "empty_after_line_filters": 0,
        "weight_exceeded": 0,
    }

    for _, row in sig_df.iterrows():
        processed += 1
        if len(signature_bowls) >= HEYBO_BOWLS_PER_PAGE:
            break
        menu_name = str(row.get("menu_name") or "Signature").strip()
        row_sku = str(row.get("sku_code") or "").strip()
        if not row_sku:
            skip_reasons["missing_row_sku"] += 1
            _sig_dbg(dbg, f"skip {menu_name}: missing sku_code")
            continue
        raw_csv = row.get("ingredients_list_with_sku")
        skus = _sku_list_from_row(str(raw_csv) if raw_csv is not None else "")
        missing = [s for s in skus if s not in available]
        if missing:
            skip_reasons["missing_available_skus"] += 1
            _sig_dbg(dbg, f"skip {menu_name} ({row_sku}): unavailable SKUs {missing}")
            continue

        names: List[str] = []
        for sku in skus:
            ir = df_heybo[df_heybo["sku_code"] == sku]
            if ir.empty:
                continue
            names.append(str(ir.iloc[0]["ingredient_name"]))

        ingredients_to_remove: List[str] = []
        if exclude_ings:
            matched_remove: List[str] = []
            for ex in exclude_ings:
                for nm in names:
                    nl = nm.lower()
                    if ex == nl or ex in nl or nl in ex:
                        matched_remove.append(ex)
                        break
            if matched_remove and not specific_bowl:
                skip_reasons["excluded_ingredient"] += 1
                _sig_dbg(dbg, f"skip {menu_name} ({row_sku}): excluded ingredient matched (no specific bowl)")
                continue
            ingredients_to_remove = list(dict.fromkeys(matched_remove))

        recipe_name_lower = {n.lower() for n in names}
        include_missing_exact = bool(
            include_ings and any(ing.strip().lower() not in recipe_name_lower for ing in include_ings)
        )

        kept_skus = _recipe_skus_after_exclude_allergen(
            skus, df_heybo, ingredients_to_remove, allergen_filters
        )
        if not kept_skus:
            skip_reasons["empty_after_line_filters"] += 1
            _sig_dbg(dbg, f"skip {menu_name} ({row_sku}): no recipe lines left after exclude/allergen filters")
            continue

        added_include_skus: List[str] = []
        for ing in include_ings:
            if ing.strip().lower() in recipe_name_lower:
                continue
            sku_add = _lookup_first_available_sku_by_name(df_heybo, ing, available)
            if sku_add:
                added_include_skus.append(sku_add)
                _sig_dbg(dbg, f"{menu_name}: include add '{ing}' -> {sku_add}")
            else:
                _sig_dbg(dbg, f"{menu_name}: include not in DB / unavailable: '{ing}'")

        bowl = _empty_heybo_bowl_shell(shop_name, bowl_type, menu_name, row_sku)
        bowl["image_id"] = sku_to_image_id.get(row_sku)

        ordered_skus: List[str] = []
        seen_sku: Set[str] = set()

        def _take_sku(s: str) -> None:
            if s in seen_sku:
                return
            if _append_sku_to_bowl(bowl, df_heybo, s):
                seen_sku.add(s)
                ordered_skus.append(s)

        for s in kept_skus:
            _take_sku(s)
        for s in added_include_skus:
            _take_sku(s)

        extras_added = False
        for exn in extra_ings:
            sku_e = _lookup_first_available_sku_by_name(df_heybo, exn, available)
            if not sku_e:
                _sig_dbg(dbg, f"{menu_name}: extra not found / unavailable: '{exn}'")
                continue
            irx = df_heybo[df_heybo["sku_code"] == sku_e]
            if irx.empty:
                continue
            cat = str(irx.iloc[0].get("category") or "").strip()
            if cat not in HEYBO_BOWL_COMPONENT_KEYS:
                _sig_dbg(dbg, f"{menu_name}: extra '{exn}' category not in bowl keys: {cat!r}")
                continue
            max_c = _extras_max_for_category(heybo_cfg, cat)
            extra_bowl_key = _HEYBO_MAIN_CATEGORY_TO_EXTRA_BOWL_KEY.get(cat)
            # Second portion of something already on the bowl: put it on Extra * only so main
            # lines stay catalog-shaped and tier pricing (set-based kept names) stays correct.
            if sku_e in seen_sku and extra_bowl_key is not None:
                if len(bowl.get(extra_bowl_key, [])) >= max_c:
                    _sig_dbg(
                        dbg,
                        f"{menu_name}: extra portion '{exn}' skipped ({extra_bowl_key!r} at cap {max_c})",
                    )
                    continue
                nm = str(irx.iloc[0].get("ingredient_name") or "").strip()
                if not nm:
                    _sig_dbg(dbg, f"{menu_name}: extra portion '{exn}' missing ingredient name")
                    continue
                bowl[extra_bowl_key].append(nm)
                ordered_skus.append(sku_e)
                extras_added = True
                continue
            if len(bowl.get(cat, [])) >= max_c:
                _sig_dbg(dbg, f"{menu_name}: extra '{exn}' skipped (category {cat!r} at cap {max_c})")
                continue
            # New extra (not yet on bowl): same row family as catalog mapping.
            if _append_sku_to_bowl(bowl, df_heybo, sku_e):
                ordered_skus.append(sku_e)
                seen_sku.add(sku_e)
                extras_added = True
            else:
                _sig_dbg(dbg, f"{menu_name}: extra '{exn}' append failed for SKU {sku_e!r}")

        has_extras = extras_added
        needs_customization = include_missing_exact or bool(ingredients_to_remove) or has_extras
        display_menu = menu_name + (" (Mod)" if needs_customization else "")
        bowl["Bowl Name"] = display_menu
        bowl["customized"] = needs_customization
        bowl["has_extras"] = has_extras

        nutrients, weight_g, _ = _aggregate_from_ingredients(df_heybo, ordered_skus)
        selling = float(pd.to_numeric(row.get("selling_price"), errors="coerce") or 0)
        price_config = heybo_cfg.get("price_config") or {}
        catalog_price_ok = (
            not needs_customization
            and len(ordered_skus) == len(skus)
            and sorted(ordered_skus) == sorted(skus)
        )
        if weight_g <= 0:
            weight_g = float(pd.to_numeric(row.get("amount_g"), errors="coerce") or 0)
        if sum(nutrients.values()) == 0:
            for col in NUTRIENT_COLUMNS:
                if col in row.index:
                    nutrients[col] = float(pd.to_numeric(row.get(col), errors="coerce") or 0)

        bowl["Total Weight"] = round(weight_g) if weight_g else round(float(row.get("amount_g") or 0))
        if catalog_price_ok:
            bowl["Total Cost"] = f"{selling:.2f}"
        else:
            try:
                apply_pricing_tier_split_to_bowl(bowl, df_heybo, price_config)
                total, _ = calculate_heybo_bowl_cost_with_breakdown(
                    bowl, df_heybo, user_input=user_input, price_config=price_config
                )
                bowl["Total Cost"] = f"{total:.2f}"
            except Exception as e:
                _sig_dbg(dbg, f"{menu_name}: pricing fallback to catalog: {e}")
                bowl["Total Cost"] = f"{selling:.2f}"
        bowl["Total Nutrients"] = {k: round(float(v), 2) for k, v in nutrients.items()}
        if bowl["Total Weight"] > max_bowl_weight:
            skip_reasons["weight_exceeded"] += 1
            _sig_dbg(
                dbg,
                f"skip {menu_name} ({row_sku}): weight {bowl['Total Weight']} > {max_bowl_weight}",
            )
            continue
        signature_bowls.append(bowl)
        _sig_dbg(dbg, f"added {menu_name} ({row_sku})")

    _sig_dbg(
        dbg,
        "summary: "
        f"processed={processed}, candidates={len(sig_df)}, returned={len(signature_bowls)}, "
        f"skips={skip_reasons}",
    )

    limited = signature_bowls[:HEYBO_BOWLS_PER_PAGE]
    if not limited:
        user_input["_heybo_signature_empty_reason"] = "no_bowls_passed"
    return {
        "signature_bowls": limited,
        "message_to_user": build_heybo_signature_user_message(limited, user_input),
    }
