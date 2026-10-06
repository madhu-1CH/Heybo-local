"""
CP-SAT search for Heybo BYB bowls that satisfy the current nutrient ranges.

Used as a first-pass finder, and re-invoked when random-loop relaxation levels
step up (price / CO2e / Light-Hearty / nutrients). Random generation continues
if CP-SAT does not fill the request.

Signatures never enter this path (``process_heybo_request``).

Constraints:
  - category slot min/max (normal, with only overflowing Include/Extra
    categories raised to the asked count up to customization max; or
    normal min + customization max when expanding for nutrient mins)
  - current nutrient Min/Max bands; equal Min==Max is strict-then-relax
  - global max bowl weight, with soft overshoot
  - Price Min/Max (BYB base + sum of default-category ai_price; slack from
    ``PRICE_RELAXATION_SLACKS`` — no Salad +$2 / .50/.90 rounding)
  - Sustainable / CO2e Min/Max (honors ``co2_relaxation_level``)
  - Balanced / Light / Hearty DB ranges when those preferences are active
  - incompatible ingredient pairs (hard A+B <= 1)
  - Include / Extra portion counts (Include+Extra of the same name → N copies)
  - Apriori association preference (soft Maximize, random sample per bowl)
  - Cuisine / flavor / prep: true DB-tag matches (not catalog top-up). Flavor
    FEW/MANY uses the requested intensity only. FEW matches are forced into
    every bowl; MANY rotate (including unused sauces). Remaining slots follow
    flavor fallback Maximize: High asked → High then Low then No (and the
    reverse sequences for Low / No). Filler names rotate across bowls.
    Catalog top-up does not turn the preference off. Require ≥1 true match.
  - Exclude (force off)
  - combined Warm+Cold sides minimum (normal CYO)
  - protein–sauce diversity: unique pairs first, then differ-by-1
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Any

import pandas as pd

from .bowl_utils import (
    calculate_heybo_bowl_weight,
    calculate_heybo_total_co2e,
    calculate_heybo_total_nutrients,
)
from .config import CPSAT_SEARCH_WORKERS, dbg_print
from .diet import (
    _heybo_equal_min_max_target_value,
    _heybo_equal_nutrient_point_target_slack,
    _heybo_filter_is_equal_min_max_target,
    _resolve_nutrient_filter_key,
    balanced_diet_nutrient_filters,
    balanced_suppressed_by_nutrient_filters,
    heybo_active_nutrient_filter_keys,
    heybo_meets_balanced_diet,
    meets_heybo_nutrient_filters,
)
from .diet_constants import (
    HEYBO_BOWL_COMPONENT_KEYS,
    HEYBO_EXTRA_CATEGORIES,
    NUTRIENT_COLUMNS,
)
from .light_hearty import (
    evaluate_light_hearty_bowl,
    light_hearty_bowl_criteria_for_cpsat,
    light_hearty_suppressed_by_nutrient_filters,
)
from .pricing import (
    apply_pricing_tier_split_to_bowl,
    calculate_heybo_bowl_cost_with_breakdown,
    _get_heybo_default_price,
)

_NUTRIENT_SCALE = 1000
_NUTRIENT_SPREAD_MIN_SPAN = 8.0
_NUTRIENT_SPREAD_WINDOW_FRAC = 0.35
_CPSAT_EQUAL_POINT_TIGHT_FRAC = 1.0 / 3.0

_WEIGHT_DRIFT_G = 5
_WEIGHT_TRIM_SLACK_G = 80

_CO2E_SCALE = 100
_PRICE_DRIFT_CENTS = 50
_PRICE_RELAXATION_SLACKS = (0.5, 1.0, 2.0, 5.0)
_CO2_RELAXATION_LEVEL_1_MAX = 1.80
_MIN_TOTAL_SIDES_NORMAL = 3

_CPSAT_CATEGORIES = (
    "Bases",
    "Proteins",
    "Warm sides",
    "Cold sides",
    "Dips",
    "Garnish",
    "Sauces",
)

_EXTRA_BUCKET = {
    "Proteins": "Extra Proteins",
    "Warm sides": "Extra Warm sides",
    "Cold sides": "Extra Cold sides",
}


def _map_db_category_to_cpsat(db_cat: str) -> str | None:
    if not db_cat:
        return None
    if db_cat in _CPSAT_CATEGORIES:
        return db_cat
    mapped = HEYBO_EXTRA_CATEGORIES.get(db_cat)
    if mapped in _CPSAT_CATEGORIES:
        return mapped
    return None


def _truthy_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "false").lower() in ("true", "1", "yes", "on")


def _apriori_candidates_by_category(
    apriori_suggestions: list[dict] | None,
    include_list: list[str] | None,
    extra_list: list[str] | None,
) -> dict[str, list[str]]:
    user_set = {
        str(x).strip()
        for x in list(include_list or []) + list(extra_list or [])
        if x and str(x).strip()
    }
    if not user_set or not apriori_suggestions:
        return {}

    by_cat: dict[str, list[tuple[int, float, float, str]]] = {}
    for suggestion in apriori_suggestions:
        name = str(suggestion.get("ingredient_name") or "").strip()
        if not name or name in user_set:
            continue
        mapped = _map_db_category_to_cpsat(str(suggestion.get("category") or ""))
        if not mapped:
            continue
        try:
            conf = float(suggestion.get("confidence") or 0)
        except (TypeError, ValueError):
            conf = 0.0
        try:
            freq = float(suggestion.get("frequency") or 0)
        except (TypeError, ValueError):
            freq = 0.0
        paired = {
            str(x).strip()
            for x in (suggestion.get("paired_with") or [])
            if x and str(x).strip()
        }
        overlap = len(paired & user_set)
        by_cat.setdefault(mapped, []).append((overlap, conf, freq, name))

    candidates: dict[str, list[str]] = {}
    for cat, rows in by_cat.items():
        rows.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        names: list[str] = []
        seen: set[str] = set()
        for _overlap, _c, _f, name in rows:
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
        if names:
            candidates[cat] = names
    return candidates


def _sample_apriori_preferred(
    candidates: dict[str, list[str]] | None,
    top_n: int = 3,
) -> dict[str, list[str]]:
    """
    Random Apriori boost per category for this bowl.

    Cap is 3, and k may be 0 so a single partner is not locked on every bowl.
    """
    sampled: dict[str, list[str]] = {}
    try:
        cap = max(0, int(top_n))
    except (TypeError, ValueError):
        cap = 3
    for cat, names in (candidates or {}).items():
        uniq = [n for n in names if n]
        if not uniq:
            continue
        k = random.randint(0, min(cap, len(uniq)))
        if k <= 0:
            continue
        sampled[cat] = random.sample(uniq, k)
    return sampled


def _requested_portion_counts(
    include_list: list[str] | None = None,
    extra_list: list[str] | None = None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ing in list(include_list or []) + list(extra_list or []):
        name = str(ing).strip() if ing else ""
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _asked_counts_by_cpsat_category(
    include_list: list[str] | None,
    extra_list: list[str] | None,
    df,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if df is None or getattr(df, "empty", True):
        return counts
    for ingredient in list(include_list or []) + list(extra_list or []):
        name = str(ingredient).strip() if ingredient else ""
        if not name:
            continue
        try:
            rows = df.loc[df["ingredient_name"] == name, "category"]
        except Exception:
            continue
        if rows.empty:
            continue
        mapped = _map_db_category_to_cpsat(str(rows.iloc[0]))
        if not mapped:
            continue
        counts[mapped] = counts.get(mapped, 0) + 1
    return counts


def _heybo_ask_aware_limit(
    cat: str,
    normal_min: int,
    normal_max: int,
    high_limits: dict,
    requested_count_by_category: dict,
) -> tuple[int, int]:
    hmin, hmax = high_limits.get(cat, (normal_min, normal_max))
    try:
        hmax_i = int(hmax)
    except (TypeError, ValueError):
        hmax_i = int(normal_max)
    req = int((requested_count_by_category or {}).get(cat, 0) or 0)
    eff_max = min(hmax_i, max(int(normal_max), req))
    eff_min = int(normal_min)
    if eff_min > eff_max:
        eff_min = eff_max
    return (eff_min, eff_max)


def _category_limits(
    normal_flow_category_limits: dict,
    customization_flow_extra_category_max: dict,
    omit_categories: set[str] | None,
    expand_to_customization_max: bool = False,
    include_list: list[str] | None = None,
    extra_list: list[str] | None = None,
    df=None,
    variety_bowl: bool = False,
) -> dict[str, tuple[int, int]]:
    asked = {} if variety_bowl else _asked_counts_by_cpsat_category(include_list, extra_list, df)
    high = customization_flow_extra_category_max or {}
    limits: dict[str, tuple[int, int]] = {}
    for cat in _CPSAT_CATEGORIES:
        if omit_categories and cat in omit_categories:
            limits[cat] = (0, 0)
            continue
        pair = (normal_flow_category_limits or {}).get(cat, (0, 1))
        try:
            nmin, nmax = int(pair[0]), int(pair[1])
        except (TypeError, ValueError, IndexError):
            nmin, nmax = 0, 1
        if expand_to_customization_max:
            hpair = high.get(cat, (nmin, nmax))
            try:
                hmax = int(hpair[1])
            except (TypeError, ValueError, IndexError):
                hmax = nmax
            limits[cat] = (nmin, max(nmax, hmax))
        elif not variety_bowl and asked.get(cat, 0) > nmax:
            limits[cat] = _heybo_ask_aware_limit(cat, nmin, nmax, high, asked)
        else:
            limits[cat] = (nmin, nmax)
    return limits


def _only_exact_category_limits(
    customization_category_limits: dict | None,
    include_list: list[str] | None,
    extra_list: list[str] | None,
    df,
    omit_categories: set[str] | None,
) -> tuple[dict[str, tuple[int, int]], list[str]]:
    asked = _asked_counts_by_cpsat_category(include_list, extra_list, df)
    cust = customization_category_limits or {}
    limits: dict[str, tuple[int, int]] = {}
    notices: list[str] = []
    for cat in _CPSAT_CATEGORIES:
        if omit_categories and cat in omit_categories:
            limits[cat] = (0, 0)
            continue
        pair = cust.get(cat)
        try:
            cmin, cmax = int(pair[0]), int(pair[1])
        except (TypeError, ValueError, IndexError):
            cmin, cmax = 0, max(asked.get(cat, 0), 1)
        n_asked = int(asked.get(cat, 0) or 0)
        target = max(n_asked, cmin) if cmin > 0 else n_asked
        if cmax > 0:
            target = min(target, cmax)
        elif n_asked > 0:
            target = 0
        if n_asked > cmax > 0:
            notices.append(
                f"{cat}: {n_asked} requested, limited to customization limit of {cmax}"
            )
        limits[cat] = (target, target)
    return limits, notices


def _trim_portion_req_to_category_caps(
    portion_req: dict[str, int],
    include_list: list[str] | None,
    extra_list: list[str] | None,
    df,
    limits: dict[str, tuple[int, int]],
) -> dict[str, int]:
    if not portion_req:
        return portion_req
    asked_order: list[str] = []
    seen: set[str] = set()
    for ing in list(include_list or []) + list(extra_list or []):
        name = str(ing).strip() if ing else ""
        if not name or name in seen:
            continue
        seen.add(name)
        asked_order.append(name)
    remaining = dict(portion_req)
    kept: dict[str, int] = {}
    used_by_cat: dict[str, int] = {cat: 0 for cat in _CPSAT_CATEGORIES}
    for name in asked_order:
        req = int(remaining.get(name, 0) or 0)
        if req <= 0:
            continue
        rows = df.loc[df["ingredient_name"] == name, "category"] if df is not None else None
        mapped = None
        if rows is not None and not rows.empty:
            mapped = _map_db_category_to_cpsat(str(rows.iloc[0]))
        if not mapped:
            kept[name] = req
            continue
        cap = int((limits.get(mapped) or (0, req))[1] or 0)
        room = max(0, cap - used_by_cat.get(mapped, 0))
        take = min(req, room)
        if take > 0:
            kept[name] = take
            used_by_cat[mapped] = used_by_cat.get(mapped, 0) + take
    return kept


def _ingredient_nutrient_scaled(df, ingredient_name: str, nutrient_key: str) -> int:
    rows = df.loc[df["ingredient_name"] == ingredient_name, nutrient_key]
    if rows.empty:
        return 0
    val = pd.to_numeric(rows.iloc[0], errors="coerce")
    if pd.isna(val):
        return 0
    return int(round(float(val) * _NUTRIENT_SCALE))


def _rounding_drift_units(limits: dict[str, tuple[int, int]]) -> int:
    slot_cap = sum(int(cmax) for _cmin, cmax in (limits or {}).values())
    return max(5, (slot_cap + 1) // 2)


def _scaled_nutrient_bounds(min_raw, max_raw, drift: int) -> tuple[int | None, int | None]:
    scaled_min = None
    scaled_max = None
    if min_raw is not None:
        scaled_min = int(round(float(min_raw) * _NUTRIENT_SCALE)) - drift
    if max_raw is not None:
        scaled_max = int(round(float(max_raw) * _NUTRIENT_SCALE)) + drift
    return scaled_min, scaled_max


def _cpsat_filter_is_equal_point(nf: dict | None) -> bool:
    if not isinstance(nf, dict):
        return False
    if nf.get("_equal_point_target_opened") or nf.get("_equal_min_max_target"):
        return True
    try:
        return bool(_heybo_filter_is_equal_min_max_target(nf))
    except Exception:
        return False


def _cpsat_equal_point_tight_slack(nutrient_key: str) -> float:
    full = float(_heybo_equal_nutrient_point_target_slack(nutrient_key))
    if full <= 0:
        return 0.0
    tight = full * _CPSAT_EQUAL_POINT_TIGHT_FRAC
    return min(full, max(tight, min(5.0, full)))


def _equal_point_strict_nutrient_windows(
    nutrient_filters: list[dict] | None,
) -> dict[str, tuple[float, float]]:
    overrides: dict[str, tuple[float, float]] = {}
    for nf in nutrient_filters or []:
        if not _cpsat_filter_is_equal_point(nf):
            continue
        key = _resolve_nutrient_filter_key(str(nf.get("Nutrient") or "").strip())
        if not key:
            continue
        target = _heybo_equal_min_max_target_value(nf)
        if target is None:
            rng = nf.get("Range") or {}
            try:
                target = float(rng.get("Min"))
            except (TypeError, ValueError):
                continue
        overrides[key] = (float(target), float(target))
    return overrides


def _min_anchored_nutrient_windows(
    nutrient_filters: list[dict] | None,
) -> dict[str, tuple[float, float]]:
    overrides: dict[str, tuple[float, float]] = {}
    for nf in nutrient_filters or []:
        key = _resolve_nutrient_filter_key(str(nf.get("Nutrient") or "").strip())
        if not key:
            continue
        rng = nf.get("Range") or {}
        min_raw, max_raw = rng.get("Min"), rng.get("Max")
        if min_raw is None or max_raw is None:
            continue
        try:
            lo, hi = float(min_raw), float(max_raw)
        except (TypeError, ValueError):
            continue
        if hi <= lo + 1e-6:
            continue
        if _cpsat_filter_is_equal_point(nf):
            tight_hi = min(hi, lo + _cpsat_equal_point_tight_slack(key))
        else:
            span = hi - lo
            if span < _NUTRIENT_SPREAD_MIN_SPAN:
                continue
            window = max(span * _NUTRIENT_SPREAD_WINDOW_FRAC, min(5.0, span))
            window = min(window, span)
            tight_hi = lo + window
        if tight_hi >= hi - 1e-6 or tight_hi <= lo + 1e-6:
            continue
        overrides[key] = (lo, tight_hi)
    return overrides


def _random_category_count_targets(
    limits: dict[str, tuple[int, int]],
) -> dict[str, tuple[int, int]]:
    overrides: dict[str, tuple[int, int]] = {}
    for cat, (cmin, cmax) in (limits or {}).items():
        lo, hi = int(cmin), int(cmax)
        if hi <= lo:
            continue
        n = random.randint(lo, hi)
        overrides[cat] = (n, n)
    return overrides


def _cuisine_filters_requested(user_input: dict | None) -> bool:
    ui = user_input or {}
    if ui.get("_relax_cuisine_filter"):
        return False
    cf = ui.get("CuisineFilters") or {}
    if isinstance(cf, dict):
        return any(v not in (None, False, 0, "0") and str(v or "").strip() for v in cf.values())
    return bool(cf)


def _cuisine_filters_active(user_input: dict | None) -> bool:
    ui = user_input or {}
    if ui.get("_relax_cuisine_filter") or ui.get("cuisine_relaxation_enabled"):
        return False
    return _cuisine_filters_requested(user_input)


def _cuisine_exclude_active(user_input: dict | None) -> bool:
    ui = user_input or {}
    if ui.get("_relax_cuisine_filter") or ui.get("cuisine_relaxation_enabled"):
        return False
    cf = ui.get("CuisineFilters") or {}
    if not isinstance(cf, dict):
        return False
    return any(v is False for v in cf.values())


def _flavor_preferences_requested(
    user_input: dict | None,
    flavor_preferences: dict | None = None,
) -> bool:
    """User asked for a flavor intensity. True even after catalog top-up."""
    ui = user_input or {}
    if ui.get("_relax_flavor_filter"):
        return False
    fp = flavor_preferences if flavor_preferences is not None else ui.get("FlavorPreferences") or {}
    if not isinstance(fp, dict):
        return False
    return any(str(v or "").strip() for v in fp.values() if v)


def _flavor_preferences_active(
    user_input: dict | None,
    flavor_preferences: dict | None = None,
) -> bool:
    ui = user_input or {}
    if ui.get("_relax_flavor_filter") or ui.get("flavor_relaxation_enabled"):
        return False
    return _flavor_preferences_requested(user_input, flavor_preferences)


def _prep_method_requested(user_input: dict | None) -> bool:
    ui = user_input or {}
    if ui.get("_relax_preparation_method_filter"):
        return False
    pm = ui.get("PreparationMethod") or {}
    if not isinstance(pm, dict):
        return bool(pm)
    return any(v not in (None, False, 0, "0") and str(v or "").strip() for v in pm.values())


def _prep_method_active(user_input: dict | None) -> bool:
    ui = user_input or {}
    if ui.get("_relax_preparation_method_filter") or ui.get("prep_relaxation_enabled"):
        return False
    return _prep_method_requested(user_input)


def _prep_exclude_active(user_input: dict | None) -> bool:
    ui = user_input or {}
    if ui.get("_relax_preparation_method_filter") or ui.get("prep_relaxation_enabled"):
        return False
    pm = ui.get("PreparationMethod") or {}
    if not isinstance(pm, dict):
        return False
    return any(v is False for v in pm.values())


def _parse_cuisine_tags(raw) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if hasattr(raw, "iloc"):
        raw = raw.iloc[0] if not raw.empty else []
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(c).strip().lower() for c in raw if str(c).strip()]
    return [c.strip().lower() for c in str(raw).split(",") if c.strip()]


def _matches_from_user_input_by_cat(
    user_input: dict | None,
    key: str,
    pools: dict[str, list[str]],
    exclude_set: set[str],
) -> dict[str, list[str]]:
    raw = (user_input or {}).get(key) or {}
    if not isinstance(raw, dict):
        return {}
    pool_sets = {cat: set(names or []) for cat, names in (pools or {}).items()}
    out: dict[str, list[str]] = {}
    for cat, names in raw.items():
        mapped = _map_db_category_to_cpsat(str(cat or "")) or str(cat or "")
        if mapped not in pool_sets:
            continue
        for n in names or []:
            name = str(n).strip()
            if not name or name in exclude_set or name not in pool_sets[mapped]:
                continue
            bucket = out.setdefault(mapped, [])
            if name not in bucket:
                bucket.append(name)
    return out


def _true_cuisine_matches_by_cpsat_category(
    df,
    user_input: dict | None,
    pools: dict[str, list[str]],
    exclude_set: set[str],
) -> dict[str, list[str]]:
    stashed = _matches_from_user_input_by_cat(
        user_input, "_heybo_true_cuisine_by_category", pools, exclude_set
    )
    if stashed:
        return stashed
    if not _cuisine_filters_requested(user_input) or df is None or getattr(df, "empty", True):
        return {}
    from .filters import _ingredient_cuisine_tokens, _parse_cuisine_include_exclude, _row_matches_salad_cuisine

    included, excluded = _parse_cuisine_include_exclude(user_input or {})
    if not included:
        return {}
    pool_sets = {cat: set(names or []) for cat, names in (pools or {}).items()}
    out: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        name = str(row.get("ingredient_name") or "").strip()
        if not name or name in exclude_set:
            continue
        tokens = _ingredient_cuisine_tokens(row.get("cuisine"))
        if not _row_matches_salad_cuisine(tokens, included, excluded):
            continue
        mapped = _map_db_category_to_cpsat(str(row.get("category") or ""))
        if not mapped or name not in pool_sets.get(mapped, set()):
            continue
        bucket = out.setdefault(mapped, [])
        if name not in bucket:
            bucket.append(name)
    return out


def _true_prep_matches_by_cpsat_category(
    df,
    user_input: dict | None,
    pools: dict[str, list[str]],
    exclude_set: set[str],
) -> dict[str, list[str]]:
    stashed = _matches_from_user_input_by_cat(
        user_input, "_heybo_true_prep_by_category", pools, exclude_set
    )
    if stashed:
        return stashed
    if not _prep_method_requested(user_input) or df is None or getattr(df, "empty", True):
        return {}
    from .filters import _heybo_row_matches_preparation_method_salad_style

    pm = (user_input or {}).get("PreparationMethod") or {}
    included = [str(k).strip() for k, v in pm.items() if v is True and str(k).strip()]
    excluded = [str(k).strip() for k, v in pm.items() if v is False and str(k).strip()]
    if not included:
        return {}
    pool_sets = {cat: set(names or []) for cat, names in (pools or {}).items()}
    out: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        name = str(row.get("ingredient_name") or "").strip()
        if not name or name in exclude_set:
            continue
        if not _heybo_row_matches_preparation_method_salad_style(
            row.get("preparation_method"), name, included, excluded
        ):
            continue
        mapped = _map_db_category_to_cpsat(str(row.get("category") or ""))
        if not mapped or name not in pool_sets.get(mapped, set()):
            continue
        bucket = out.setdefault(mapped, [])
        if name not in bucket:
            bucket.append(name)
    return out


_FLAVOR_SCORE_COLS = frozenset({"sweet", "sour", "salty", "bitter", "spicy", "umami"})
_FLAVOR_FALLBACK_ORDER = {
    "High": ["High", "Low", "No"],
    "Low": ["Low", "High", "No"],
    "No": ["No", "Low", "High"],
}
_FLAVOR_FALLBACK_WEIGHTS = (6, 3, 1)


def _heybo_flavor_thresholds(user_input: dict | None) -> dict:
    return ((user_input or {}).get("_heybo_cfg") or {}).get("flavor_thresholds") or {}


def _flavor_band_for_score(flavor_col: str, value: float, thresholds: dict) -> str | None:
    by_flavor = (thresholds or {}).get(flavor_col) or {}
    for band in ("High", "Low", "No"):
        if band not in by_flavor:
            continue
        try:
            lo, hi = by_flavor[band]
            lo_f, hi_f = float(lo), float(hi)
        except (TypeError, ValueError):
            continue
        if lo_f <= value <= hi_f:
            return band
    return None


def _true_flavor_matches_from_scores(
    df,
    user_input: dict | None,
    flavor_preferences: dict | None,
    pools: dict[str, list[str]],
    exclude_set: set[str],
) -> dict[str, list[str]]:
    """Names in the requested intensity band only (not High→Low→No)."""
    if not _flavor_preferences_requested(user_input, flavor_preferences):
        return {}
    if df is None or getattr(df, "empty", True):
        return {}
    ui = user_input or {}
    fp = flavor_preferences if flavor_preferences is not None else ui.get("FlavorPreferences") or {}
    if not isinstance(fp, dict):
        return {}
    thresholds = _heybo_flavor_thresholds(ui)
    bands: list[tuple[str, float, float]] = []
    for flavor, pref in fp.items():
        flavor_col = str(flavor or "").strip().lower()
        if flavor_col not in _FLAVOR_SCORE_COLS:
            continue
        pref_s = str(pref or "").strip() if not isinstance(pref, dict) else ""
        if not pref_s:
            continue
        by_flavor = thresholds.get(flavor_col) or {}
        if pref_s not in by_flavor:
            continue
        try:
            lo, hi = by_flavor[pref_s]
            bands.append((flavor_col, float(lo), float(hi)))
        except (TypeError, ValueError):
            continue
    if not bands:
        return {}
    pool_sets = {cat: set(names or []) for cat, names in (pools or {}).items()}
    out: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        name = str(row.get("ingredient_name") or "").strip()
        if not name or name in exclude_set:
            continue
        hits = False
        for flavor_col, lo, hi in bands:
            if flavor_col not in row.index:
                continue
            val = pd.to_numeric(row.get(flavor_col), errors="coerce")
            if pd.isna(val) or float(val) < 0:
                continue
            if lo <= float(val) <= hi:
                hits = True
                break
        if not hits:
            continue
        mapped = _map_db_category_to_cpsat(str(row.get("category") or ""))
        if not mapped or name not in pool_sets.get(mapped, set()):
            continue
        if name in seen.setdefault(mapped, set()):
            continue
        seen[mapped].add(name)
        out.setdefault(mapped, []).append(name)
    return out


def _flavor_matches_by_cpsat_category(
    df,
    user_input: dict | None,
    flavor_preferences: dict | None,
    pools: dict[str, list[str]],
    exclude_set: set[str],
) -> dict[str, list[str]]:
    stashed = _matches_from_user_input_by_cat(
        user_input, "_heybo_true_flavor_by_category", pools, exclude_set
    )
    if stashed:
        return stashed
    return _true_flavor_matches_from_scores(
        df, user_input, flavor_preferences, pools, exclude_set
    )


def _flavor_fallback_rank_by_name(
    df,
    user_input: dict | None,
    pools: dict[str, list[str]],
    exclude_set: set[str],
    flavor_preferences: dict | None = None,
) -> dict[str, int]:
    """Ingredient → fallback rank (0 = requested intensity). High asked → High=0, Low=1, No=2."""
    if not _flavor_preferences_requested(user_input, flavor_preferences):
        return {}
    if df is None or getattr(df, "empty", True):
        return {}
    ui = user_input or {}
    fp = flavor_preferences if flavor_preferences is not None else ui.get("FlavorPreferences") or {}
    if not isinstance(fp, dict):
        return {}
    thresholds = _heybo_flavor_thresholds(ui)
    specs: list[tuple[str, list[str]]] = []
    for flavor, pref in fp.items():
        flavor_col = str(flavor or "").strip().lower()
        if flavor_col not in _FLAVOR_SCORE_COLS:
            continue
        pref_s = str(pref or "").strip() if not isinstance(pref, dict) else ""
        if not pref_s or pref_s not in _FLAVOR_FALLBACK_ORDER:
            continue
        specs.append((flavor_col, list(_FLAVOR_FALLBACK_ORDER[pref_s])))
    if not specs:
        return {}
    pool_all: set[str] = set()
    for names in (pools or {}).values():
        pool_all.update(names or [])
    out: dict[str, int] = {}
    for _, row in df.iterrows():
        name = str(row.get("ingredient_name") or "").strip()
        if not name or name in exclude_set or name not in pool_all:
            continue
        best: int | None = None
        for flavor_col, sequence in specs:
            if flavor_col not in row.index:
                continue
            val = pd.to_numeric(row.get(flavor_col), errors="coerce")
            if pd.isna(val) or float(val) < 0:
                continue
            band = _flavor_band_for_score(flavor_col, float(val), thresholds)
            if band is None:
                continue
            try:
                rank = sequence.index(band)
            except ValueError:
                continue
            if best is None or rank < best:
                best = rank
        if best is not None:
            out[name] = best
    return out


def _few_match_names_to_force(
    matches_by_cat: dict[str, list[str]],
    limits: dict[str, tuple[int, int]],
    already_forced: set[str],
    ingredient_category: dict[str, str],
    label: str,
) -> set[str]:
    """FEW (≤ max × 1.5) → force into every bowl; MANY → rotate."""
    forced: set[str] = set()
    forced_count_by_cat: dict[str, int] = {}
    for name in already_forced:
        cat = ingredient_category.get(name)
        if cat:
            forced_count_by_cat[cat] = forced_count_by_cat.get(cat, 0) + 1
    for cat, names in (matches_by_cat or {}).items():
        uniq = list(dict.fromkeys(n for n in names if n))
        if not uniq:
            continue
        try:
            cmax_i = int((limits.get(cat) or (0, 1))[1])
        except (TypeError, ValueError):
            cmax_i = 1
        if cmax_i < 1:
            continue
        if len(uniq) > cmax_i * 1.5:
            dbg_print(
                f"[CP-SAT] {cat}: MANY {label} matches ({len(uniq)} > {cmax_i * 1.5:.1f}); "
                "rotating across bowls"
            )
            continue
        room = cmax_i - forced_count_by_cat.get(cat, 0)
        to_force = [n for n in uniq if n not in already_forced and n not in forced][: max(0, room)]
        forced.update(to_force)
        if to_force:
            forced_count_by_cat[cat] = forced_count_by_cat.get(cat, 0) + len(to_force)
            dbg_print(
                f"[CP-SAT] {cat}: FEW {label} matches ({len(uniq)} ≤ {cmax_i * 1.5:.1f}); "
                f"forcing in every bowl: {', '.join(to_force)}"
            )
    return forced


def _raise_protein_min_when_available(
    limits: dict[str, tuple[int, int]],
    protein_pool: list[str] | None,
    only_exact_bowl: bool,
) -> dict[str, tuple[int, int]]:
    if only_exact_bowl or not protein_pool:
        return limits
    cmin, cmax = limits.get("Proteins", (0, 1))
    try:
        cmin_i, cmax_i = int(cmin), int(cmax)
    except (TypeError, ValueError):
        return limits
    if cmax_i < 1 or cmin_i >= 1:
        return limits
    out = dict(limits)
    out["Proteins"] = (1, cmax_i)
    dbg_print(
        f"[CP-SAT] protein min raised {cmin_i}→1 "
        f"(pool has {len(protein_pool)} protein(s); DB min is 0)"
    )
    return out


def _excluded_cuisine_ingredient_names(df, user_input: dict | None) -> set[str]:
    if not _cuisine_exclude_active(user_input):
        return set()
    cf = (user_input or {}).get("CuisineFilters") or {}
    excluded = {str(k).strip().lower() for k, v in cf.items() if v is False and str(k).strip()}
    if not excluded or df is None or getattr(df, "empty", True):
        return set()
    if "ingredient_name" not in df.columns:
        return set()
    names: set[str] = set()
    cuisine_col = "cuisine" if "cuisine" in df.columns else None
    if not cuisine_col:
        return set()
    for _, row in df.iterrows():
        tags = set(_parse_cuisine_tags(row.get(cuisine_col)))
        if tags & excluded:
            name = str(row.get("ingredient_name") or "").strip()
            if name:
                names.add(name)
    return names


def _excluded_prep_ingredient_names(df, user_input: dict | None) -> set[str]:
    if not _prep_exclude_active(user_input):
        return set()
    pm = (user_input or {}).get("PreparationMethod") or {}
    excluded = {str(k).strip().lower() for k, v in pm.items() if v is False and str(k).strip()}
    if not excluded or df is None or getattr(df, "empty", True):
        return set()
    if "ingredient_name" not in df.columns or "preparation_method" not in df.columns:
        return set()
    names: set[str] = set()
    for _, row in df.iterrows():
        method = str(row.get("preparation_method") or "").strip().lower()
        if method in excluded:
            name = str(row.get("ingredient_name") or "").strip()
            if name:
                names.add(name)
    return names


def _flavor_matched_names(df, user_input: dict | None, flavor_preferences: dict | None) -> set[str]:
    ui = user_input or {}
    fp = flavor_preferences if flavor_preferences is not None else ui.get("FlavorPreferences") or {}
    if not _flavor_preferences_requested(ui, fp) or df is None or getattr(df, "empty", True):
        return set()
    heybo_cfg = ui.get("_heybo_cfg") or {}
    flavor_thresholds = (heybo_cfg or {}).get("flavor_thresholds") or {}
    fallback_order = {
        "High": ["High", "Low", "No"],
        "Low": ["Low", "High", "No"],
        "No": ["No", "Low", "High"],
    }
    names: set[str] = set()
    for _, row in df.drop_duplicates(subset=["ingredient_name"]).iterrows():
        matched = False
        for flavor, intensity in (fp or {}).items():
            if not intensity or (isinstance(intensity, str) and not intensity.strip()):
                continue
            intensity = str(intensity).strip()
            flavor_col = str(flavor).lower()
            thresholds_for_flavor = flavor_thresholds.get(flavor_col)
            if flavor_col not in row.index or not thresholds_for_flavor or intensity not in fallback_order:
                continue
            try:
                score = float(pd.to_numeric(row.get(flavor_col), errors="coerce") or 0)
            except (TypeError, ValueError):
                continue
            for attempt in fallback_order[intensity]:
                if attempt not in thresholds_for_flavor:
                    continue
                mn, mx = thresholds_for_flavor[attempt]
                if mn <= score <= mx:
                    matched = True
                    break
            if matched:
                break
        if matched:
            name = str(row.get("ingredient_name") or "").strip()
            if name:
                names.add(name)
    return names


def _prep_matched_names(df, user_input: dict | None) -> set[str]:
    if not _prep_method_requested(user_input):
        return set()
    pm = (user_input or {}).get("PreparationMethod") or {}
    included = {str(k).strip().lower() for k, v in pm.items() if v is True and str(k).strip()}
    if not included or df is None or getattr(df, "empty", True):
        return set()
    if "ingredient_name" not in df.columns or "preparation_method" not in df.columns:
        return set()
    names: set[str] = set()
    for _, row in df.iterrows():
        method = str(row.get("preparation_method") or "").strip().lower()
        if method in included:
            name = str(row.get("ingredient_name") or "").strip()
            if name:
                names.add(name)
    return names


def _companion_match_name_sets(
    user_input: dict | None,
    df,
    flavor_preferences: dict | None = None,
    cuisine_matched_names: set[str] | list[str] | None = None,
) -> tuple[set[str], set[str], set[str]]:
    ui = user_input or {}
    cuisine_names: set[str] = set()
    if _cuisine_filters_requested(ui):
        raw = cuisine_matched_names if cuisine_matched_names is not None else ui.get("_cuisine_matched_ingredient_names")
        if raw:
            cuisine_names.update(str(x).strip() for x in raw if x)
    flavor_names = _flavor_matched_names(df, ui, flavor_preferences)
    prep_names = _prep_matched_names(df, ui)
    return cuisine_names, flavor_names, prep_names


def _balanced_is_active(user_input: dict | None) -> bool:
    ui = user_input or {}
    if not _truthy_flag(ui.get("Balanced", False)):
        return False
    if ui.get("_balanced_suppressed_by_nutrients"):
        return False
    return not balanced_suppressed_by_nutrient_filters(ui.get("NutrientFilters") or [])


def build_cpsat_filters_from_preferences(
    user_input: dict | None,
    existing_nutrient_filters: list[dict] | None = None,
) -> tuple[list[dict], tuple[float, float] | None, list[str]]:
    """Merge NutrientFilters with Balanced / Light / Hearty DB criteria."""
    ui = user_input or {}
    existing = list(existing_nutrient_filters or [])
    active = set(heybo_active_nutrient_filter_keys(existing))
    merged = list(existing)
    sources: list[str] = []
    if existing:
        sources.append("NutrientFilters")

    weight_range = None

    if _balanced_is_active(ui):
        bal = balanced_diet_nutrient_filters(
            relaxation_level=int(ui.get("balanced_relaxation_level") or 0)
        )
        added = 0
        for nf in bal:
            key = _resolve_nutrient_filter_key(str(nf.get("Nutrient") or ""))
            if not key or key in active:
                continue
            merged.append(nf)
            active.add(key)
            added += 1
        if added:
            sources.append("Balanced")

    light = _truthy_flag(ui.get("Light", False))
    hearty = _truthy_flag(ui.get("Hearty", False))
    lh_suppressed = bool(ui.get("_light_hearty_suppressed_by_nutrients")) or (
        light_hearty_suppressed_by_nutrient_filters(ui.get("NutrientFilters") or [])
    )
    if (light or hearty) and not (light and hearty) and not lh_suppressed:
        lh_filters, weight_range = light_hearty_bowl_criteria_for_cpsat(
            light,
            hearty,
            relaxation_level=int(ui.get("light_hearty_relaxation_level") or 0),
            skip_nutrient_keys=active,
        )
        added = 0
        for nf in lh_filters:
            key = _resolve_nutrient_filter_key(str(nf.get("Nutrient") or ""))
            if not key or key in active:
                continue
            merged.append(nf)
            active.add(key)
            added += 1
        if added or weight_range is not None:
            sources.append("Light" if light and not hearty else "Hearty")

    price = ui.get("Price") or {}
    if isinstance(price, dict) and (price.get("Min") is not None or price.get("Max") is not None):
        sources.append("Price")
    if _truthy_flag(ui.get("Sustainable", False)):
        sources.append("Sustainable")
    if _cuisine_filters_active(ui) or _cuisine_exclude_active(ui):
        sources.append("Cuisine")
    if _flavor_preferences_active(ui):
        sources.append("Flavor")
    if _prep_method_active(ui) or _prep_exclude_active(ui):
        sources.append("Prep")
    ings = ui.get("Ingredients") or {}
    if isinstance(ings, dict) and (ings.get("Include") or ings.get("Extra") or ings.get("Exclude")):
        sources.append("Ingredients")
    if ui.get("AllergenFilters"):
        sources.append("Allergen")
    if ui.get("DietFilters"):
        sources.append("Diet")

    return merged, weight_range, sources


def _ingredient_weight_g(df, ingredient_name: str) -> int:
    rows = df.loc[df["ingredient_name"] == ingredient_name, "serving_amount_per_portion_in_grams"]
    if rows.empty:
        return 0
    val = pd.to_numeric(rows.iloc[0], errors="coerce")
    if pd.isna(val):
        return 0
    return max(0, int(round(float(val))))


def _ingredient_price_cents(df, ingredient_name: str, category: str) -> int:
    p = _get_heybo_default_price(ingredient_name, df, category)
    return max(0, int(round(float(p) * 100)))


def _ingredient_co2e_scaled(df, ingredient_name: str) -> int:
    rows = df.loc[df["ingredient_name"] == ingredient_name, "co2e_values_per_serving"]
    if rows.empty:
        return 0
    val = pd.to_numeric(rows.iloc[0], errors="coerce")
    if pd.isna(val):
        return 0
    return max(0, int(round(float(val) * _CO2E_SCALE)))


def _effective_price_bounds(user_input: dict) -> tuple[float | None, float | None]:
    price_filter = user_input.get("Price") or {}
    if not isinstance(price_filter, dict):
        return None, None
    min_p = price_filter.get("Min")
    max_p = price_filter.get("Max")
    if min_p is None and max_p is None:
        return None, None
    a = float(min_p) if min_p is not None else None
    b = float(max_p) if max_p is not None else None
    idx = min(int(user_input.get("price_relaxation_level") or 0), len(_PRICE_RELAXATION_SLACKS) - 1)
    slack = _PRICE_RELAXATION_SLACKS[idx]
    if a is not None and b is not None and round(a, 2) == round(b, 2):
        return a - slack, a + slack
    lo, hi = a, b
    if lo is not None:
        lo -= slack
    if hi is not None:
        hi += slack
    return lo, hi


def _cpsat_ingredient_price_sum_bounds(
    user_input: dict,
    drift_cents: int = _PRICE_DRIFT_CENTS,
) -> tuple[int | None, int | None]:
    """Bounds on sum(default ai_price) in cents, after subtracting BYB base price."""
    lo, hi = _effective_price_bounds(user_input)
    if lo is None and hi is None:
        return None, None
    heybo_cfg = user_input.get("_heybo_cfg") or {}
    base = float((heybo_cfg.get("price_config") or {}).get("base_price") or 0.0)
    min_sum: int | None = None
    max_sum: int | None = None
    if lo is not None:
        min_sum = max(0, int(round((float(lo) - base) * 100)) - drift_cents)
    if hi is not None and hi != float("inf"):
        max_sum = max(0, int(round((float(hi) - base) * 100)) + drift_cents)
    return min_sum, max_sum


def _heybo_price_headroom_dollars(user_input: dict) -> float | None:
    """Ingredient spend allowed after BYB min. None = no Price Max."""
    _lo, hi = _effective_price_bounds(user_input)
    if hi is None or hi == float("inf"):
        return None
    heybo_cfg = user_input.get("_heybo_cfg") or {}
    base = float((heybo_cfg.get("price_config") or {}).get("base_price") or 0.0)
    return max(0.0, float(hi) - base)


def _heybo_tight_floor_price(user_input: dict) -> bool:
    """Max ≈ BYB min: Extra * SKU overflow cannot fit (same rule as random joint-fill)."""
    if int((user_input or {}).get("price_relaxation_level") or 0) > 0:
        return False
    headroom = _heybo_price_headroom_dollars(user_input)
    return headroom is not None and headroom < 1.0


def _heybo_price_filter_active(user_input: dict) -> bool:
    lo, hi = _effective_price_bounds(user_input)
    return lo is not None or (hi is not None and hi != float("inf"))


def _heybo_pricing_tier_slots(user_input: dict) -> tuple[int, int]:
    heybo_cfg = (user_input or {}).get("_heybo_cfg") or {}
    pricing = ((heybo_cfg.get("price_config") or {}).get("pricing_limits") or {})
    try:
        prot = int((pricing.get("Proteins") or (0, 1))[1] or 1)
    except (TypeError, ValueError, IndexError):
        prot = 1
    try:
        sides = int((pricing.get("Sides") or (0, 3))[1] or 3)
    except (TypeError, ValueError, IndexError):
        sides = 3
    return max(1, prot), max(0, sides)


def _apply_heybo_tight_price_slot_caps(
    user_input: dict,
    limits: dict[str, tuple[int, int]],
) -> dict[str, tuple[int, int]]:
    """Keep Proteins / Warm+Cold inside free Pricing tiers when Max ≈ BYB min.

    SAT prices default-category ai_price (often $0 for Proteins). Companion
    pricing charges Extra Proteins / Extra sides SKUs after tier split, so
    unconstrained extra slots produce a flood of 'price above max' rejects.
    """
    if not _heybo_tight_floor_price(user_input):
        return limits
    tier_prot, tier_sides = _heybo_pricing_tier_slots(user_input)
    out = dict(limits)
    if "Proteins" in out:
        pmin, pmax = out["Proteins"]
        try:
            pmin_i, pmax_i = int(pmin), int(pmax)
        except (TypeError, ValueError):
            pmin_i, pmax_i = 0, 1
        # Same as generation joint-fill: keep pmin, clamp pmax to Pricing default slots.
        out["Proteins"] = (pmin_i, min(pmax_i, max(1, tier_prot)))
    wmin, wmax = out.get("Warm sides", (0, 0))
    cmin, cmax = out.get("Cold sides", (0, 0))
    try:
        wmax_i, cmax_i = int(wmax), int(cmax)
    except (TypeError, ValueError):
        dbg_print(
            f"[CP-SAT] tight price: cap Proteins max={out.get('Proteins', (0, 1))[1]} "
            "(no Extra protein overflow)"
        )
        return out
    # Prefer max cold (often high protein) then fill remaining tier with warm.
    c_keep = min(cmax_i, tier_sides)
    w_keep = min(wmax_i, max(0, tier_sides - c_keep))
    out["Cold sides"] = (cmin, c_keep)
    out["Warm sides"] = (wmin, w_keep)
    dbg_print(
        f"[CP-SAT] tight price: cap Proteins max={out['Proteins'][1]}, "
        f"Warm max={w_keep}, Cold max={c_keep} (no Extra SKU overflow)"
    )
    return out


def _co2e_bounds_for_user(user_input: dict) -> tuple[float | None, float | None]:
    if not _truthy_flag(user_input.get("Sustainable", False)):
        return None, None
    heybo_cfg = user_input.get("_heybo_cfg") or {}
    cfg = (heybo_cfg.get("co2_config") or {})
    rng = cfg.get("Sustainable") or {}
    lo = pd.to_numeric(rng.get("min"), errors="coerce")
    hi = pd.to_numeric(rng.get("max"), errors="coerce")
    if pd.isna(lo) or pd.isna(hi):
        return None, None
    level = int(user_input.get("co2_relaxation_level") or 0)
    if level <= 0:
        return float(lo), float(hi)
    if level == 1:
        return float(lo), _CO2_RELAXATION_LEVEL_1_MAX
    return float(lo), float("inf")


def _cpsat_co2e_scaled_bounds(user_input: dict, drift: int) -> tuple[int | None, int | None]:
    min_co2e, max_co2e = _co2e_bounds_for_user(user_input)
    if min_co2e is None:
        return None, None
    scaled_min = max(0, int(round(min_co2e * _CO2E_SCALE)) - drift)
    if max_co2e == float("inf"):
        return scaled_min, None
    scaled_max = int(round(max_co2e * _CO2E_SCALE)) + drift
    return scaled_min, scaled_max


def _split_user_extras_into_bowl(
    bowl: dict,
    include_list: list[str] | None,
    extra_list: list[str] | None,
    df,
) -> None:
    extra_counts = Counter(str(x).strip() for x in (extra_list or []) if x and str(x).strip())
    include_counts = Counter(str(x).strip() for x in (include_list or []) if x and str(x).strip())
    for main_cat, extra_cat in _EXTRA_BUCKET.items():
        items = list(bowl.get(main_cat) or []) + list(bowl.get(extra_cat) or [])
        if not items:
            bowl[main_cat] = []
            bowl[extra_cat] = []
            continue
        inc_left = Counter({k: include_counts[k] for k in items})
        ext_left = Counter({k: extra_counts[k] for k in items})
        new_main: list[str] = []
        new_extra: list[str] = []
        for ing in items:
            if inc_left.get(ing, 0) > 0:
                inc_left[ing] -= 1
                new_main.append(ing)
            elif ext_left.get(ing, 0) > 0:
                has_extra_row = False
                if df is not None and not getattr(df, "empty", True):
                    m = (df["ingredient_name"] == ing) & (df["category"] == extra_cat)
                    has_extra_row = bool(m.any())
                if has_extra_row:
                    ext_left[ing] -= 1
                    new_extra.append(ing)
                else:
                    new_main.append(ing)
            else:
                new_main.append(ing)
        bowl[main_cat] = new_main
        bowl[extra_cat] = new_extra


def _bowl_ingredient_frozenset(selection: dict) -> frozenset:
    names: list[str] = []
    for key in HEYBO_BOWL_COMPONENT_KEYS:
        names.extend(selection.get(key) or [])
    return frozenset(str(n).strip() for n in names if n and str(n).strip())


def _primary_protein_sauce(bowl: dict) -> tuple[str, str]:
    proteins = list(bowl.get("Proteins") or []) + list(bowl.get("Extra Proteins") or [])
    protein = str(proteins[0]).strip() if proteins else ""
    sauces = list(bowl.get("Sauces") or [])
    sauce = str(sauces[0]).strip() if sauces else ""
    return protein, sauce


def _primary_bowl_key(protein: str, sauce: str) -> str:
    if protein and sauce:
        return f"{protein}_{sauce}"
    if sauce:
        return sauce
    return "Lulu-BYB"


def _finalize_bowl_dict(
    selection: dict[str, list[str]],
    user_input: dict,
    df,
    include_list: list[str] | None,
    extra_list: list[str] | None,
) -> dict[str, Any]:
    bowl: dict[str, Any] = {
        "Shop Name": user_input.get("ShopName"),
        "Bowl Type": user_input.get("Bowl Type", "bowl"),
        "Bases": list(selection.get("Bases") or []),
        "Proteins": list(selection.get("Proteins") or []),
        "Extra Proteins": [],
        "Warm sides": list(selection.get("Warm sides") or []),
        "Extra Warm sides": [],
        "Cold sides": list(selection.get("Cold sides") or []),
        "Extra Cold sides": [],
        "Dips": list(selection.get("Dips") or []),
        "Garnish": list(selection.get("Garnish") or []),
        "Sauces": list(selection.get("Sauces") or []),
        "Total Cost": [],
        "Total Nutrients": {},
        "_from_cpsat": True,
    }
    _split_user_extras_into_bowl(bowl, include_list, extra_list, df)
    heybo_cfg = user_input.get("_heybo_cfg") or {}
    price_config = heybo_cfg.get("price_config") or {}
    try:
        apply_pricing_tier_split_to_bowl(bowl, df, price_config)
    except Exception as e:
        dbg_print(f"[CP-SAT] pricing tier split skipped: {e}")
    total_nutrients = calculate_heybo_total_nutrients(bowl, df)
    bowl_weight = calculate_heybo_bowl_weight(bowl, df)
    try:
        bowl_cost, breakdown = calculate_heybo_bowl_cost_with_breakdown(
            bowl, df, user_input=user_input, price_config=price_config
        )
    except Exception:
        bowl_cost, breakdown = 0.0, {}
    bowl["Total Weight"] = round(bowl_weight)
    bowl["Total Cost"] = f"{bowl_cost:.2f}"
    bowl["PriceBreakdown"] = breakdown
    bowl["Total Nutrients"] = {
        nutrient: round(total_nutrients.get(nutrient, 0), 2) for nutrient in NUTRIENT_COLUMNS
    }
    bowl["Total_CO2e_g"] = calculate_heybo_total_co2e(bowl, df)
    return bowl


def _parse_bowl_cost(bowl: dict) -> float | None:
    raw = bowl.get("Total Cost")
    if raw is None or raw == "" or raw == []:
        return None
    try:
        return float(str(raw).replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def _incompatible_in_bowl(bowl: dict, incompatible_pairs: dict | None) -> bool:
    if not incompatible_pairs:
        return False
    all_ings: set[str] = set()
    for key in HEYBO_BOWL_COMPONENT_KEYS:
        all_ings.update(str(x).strip() for x in (bowl.get(key) or []) if x)
    seen: set[tuple[str, str]] = set()
    for ing_a, bad in (incompatible_pairs or {}).items():
        if ing_a not in all_ings:
            continue
        for ing_b in bad or set():
            if ing_b not in all_ings:
                continue
            pair = tuple(sorted((ing_a, ing_b)))
            if pair in seen:
                continue
            seen.add(pair)
            return True
    return False


def validate_cpsat_bowl_companion_filters(
    bowl: dict,
    user_input: dict,
    df,
    flavor_preferences: dict | None = None,
    cuisine_matched_names: set[str] | list[str] | None = None,
    include_list: list | None = None,
    extra_list: list | None = None,
    filtered_ingredients=None,
    skip_nutrient_keys: set | None = None,
    nutrient_filters: list | None = None,
    only_exact_bowl: bool = False,
    incompatible_pairs: dict | None = None,
    **kwargs,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if only_exact_bowl:
        return True, reasons

    if _incompatible_in_bowl(bowl, incompatible_pairs):
        reasons.append("incompatible pair")

    catalog_names: set[str] = set()
    if filtered_ingredients is not None and not getattr(filtered_ingredients, "empty", True):
        try:
            catalog_names = set(filtered_ingredients["ingredient_name"].astype(str).str.strip())
        except Exception:
            catalog_names = set()
    force_on = set(_requested_portion_counts(include_list, extra_list).keys())
    if catalog_names:
        for key in HEYBO_BOWL_COMPONENT_KEYS:
            for ing in bowl.get(key) or []:
                name = str(ing).strip()
                if name and name not in catalog_names and name not in force_on:
                    reasons.append(f"not in filtered catalog: {name}")
                    break

    cuisine_names, flavor_names, prep_names = _companion_match_name_sets(
        user_input,
        df if filtered_ingredients is None else filtered_ingredients,
        flavor_preferences=flavor_preferences,
        cuisine_matched_names=cuisine_matched_names,
    )
    true_cuisine = {
        str(n).strip()
        for n in ((user_input or {}).get("_heybo_true_cuisine_names") or set())
        if n and str(n).strip()
    }
    if true_cuisine:
        cuisine_names = true_cuisine
    bowl_names = _bowl_ingredient_frozenset(bowl)
    if _cuisine_filters_active(user_input) and cuisine_names and not (bowl_names & cuisine_names):
        reasons.append("no cuisine match")
    if _flavor_preferences_active(user_input, flavor_preferences) and flavor_names and not (bowl_names & flavor_names):
        reasons.append("no flavor match")
    if _prep_method_active(user_input) and prep_names and not (bowl_names & prep_names):
        reasons.append("no prep match")

    if not only_exact_bowl:
        lo, hi = _effective_price_bounds(user_input)
        cost = _parse_bowl_cost(bowl)
        if cost is not None:
            if lo is not None and cost < lo - 0.01:
                reasons.append(f"price below min ({cost:.2f})")
            if hi is not None and hi != float("inf") and cost > hi + 0.01:
                reasons.append(f"price above max ({cost:.2f})")

        min_co2e, max_co2e = _co2e_bounds_for_user(user_input)
        if min_co2e is not None:
            total_co2 = float(pd.to_numeric(bowl.get("Total_CO2e_g"), errors="coerce") or 0)
            if total_co2 < min_co2e or (max_co2e is not None and total_co2 > max_co2e):
                reasons.append("co2e out of band")

        nf = nutrient_filters if nutrient_filters is not None else (user_input.get("NutrientFilters") or [])
        skip = set(skip_nutrient_keys or [])
        check_nf = []
        for item in nf or []:
            key = _resolve_nutrient_filter_key(str((item or {}).get("Nutrient") or ""))
            if key and key not in skip:
                check_nf.append(item)
        if check_nf and not meets_heybo_nutrient_filters(bowl.get("Total Nutrients") or {}, check_nf):
            reasons.append("nutrients")

        light = _truthy_flag(user_input.get("Light", False))
        hearty = _truthy_flag(user_input.get("Hearty", False))
        lh_suppressed = bool(user_input.get("_light_hearty_suppressed_by_nutrients")) or (
            light_hearty_suppressed_by_nutrient_filters(user_input.get("NutrientFilters") or [])
        )
        if (light or hearty) and not (light and hearty) and not lh_suppressed:
            ok_lh, _details = evaluate_light_hearty_bowl(
                bowl.get("Total Nutrients") or {},
                float(bowl.get("Total Weight") or 0),
                light,
                hearty,
                int(user_input.get("light_hearty_relaxation_level") or 0),
                skip_nutrient_keys=list(skip) or None,
            )
            if not ok_lh:
                reasons.append("light_hearty")

        if _balanced_is_active(user_input):
            if not heybo_meets_balanced_diet(
                bowl.get("Total Nutrients") or {},
                int(user_input.get("balanced_relaxation_level") or 0),
            ):
                reasons.append("balanced")

    return (not reasons), reasons


def filter_cpsat_bowls_by_companion_filters(
    bowls: list[dict],
    **kwargs,
) -> tuple[list[dict], int]:
    accepted: list[dict] = []
    rejected = 0
    for bowl in bowls or []:
        ok, reasons = validate_cpsat_bowl_companion_filters(bowl, **kwargs)
        if ok:
            accepted.append(bowl)
        else:
            rejected += 1
            dbg_print(f"DEBUG: CP-SAT bowl rejected by companion filters: {', '.join(reasons)}")
    return accepted, rejected


def search_feasible_bowls_cpsat(
    df,
    user_input: dict,
    nutrient_filters: list[dict],
    category_pools: dict[str, list[str]],
    normal_flow_category_limits: dict,
    customization_flow_extra_category_max: dict,
    incompatible_pairs: dict[str, set[str]] | None = None,
    include_list: list[str] | None = None,
    extra_list: list[str] | None = None,
    exclude_list: list[str] | None = None,
    omit_categories: set[str] | None = None,
    max_solutions: int = 5,
    time_limit_seconds: float = 15.0,
    expand_to_customization_max: bool = False,
    weight_range: tuple[float, float] | None = None,
    previous_bowls: set | None = None,
    previous_protein_sauce_combos: set | None = None,
    filtered_ingredients=None,
    apriori_suggestions: list[dict] | None = None,
    apriori_seed_ingredients: list[str] | None = None,
    flavor_preferences: dict | None = None,
    cuisine_matched_names: set[str] | list[str] | None = None,
    only_exact_bowl: bool = False,
    customization_category_limits: dict | None = None,
    **kwargs,
) -> tuple[list[dict], str]:
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return [], "CP-SAT skipped: ortools not installed (pip install ortools)"

    exclude_set = {i.strip() for i in (exclude_list or []) if i and i.strip()}
    portion_req = _requested_portion_counts(include_list, extra_list)
    force_on_set = set(portion_req.keys())
    exclude_set.update(_excluded_cuisine_ingredient_names(df, user_input) - force_on_set)
    exclude_set.update(_excluded_prep_ingredient_names(df, user_input) - force_on_set)
    incompatible_pairs = incompatible_pairs or {}
    weight_lo = float(weight_range[0]) if weight_range else None
    weight_hi = float(weight_range[1]) if weight_range else None
    heybo_cfg = user_input.get("_heybo_cfg") or {}
    global_max_weight_g = float(heybo_cfg.get("max_bowl_weight") or 0) or 750.0

    if only_exact_bowl:
        limits, overflow_notices = _only_exact_category_limits(
            customization_category_limits,
            include_list,
            extra_list,
            df,
            omit_categories,
        )
        if overflow_notices:
            bucket = user_input.setdefault("_cpsat_category_limit_notices", [])
            for note in overflow_notices:
                if note not in bucket:
                    bucket.append(note)
        portion_req = _trim_portion_req_to_category_caps(
            portion_req, include_list, extra_list, df, limits
        )
        force_on_set = {k for k, v in portion_req.items() if int(v or 0) > 0}
        weight_lo = None
        weight_hi = None
        dbg_print(f"[CP-SAT] Only exact bowl 1 slot targets: {limits}")
    else:
        limits = _category_limits(
            normal_flow_category_limits=normal_flow_category_limits,
            customization_flow_extra_category_max=customization_flow_extra_category_max,
            omit_categories=omit_categories,
            expand_to_customization_max=expand_to_customization_max,
            include_list=include_list,
            extra_list=extra_list,
            df=df,
            variety_bowl=bool((user_input or {}).get("_only_mode_variety_active") or (user_input or {}).get("_heybo_variety_bowls_active")),
        )
        limits = _apply_heybo_tight_price_slot_caps(user_input, limits)

    pools: dict[str, list[str]] = {}
    ingredient_category: dict[str, str] = {}
    for cat in _CPSAT_CATEGORIES:
        raw = list(category_pools.get(cat) or [])
        seen: set[str] = set()
        cleaned: list[str] = []
        for ing in raw:
            ing = str(ing).strip()
            if not ing or ing in exclude_set or ing in seen:
                continue
            seen.add(ing)
            cleaned.append(ing)
            ingredient_category[ing] = cat
        pools[cat] = cleaned

    def _append_to_pool(cat: str, ing: str) -> None:
        ing = str(ing).strip()
        if not ing or ing in exclude_set:
            return
        if ing not in pools.setdefault(cat, []):
            pools[cat].append(ing)
        ingredient_category[ing] = cat

    apriori_seed = list(apriori_seed_ingredients or []) or (
        list(include_list or []) + list(extra_list or [])
    )
    apriori_candidates: dict[str, list[str]] = {}
    if not only_exact_bowl:
        apriori_candidates = _apriori_candidates_by_category(
            apriori_suggestions,
            include_list=apriori_seed,
            extra_list=[],
        )
    if apriori_candidates:
        for cat, names in apriori_candidates.items():
            if cat not in pools:
                continue
            for name in names:
                if name in exclude_set:
                    continue
                if name not in ingredient_category:
                    rows = df.loc[df["ingredient_name"] == name, "category"]
                    if rows.empty:
                        continue
                    mapped = _map_db_category_to_cpsat(str(rows.iloc[0]))
                    if mapped == cat:
                        _append_to_pool(cat, name)
                elif ingredient_category.get(name) == cat:
                    _append_to_pool(cat, name)

    for ing in force_on_set:
        if ing in exclude_set:
            continue
        if ing not in ingredient_category:
            rows = df.loc[df["ingredient_name"] == ing, "category"]
            if rows.empty:
                return [], f"CP-SAT infeasible: Include/Extra ingredient '{ing}' not in menu pool"
            mapped_cats: set[str] = set()
            for db_cat in rows["category"].tolist():
                mapped = _map_db_category_to_cpsat(str(db_cat))
                if mapped:
                    mapped_cats.add(mapped)
            if not mapped_cats:
                return [], f"CP-SAT infeasible: Include/Extra ingredient '{ing}' has no CP-SAT category"
            for mapped in mapped_cats:
                _append_to_pool(mapped, ing)

    if only_exact_bowl:
        asked = _asked_counts_by_cpsat_category(include_list, extra_list, df)
        for cat in _CPSAT_CATEGORIES:
            cmin = int((limits.get(cat) or (0, 0))[0] or 0)
            if asked.get(cat, 0) >= cmin:
                pools[cat] = [
                    ing for ing in (pools.get(cat) or []) if portion_req.get(ing, 0) > 0
                ]

    cuisine_match_names, flavor_match_names, prep_match_names = _companion_match_name_sets(
        user_input,
        df if filtered_ingredients is None else filtered_ingredients,
        flavor_preferences=flavor_preferences,
        cuisine_matched_names=cuisine_matched_names,
    )
    match_df = df if filtered_ingredients is None else filtered_ingredients
    true_cuisine_by_cat = _true_cuisine_matches_by_cpsat_category(
        match_df, user_input, pools, exclude_set
    )
    true_flavor_by_cat = _flavor_matches_by_cpsat_category(
        match_df, user_input, flavor_preferences, pools, exclude_set
    )
    true_prep_by_cat = _true_prep_matches_by_cpsat_category(
        match_df, user_input, pools, exclude_set
    )
    true_cuisine_names = {n for names in true_cuisine_by_cat.values() for n in names}
    true_flavor_names = {n for names in true_flavor_by_cat.values() for n in names}
    true_prep_names = {n for names in true_prep_by_cat.values() for n in names}
    if true_cuisine_by_cat:
        dbg_print(
            "[CP-SAT] true cuisine matches by category: "
            + ", ".join(f"{cat}={len(v)}" for cat, v in true_cuisine_by_cat.items())
        )
        if user_input is not None and not user_input.get("_heybo_true_cuisine_names"):
            user_input["_heybo_true_cuisine_names"] = set(true_cuisine_names)
            user_input["_heybo_true_cuisine_by_category"] = true_cuisine_by_cat
    if true_flavor_by_cat:
        dbg_print(
            "[CP-SAT] true flavor matches (requested intensity only): "
            + ", ".join(
                f"{cat}={len(v)} [{', '.join(v)}]"
                for cat, v in true_flavor_by_cat.items()
            )
        )
    elif _flavor_preferences_requested(user_input, flavor_preferences):
        dbg_print("[CP-SAT] true flavor matches (requested intensity only): none")

    flavor_rank_by_name = _flavor_fallback_rank_by_name(
        match_df, user_input, pools, exclude_set, flavor_preferences
    )
    if flavor_rank_by_name:
        band_counts: dict[int, int] = {}
        for rank in flavor_rank_by_name.values():
            band_counts[rank] = band_counts.get(rank, 0) + 1
        fp_dbg = flavor_preferences if flavor_preferences is not None else (user_input or {}).get("FlavorPreferences") or {}
        seq_bits = []
        if isinstance(fp_dbg, dict):
            for flav, pref in fp_dbg.items():
                pref_s = str(pref or "").strip()
                if pref_s in _FLAVOR_FALLBACK_ORDER:
                    seq_bits.append(f"{str(flav).lower()} {'→'.join(_FLAVOR_FALLBACK_ORDER[pref_s])}")
        dbg_print(
            "[CP-SAT] flavor fallback Maximize: "
            + ("; ".join(seq_bits) or "n/a")
            + " | "
            + ", ".join(
                f"rank{r}(w={_FLAVOR_FALLBACK_WEIGHTS[r]})={band_counts.get(r, 0)}"
                for r in range(len(_FLAVOR_FALLBACK_WEIGHTS))
            )
        )

    limits = _raise_protein_min_when_available(
        limits, pools.get("Proteins") or [], only_exact_bowl
    )
    protein_pool_set = set(pools.get("Proteins") or [])
    cuisine_proteins = {n for n in true_cuisine_names if n in protein_pool_set}
    flavor_proteins = {n for n in true_flavor_names if n in protein_pool_set}
    prep_proteins = {n for n in true_prep_names if n in protein_pool_set}
    preferred_proteins = cuisine_proteins or flavor_proteins or prep_proteins
    if preferred_proteins and not only_exact_bowl:
        dbg_print(
            "[CP-SAT] preferred proteins (cuisine/flavor/prep): "
            + ", ".join(sorted(preferred_proteins))
        )

    few_forced_names: set[str] = set()
    if not only_exact_bowl:
        already = set(force_on_set)
        few_forced_names |= _few_match_names_to_force(
            true_cuisine_by_cat, limits, already | few_forced_names, ingredient_category, "cuisine"
        )
        few_forced_names |= _few_match_names_to_force(
            true_flavor_by_cat, limits, already | few_forced_names, ingredient_category, "flavor"
        )
        few_forced_names |= _few_match_names_to_force(
            true_prep_by_cat, limits, already | few_forced_names, ingredient_category, "prep"
        )
        for name in few_forced_names:
            if name not in portion_req:
                portion_req[name] = 1
            cat = ingredient_category.get(name)
            if cat:
                _append_to_pool(cat, name)
        if few_forced_names:
            force_on_set.update(few_forced_names)
            dbg_print(
                "[CP-SAT] FEW-match force in every bowl: "
                + ", ".join(sorted(few_forced_names))
            )
    few_forced_sauces = {
        n for n in few_forced_names if ingredient_category.get(n) == "Sauces"
    }

    drift = _rounding_drift_units(limits)
    co2e_drift = max(2, drift // 10)
    co2e_min_scaled, co2e_max_scaled = _cpsat_co2e_scaled_bounds(user_input, co2e_drift)
    price_min_cents, price_max_cents = _cpsat_ingredient_price_sum_bounds(user_input)
    if only_exact_bowl:
        price_min_cents = None
        price_max_cents = None
        co2e_min_scaled = None
        co2e_max_scaled = None
        nutrient_filters = []
        dbg_print("[CP-SAT] Only exact: skipping nutrient / price / CO2e / Light-Hearty mins")

    build_obj_terms: list[Any] = []

    def _build_model(
        nutrient_overrides: dict[str, tuple[float, float]] | None = None,
        category_overrides: dict[str, tuple[int, int]] | None = None,
    ):
        model = cp_model.CpModel()
        vars_by_slot: dict[tuple[str, str], Any] = {}
        slots_by_ingredient: dict[str, list[Any]] = {}
        present_by_ingredient: dict[str, Any] = {}
        vars_by_category: dict[str, list[Any]] = {cat: [] for cat in _CPSAT_CATEGORIES}

        for cat in _CPSAT_CATEGORIES:
            for ing in pools.get(cat) or []:
                slot = (cat, ing)
                if slot in vars_by_slot:
                    continue
                req = int(portion_req.get(ing, 0) or 0)
                max_portions = max(1, req)
                var = model.NewIntVar(0, max_portions, f"{cat}:{ing}:portions")
                vars_by_slot[slot] = var
                vars_by_category[cat].append(var)
                slots_by_ingredient.setdefault(ing, []).append(var)
                if ing in exclude_set:
                    model.Add(var == 0)

        for ing, req in portion_req.items():
            ing_vars = slots_by_ingredient.get(ing) or []
            if ing_vars and req > 0:
                model.Add(sum(ing_vars) == int(req))

        for ing, slot_vars in slots_by_ingredient.items():
            present = model.NewBoolVar(f"{ing}:present")
            present_by_ingredient[ing] = present
            total = sum(slot_vars)
            model.Add(total >= 1).OnlyEnforceIf(present)
            model.Add(total == 0).OnlyEnforceIf(present.Not())

        for cat, (cmin, cmax) in limits.items():
            if category_overrides and cat in category_overrides:
                cmin, cmax = category_overrides[cat]
            if cat == "Proteins":
                limit_min = int(limits.get(cat, (0, 0))[0])
                cmin = max(int(cmin), limit_min)
                cmax = max(int(cmax), int(cmin))
            cat_vars = vars_by_category.get(cat) or []
            if not cat_vars:
                if cmin > 0:
                    return None, None, None, None, (
                        f"CP-SAT infeasible: category '{cat}' requires min={cmin} but pool is empty"
                    )
                continue
            forced = sum(int(portion_req.get(ing, 0) or 0) for ing in (pools.get(cat) or []))
            max_possible = sum(
                max(1, int(portion_req.get(ing, 0) or 0)) for ing in (pools.get(cat) or [])
            )
            cmax = min(int(cmax), max_possible)
            cmin = min(int(cmin), cmax)
            if forced > cmax and not only_exact_bowl:
                cmax = forced
            cmin = max(int(cmin), forced)
            cmin = min(cmin, cmax)
            model.Add(sum(cat_vars) >= int(cmin))
            model.Add(sum(cat_vars) <= int(cmax))

        if not only_exact_bowl:
            warm_vars = vars_by_category.get("Warm sides") or []
            cold_vars = vars_by_category.get("Cold sides") or []
            side_cap = (
                int((limits.get("Warm sides") or (0, 0))[1] or 0)
                + int((limits.get("Cold sides") or (0, 0))[1] or 0)
            )
            required_sides = min(_MIN_TOTAL_SIDES_NORMAL, side_cap)
            if required_sides > 0 and (warm_vars or cold_vars):
                model.Add(sum(warm_vars) + sum(cold_vars) >= int(required_sides))

        for nf in nutrient_filters or []:
            key = _resolve_nutrient_filter_key(str(nf.get("Nutrient") or "").strip())
            if not key:
                continue
            rng = nf.get("Range") or {}
            min_raw = rng.get("Min")
            max_raw = rng.get("Max")
            if nutrient_overrides and key in nutrient_overrides:
                min_raw, max_raw = nutrient_overrides[key]
            scaled_min, scaled_max = _scaled_nutrient_bounds(min_raw, max_raw, drift)
            coeff: list[tuple[Any, int]] = []
            for (_cat, ing), var in vars_by_slot.items():
                scaled = _ingredient_nutrient_scaled(df, ing, key)
                if scaled:
                    coeff.append((var, scaled))
            total_expr = sum(c * v for v, c in coeff) if coeff else 0
            if scaled_min is not None and float(min_raw or 0) > 0:
                if not coeff:
                    return None, None, None, None, (
                        f"CP-SAT infeasible: no menu ingredient contributes to {key} "
                        f"but Min={min_raw}"
                    )
                model.Add(total_expr >= scaled_min)
            if scaled_max is not None:
                if not coeff:
                    continue
                model.Add(total_expr <= scaled_max)

        hard_weight_hi = global_max_weight_g
        if weight_hi is not None:
            hard_weight_hi = min(hard_weight_hi, float(weight_hi))
        solver_weight_hi = hard_weight_hi + _WEIGHT_TRIM_SLACK_G
        if weight_lo is not None or solver_weight_hi > 0:
            weight_terms: list[tuple[Any, int]] = []
            for (_cat, ing), var in vars_by_slot.items():
                grams = _ingredient_weight_g(df, ing)
                if grams:
                    weight_terms.append((var, grams))
            if not weight_terms and weight_lo is not None and weight_lo > 0:
                return None, None, None, None, (
                    "CP-SAT infeasible: Light/Hearty weight Min set but no ingredient weights in menu"
                )
            weight_expr = sum(g * v for v, g in weight_terms) if weight_terms else 0
            if weight_lo is not None:
                model.Add(weight_expr >= max(0, int(round(weight_lo)) - _WEIGHT_DRIFT_G))
            model.Add(weight_expr <= int(round(solver_weight_hi)) + _WEIGHT_DRIFT_G)

        if co2e_min_scaled is not None or co2e_max_scaled is not None:
            co2e_terms: list[tuple[Any, int]] = []
            for (_cat, ing), var in vars_by_slot.items():
                scaled = _ingredient_co2e_scaled(df, ing)
                if scaled:
                    co2e_terms.append((var, scaled))
            if co2e_min_scaled is not None and co2e_min_scaled > 0 and not co2e_terms:
                return None, None, None, None, (
                    "CP-SAT infeasible: Sustainable CO2e Min set but no CO2e data in menu pool"
                )
            if co2e_terms:
                co2e_expr = sum(c * v for v, c in co2e_terms)
                if co2e_min_scaled is not None:
                    model.Add(co2e_expr >= co2e_min_scaled)
                if co2e_max_scaled is not None:
                    model.Add(co2e_expr <= co2e_max_scaled)

        if price_min_cents is not None or price_max_cents is not None:
            price_terms: list[tuple[Any, int]] = []
            for (cat, ing), var in vars_by_slot.items():
                cents = _ingredient_price_cents(df, ing, cat)
                if cents:
                    price_terms.append((var, cents))
            if price_min_cents is not None and price_min_cents > 0 and not price_terms:
                return None, None, None, None, (
                    "CP-SAT infeasible: Price Min set but no ai_price data in menu pool"
                )
            if price_terms:
                price_expr = sum(c * v for v, c in price_terms)
                if price_min_cents is not None:
                    model.Add(price_expr >= price_min_cents)
                if price_max_cents is not None:
                    model.Add(price_expr <= price_max_cents)

        seen_pairs: set[tuple[str, str]] = set()
        for ing_a, bad in incompatible_pairs.items():
            if ing_a not in present_by_ingredient:
                continue
            for ing_b in bad or set():
                if ing_b not in present_by_ingredient:
                    continue
                pair = tuple(sorted((ing_a, ing_b)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                model.Add(present_by_ingredient[ing_a] + present_by_ingredient[ing_b] <= 1)

        obj_terms: list[Any] = []
        apriori_preferred = _sample_apriori_preferred(apriori_candidates)
        for cat, preferred in (apriori_preferred or {}).items():
            for rank, name in enumerate(preferred):
                if name in present_by_ingredient:
                    weight = max(1, 3 - rank)
                    obj_terms.append(weight * present_by_ingredient[name])
        true_preferred = true_cuisine_names | true_flavor_names | true_prep_names
        name_weights: dict[str, int] = {}
        for name in true_preferred | set(preferred_proteins or []):
            name_weights[name] = max(name_weights.get(name, 0), 6)
        for name, rank in (flavor_rank_by_name or {}).items():
            if rank < 0 or rank >= len(_FLAVOR_FALLBACK_WEIGHTS):
                continue
            name_weights[name] = max(name_weights.get(name, 0), _FLAVOR_FALLBACK_WEIGHTS[rank])
        for name, weight in name_weights.items():
            if name in present_by_ingredient:
                obj_terms.append(weight * present_by_ingredient[name])

        def _require_at_least_one(match_names: set[str], label: str) -> str | None:
            if not match_names:
                return None
            vars_in_model = [
                present_by_ingredient[n] for n in match_names if n in present_by_ingredient
            ]
            if not vars_in_model:
                return f"CP-SAT infeasible: {label} matches are not in the current menu pool"
            model.Add(sum(vars_in_model) >= 1)
            return None

        if not only_exact_bowl and preferred_proteins:
            err = _require_at_least_one(preferred_proteins, "preferred protein")
            if err:
                return None, None, None, None, err
        if not only_exact_bowl and _cuisine_filters_requested(user_input):
            err = _require_at_least_one(true_cuisine_names or cuisine_match_names, "cuisine")
            if err:
                return None, None, None, None, err
        if not only_exact_bowl and _flavor_preferences_requested(user_input, flavor_preferences):
            err = _require_at_least_one(true_flavor_names or flavor_match_names, "flavor")
            if err:
                return None, None, None, None, err
        if not only_exact_bowl and _prep_method_requested(user_input):
            err = _require_at_least_one(true_prep_names or prep_match_names, "prep method")
            if err:
                return None, None, None, None, err
        pref_sauces: set[str] = set()
        if _cuisine_filters_requested(user_input):
            pref_sauces.update(true_cuisine_by_cat.get("Sauces") or [])
        if _flavor_preferences_requested(user_input, flavor_preferences):
            pref_sauces.update(true_flavor_by_cat.get("Sauces") or [])
        if _prep_method_requested(user_input):
            pref_sauces.update(true_prep_by_cat.get("Sauces") or [])
        sauce_min = int((limits.get("Sauces") or (0, 0))[0] or 0)
        if (
            not only_exact_bowl
            and pref_sauces
            and sauce_min >= 1
            and not few_forced_sauces
        ):
            err = _require_at_least_one(pref_sauces, "cuisine/flavor/prep sauce")
            if err:
                return None, None, None, None, err

        build_obj_terms.clear()
        if obj_terms and not only_exact_bowl:
            build_obj_terms.extend(obj_terms)

        return model, vars_by_slot, vars_by_category, present_by_ingredient, None

    def _forbid_protein_sauce_pair(
        model,
        present_by_ingredient: dict[str, Any],
        protein: str,
        sauce: str,
    ) -> None:
        if protein and sauce:
            if protein in present_by_ingredient and sauce in present_by_ingredient:
                model.Add(present_by_ingredient[protein] + present_by_ingredient[sauce] <= 1)

    def _apply_diversity(
        model,
        vars_by_slot: dict[tuple[str, str], Any],
        present_by_ingredient: dict[str, Any],
        unique_ps: bool,
        used_ps: list[tuple[str, str]],
        chosen_sets: list[list[str]],
    ) -> list[Any]:
        diversity_obj: list[Any] = []
        if unique_ps:
            used_proteins = {p for p, _s in used_ps if p}
            used_sauces = {s for _p, s in used_ps if s}
            rotate_proteins = preferred_proteins if preferred_proteins else set(
                pools.get("Proteins") or []
            )
            unused_proteins = [
                n
                for n in rotate_proteins
                if n in present_by_ingredient
                and n not in used_proteins
                and n not in few_forced_names
                and n not in force_on_set
            ]
            rotated_proteins = False
            if unused_proteins and used_proteins:
                forbade_protein = False
                for name in used_proteins:
                    if name in few_forced_names or name in force_on_set:
                        continue
                    if name in rotate_proteins and name in present_by_ingredient:
                        model.Add(present_by_ingredient[name] == 0)
                        forbade_protein = True
                if forbade_protein:
                    dbg_print(
                        f"[CP-SAT] diversity: {len(unused_proteins)} unused protein(s) remain; "
                        "forbidding already-used proteins"
                    )
                    rotated_proteins = True

            rotate_sauces: set[str] = set()
            for by_cat in (true_cuisine_by_cat, true_flavor_by_cat, true_prep_by_cat):
                rotate_sauces.update(by_cat.get("Sauces") or [])
            # Full-pool unused-sauce rotation with unique proteins makes the
            # next bowl need a new protein AND a new sauce. Under Price /
            # NutrientFilters that burns solves; only rotate true matches.
            if not rotate_sauces and not nutrient_filters and not _heybo_price_filter_active(
                user_input
            ):
                rotate_sauces = {
                    n for n in (pools.get("Sauces") or []) if n in present_by_ingredient
                }
            unused_sauces = [
                n
                for n in rotate_sauces
                if n in present_by_ingredient
                and n not in used_sauces
                and n not in few_forced_names
                and n not in force_on_set
            ]
            rotated_sauces = False
            if unused_sauces and used_sauces and not few_forced_sauces:
                dbg_print(
                    f"[CP-SAT] diversity: {len(unused_sauces)} unused sauce(s) remain; "
                    "forbidding already-used sauces"
                )
                for name in used_sauces:
                    if name in few_forced_names or name in force_on_set:
                        continue
                    if name in rotate_sauces and name in present_by_ingredient:
                        model.Add(present_by_ingredient[name] == 0)
                rotated_sauces = True

            if few_forced_sauces and not rotated_sauces:
                dbg_print(
                    "[CP-SAT] diversity: FEW cuisine/flavor/prep sauces are forced; "
                    "keeping that sauce on every bowl"
                )
            elif not rotated_proteins and not rotated_sauces:
                for protein, sauce in used_ps:
                    _forbid_protein_sauce_pair(model, present_by_ingredient, protein, sauce)

        # Variety: without NutrientFilters, rotate fillers by forbidding used
        # names. With NutrientFilters, never force a swap — Maximize unused
        # sides so bowls diverge when the band allows it, and keep the same
        # packing when only a few names still fit (same as salad).
        filler_cats: tuple[str, ...] = (
            "Bases",
            "Warm sides",
            "Cold sides",
            "Dips",
            "Garnish",
        )
        if not unique_ps:
            filler_cats = filler_cats + ("Sauces",)
        used_by_cat: dict[str, set[str]] = {cat: set() for cat in filler_cats}
        for chosen in chosen_sets:
            for name in chosen or []:
                cat = ingredient_category.get(name)
                if cat in used_by_cat:
                    used_by_cat[cat].add(name)
        for cat in filler_cats:
            pool = [n for n in (pools.get(cat) or []) if n in present_by_ingredient]
            if not pool:
                continue
            forced_here = {
                n
                for n in (few_forced_names | force_on_set)
                if ingredient_category.get(n) == cat
            }
            used_fillers = used_by_cat[cat] - forced_here
            unused_fillers = [
                n for n in pool if n not in used_by_cat[cat] and n not in forced_here
            ]
            try:
                cmin = int((limits.get(cat) or (0, 0))[0] or 0)
            except (TypeError, ValueError):
                cmin = 0
            need = max(0, cmin - len(forced_here))
            leftover = [n for n in pool if n not in used_fillers]
            if not unused_fillers or not used_fillers:
                continue
            if nutrient_filters:
                dbg_print(
                    f"[CP-SAT] diversity: {len(unused_fillers)} unused {cat} filler(s) remain; "
                    "Maximize unused when feasible (no hard swap)"
                )
                for n in unused_fillers:
                    if n in present_by_ingredient:
                        diversity_obj.append(
                            (2 + random.randint(0, 2)) * present_by_ingredient[n]
                        )
                for n in used_fillers:
                    if n in present_by_ingredient:
                        diversity_obj.append(-1 * present_by_ingredient[n])
                continue
            if len(leftover) < need:
                continue
            dbg_print(
                f"[CP-SAT] diversity: {len(unused_fillers)} unused {cat} filler(s) remain; "
                "forbidding already-used fillers"
            )
            for name in used_fillers:
                if name in present_by_ingredient:
                    model.Add(present_by_ingredient[name] == 0)

        for chosen in chosen_sets:
            present = [ing for ing in chosen if ing in present_by_ingredient]
            if present:
                model.Add(
                    sum(present_by_ingredient[ing] for ing in present) <= len(present) - 1
                )
        return diversity_obj

    def _accept_solution(solver, vars_by_slot):
        selection: dict[str, list[str]] = {cat: [] for cat in _CPSAT_CATEGORIES}
        chosen: list[str] = []
        for (cat, ing), var in vars_by_slot.items():
            count = int(solver.Value(var))
            if count <= 0:
                continue
            for _ in range(count):
                selection.setdefault(cat, []).append(ing)
            chosen.append(ing)
        bowl_set = _bowl_ingredient_frozenset(selection)
        if bowl_set in blocked_bowl_sets:
            dbg_print("[CP-SAT] rejected bowl: matches previous/session page bowl")
            return None, None, chosen
        bowl = _finalize_bowl_dict(
            selection,
            user_input=user_input,
            df=df,
            include_list=include_list,
            extra_list=extra_list,
        )
        final_set = _bowl_ingredient_frozenset(bowl)
        if final_set in blocked_bowl_sets:
            dbg_print("[CP-SAT] rejected bowl after finalize: matches previous/session page bowl")
            return None, None, chosen

        hard_weight_hi = global_max_weight_g
        if weight_hi is not None:
            hard_weight_hi = min(hard_weight_hi, float(weight_hi))
        if float(bowl.get("Total Weight") or 0) > hard_weight_hi + 0.5:
            dbg_print("[CP-SAT] rejected bowl: over max bowl weight after finalize")
            blocked_bowl_sets.add(final_set)
            return None, None, chosen

        ok, reasons = validate_cpsat_bowl_companion_filters(
            bowl,
            user_input=user_input,
            df=df,
            flavor_preferences=flavor_preferences,
            cuisine_matched_names=cuisine_match_names,
            include_list=include_list,
            extra_list=extra_list,
            filtered_ingredients=filtered_ingredients,
            nutrient_filters=nutrient_filters,
            only_exact_bowl=only_exact_bowl,
            incompatible_pairs=incompatible_pairs,
        )
        if not ok:
            dbg_print(f"[CP-SAT] rejected bowl: {', '.join(reasons)}")
            blocked_bowl_sets.add(final_set)
            return None, None, chosen
        return bowl, selection, chosen

    solver = cp_model.CpSolver()
    per_solve_limit = min(float(time_limit_seconds), 3.0)
    solver.parameters.max_time_in_seconds = per_solve_limit
    solver.parameters.num_search_workers = CPSAT_SEARCH_WORKERS

    bowls: list[dict] = []
    chosen_sets: list[list[str]] = []
    used_ps: list[tuple[str, str]] = []
    protein_pool = set(pools.get("Proteins") or [])
    sauce_pool = set(pools.get("Sauces") or [])
    for combo in previous_protein_sauce_combos or set():
        if not combo:
            continue
        if isinstance(combo, (list, tuple)) and len(combo) >= 2:
            used_ps.append((str(combo[0] or "").strip(), str(combo[1] or "").strip()))
            continue
        key = str(combo).strip()
        for protein in protein_pool:
            prefix = f"{protein}_"
            if key.startswith(prefix):
                sauce = key[len(prefix):]
                if sauce in sauce_pool:
                    used_ps.append((protein, sauce))
                    break
            elif key == protein:
                used_ps.append((protein, ""))
                break
        else:
            if key in sauce_pool:
                used_ps.append(("", key))
    blocked_bowl_sets: set = set(previous_bowls or set())
    unique_ps_count = 0
    fallback_count = 0
    spread_hits = 0
    spread_fallbacks = 0

    def _solve_one(*, unique_ps: bool, allow_spread: bool):
        nonlocal spread_hits, spread_fallbacks
        attempts: list[tuple[
            dict[str, tuple[float, float]] | None,
            dict[str, tuple[int, int]] | None,
        ]] = []
        if allow_spread:
            strict_nut = _equal_point_strict_nutrient_windows(nutrient_filters) or None
            tight_nut = _min_anchored_nutrient_windows(nutrient_filters) or None
            if strict_nut:
                attempts.append((strict_nut, None))
            if tight_nut and tight_nut != strict_nut:
                attempts.append((tight_nut, None))
            nut_for_spread = tight_nut or strict_nut
            cat = _random_category_count_targets(limits) or None
            if cat is not None:
                attempts.append((nut_for_spread, cat))
        attempts.append((None, None))

        for nut_overrides, cat_overrides in attempts:
            for _retry in range(6):
                model, vars_by_slot, vars_by_category, present_by_ingredient, build_err = (
                    _build_model(nut_overrides, cat_overrides)
                )
                if build_err or model is None:
                    if nut_overrides is None and cat_overrides is None and _retry == 0:
                        return None, build_err
                    break
                extra_obj = _apply_diversity(
                    model,
                    vars_by_slot,
                    present_by_ingredient or {},
                    unique_ps=unique_ps,
                    used_ps=used_ps,
                    chosen_sets=chosen_sets,
                )
                combined_obj = list(build_obj_terms) + list(extra_obj or [])
                if combined_obj and not only_exact_bowl:
                    model.Maximize(sum(combined_obj))
                solver.parameters.random_seed = random.randint(1, 2_000_000_000)
                status = solver.Solve(model)
                used_spread = nut_overrides is not None or cat_overrides is not None
                if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    if used_spread and _retry == 0:
                        spread_fallbacks += 1
                    break
                bowl, selection, chosen = _accept_solution(solver, vars_by_slot)
                if bowl is None:
                    if chosen:
                        continue
                    if used_spread:
                        spread_fallbacks += 1
                        break
                    return None, "postcheck_failed"
                if used_spread:
                    spread_hits += 1
                return (bowl, selection, chosen), None
        return None, None

    if only_exact_bowl:
        result, err = _solve_one(unique_ps=False, allow_spread=False)
        if result is None:
            if err and err not in (None, "postcheck_failed"):
                return [], err
            return [], "CP-SAT: Only exact bowl 1 infeasible; random assembly will try"
        bowl, selection, chosen = result
        bowls.append(bowl)
        return bowls, f"CP-SAT Only exact bowl 1 found within {per_solve_limit:g}s per solve"

    for sol_idx in range(max(1, int(max_solutions))):
        result, err = _solve_one(unique_ps=True, allow_spread=True)
        if result is None:
            if sol_idx == 0:
                if err and err not in (None, "postcheck_failed"):
                    return [], err
                if err == "postcheck_failed":
                    return [], (
                        "CP-SAT: solver found an assignment but it failed post-validation"
                    )
                return [], (
                    "CP-SAT: no bowl satisfies current nutrient ranges and category limits "
                    f"(customization_max={expand_to_customization_max})"
                )
            break
        bowl, selection, chosen = result
        bowls.append(bowl)
        chosen_sets.append(chosen)
        blocked_bowl_sets.add(_bowl_ingredient_frozenset(bowl))
        used_ps.append(_primary_protein_sauce(bowl))
        unique_ps_count += 1
        if len(bowls) >= max_solutions:
            break

    if len(bowls) < max_solutions and bowls:
        while len(bowls) < max_solutions:
            result, _err = _solve_one(unique_ps=False, allow_spread=True)
            if result is None:
                break
            bowl, selection, chosen = result
            bowls.append(bowl)
            chosen_sets.append(chosen)
            blocked_bowl_sets.add(_bowl_ingredient_frozenset(bowl))
            fallback_count += 1

    header = (
        f"CP-SAT found {len(bowls)} bowl(s) "
        f"(unique protein-sauce={unique_ps_count}, differ-by-1={fallback_count}, "
        f"spread hits={spread_hits}, spread fallbacks={spread_fallbacks})"
    )
    return bowls, header
