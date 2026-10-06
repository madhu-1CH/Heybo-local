"""
Heybo filters: preprocess user filters and apply all ingredient filters
(allergen, diet, cuisine, preparation method, flavor).
Input is assumed valid and already canonical from preference_filters.
"""
from typing import Tuple, Optional, List, Dict, Any, Set

import pandas as pd

from .diet import apply_heybo_diet_filters_with_fallback
from .light_hearty import apply_light_hearty_filter
from .diet_constants import HEYBO_EXTRA_CATEGORIES, NUTRIENT_COLUMNS, NUTRIENT_NAME_MAP


def _heybo_mapped_category(cat) -> str:
    """Extra Proteins / Extra Warm/Cold → their base CP-SAT / fill category."""
    s = str(cat or "")
    return HEYBO_EXTRA_CATEGORIES.get(s, s)


def _heybo_record_true_matches(store: dict, db_cat, names) -> None:
    mapped = _heybo_mapped_category(db_cat)
    bucket = store.setdefault(mapped, [])
    for raw in names or []:
        name = str(raw).strip()
        if name and name not in bucket:
            bucket.append(name)


def _get_filter_list(val) -> List[str]:
    """
    Accept list values as-is, or dict values in the same style as PreparationMethod:
    {"Filter A": True, "Filter B": False}. Only True keys are active.
    """
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, dict):
        return [str(k).strip() for k, v in val.items() if v is True and str(k).strip()]
    return []


def _parse_allergen_tokens(allergens_raw: Any) -> Set[str]:
    """Comma-separated allergen tags from heybo.ingredients_details.allergens."""
    if allergens_raw is None or (isinstance(allergens_raw, float) and pd.isna(allergens_raw)):
        return set()
    return {a.strip().lower() for a in str(allergens_raw).split(",") if a.strip()}


def ingredient_lists_excluded_allergen(allergens_raw: Any, filter_names: List[str]) -> bool:
    """True when any user allergen name matches a comma-separated tag on the ingredient row."""
    if not filter_names:
        return False
    tokens = _parse_allergen_tokens(allergens_raw)
    if not tokens:
        return False
    return any(str(f).strip().lower() in tokens for f in filter_names if str(f).strip())


def ingredient_names_excluded_by_allergen(
    df: pd.DataFrame,
    filter_names: List[str],
) -> Set[str]:
    """
    Ingredient names to drop when ANY catalog row (any SKU/category) lists an excluded tag.

    Heybo stores one row per sku_code; the same ingredient_name can appear on Proteins and
    Extra Proteins rows with different allergen strings. Row-only filtering can leave a
  "clean" duplicate row while a meat-tagged SKU still exists for that name.
    """
    if df.empty or not filter_names or "allergens" not in df.columns or "ingredient_name" not in df.columns:
        return set()
    excluded: Set[str] = set()
    for name, group in df.groupby("ingredient_name", sort=False):
        if any(
            ingredient_lists_excluded_allergen(raw, filter_names)
            for raw in group["allergens"]
        ):
            excluded.add(name)
    return excluded


def _safe_float_nutrient(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _equal_min_max_target(rng: Dict[str, Any]) -> Optional[float]:
    """When Min and Max are the same, return that single target value."""
    mn = _safe_float_nutrient((rng or {}).get("Min"))
    mx = _safe_float_nutrient((rng or {}).get("Max"))
    if mn is not None and mx is not None and abs(mn - mx) < 1e-6:
        return mn
    return None


def _heybo_apply_equal_min_max_target_to_range(
    rng: Dict[str, Any],
    nutrient_key: str,
    constraint: Dict[str, Any],
    unit_s: str = "",
) -> List[str]:
    """
    Equal Min/Max nutrient requests (e.g. protein 50/50). Salad parity:

    Align keeps Min=Max=target (hard customization clamps only).
    ``open_heybo_equal_nutrient_point_targets`` then opens a soft Max (+slack).
    Progressive relaxation may widen Max further; Min is never reduced.
    """
    adjustments: List[str] = []
    equal_target = _equal_min_max_target(rng)
    if equal_target is None:
        return adjustments

    def _fmt(v: Optional[float]) -> str:
        if v is None:
            return ""
        s = f"{v:g}" if isinstance(v, float) and not v.is_integer() else f"{int(v)}"
        return f"{s}{unit_s}" if unit_s else s

    target = float(equal_target)
    cust_max = _safe_float_nutrient(constraint.get("customization_max"))
    cust_min = _safe_float_nutrient(constraint.get("customization_min"))
    rng["Min"] = target
    rng["Max"] = target
    if cust_max is not None and target > cust_max:
        rng["Min"] = cust_max
        rng["Max"] = cust_max
        adjustments.append(
            f"minimum and maximum {nutrient_key} capped to {_fmt(cust_max)} "
            f"(customization maximum — absolute ceiling; user target {_fmt(target)})"
        )
        return adjustments
    if cust_min is not None and target < cust_min:
        rng["Min"] = cust_min
        rng["Max"] = cust_min
        adjustments.append(
            f"minimum and maximum {nutrient_key} raised to {_fmt(cust_min)} "
            f"(customization minimum — absolute floor; user target {_fmt(target)})"
        )
        return adjustments
    return adjustments


def _heybo_nutrient_band_max(constraint: Dict[str, Any]) -> Optional[float]:
    """
    Upper bound for high-nutrient / customization-overflow requests.
    Uses customization_max when set and above high_max; otherwise high_max.
    """
    high_max = _safe_float_nutrient(constraint.get("high_max"))
    customization_max = _safe_float_nutrient(constraint.get("customization_max"))
    if customization_max is not None:
        if high_max is None or customization_max > high_max:
            return customization_max
    return high_max


def _heybo_nutrient_band_min(constraint: Dict[str, Any]) -> Optional[float]:
    """
    Absolute floor for low-nutrient / customization-underflow requests.
    Returns customization_min when set — this is the hard lower bound below which
    no bowl's nutrient total should go, regardless of what the user typed.
    Returns None when not configured (no absolute floor enforced).
    """
    return _safe_float_nutrient(constraint.get("customization_min"))


def _heybo_apply_customization_overflow_to_range(
    rng: Dict[str, Any],
    *,
    nutrient_key: str,
    high_min: Optional[float],
    high_max: Optional[float],
    band_max: Optional[float],
    original_min: Optional[float],
    original_max: Optional[float],
    unit_s: str,
) -> List[str]:
    """
    When the user target exceeds normal high_max, widen to customization_max (if configured).
    Returns adjustment messages.
    """
    adjustments: List[str] = []

    def _fmt(v: Optional[float]) -> str:
        if v is None:
            return ""
        s = f"{v:g}" if isinstance(v, float) and not v.is_integer() else f"{int(v)}"
        return f"{s}{unit_s}" if unit_s else s

    min_val = _safe_float_nutrient(rng.get("Min"))
    max_val = _safe_float_nutrient(rng.get("Max"))
    if high_max is None:
        return adjustments

    if min_val is not None and min_val > high_max:
        if band_max is not None and band_max > high_max:
            if min_val > band_max:
                rng["Min"] = band_max
                rng["Max"] = min_val
                adjustments.append(
                    f"minimum {nutrient_key} set to {_fmt(band_max)} (customization minimum) and "
                    f"maximum set to {_fmt(min_val)} (user target above customization maximum "
                    f"{_fmt(band_max)})"
                )
            else:
                # Respect user's explicit Max if it falls within the customization band;
                # only widen to band_max when no Max was given or it exceeds the band.
                if max_val is not None and max_val <= band_max:
                    rng["Max"] = max_val
                else:
                    rng["Max"] = band_max
                adjustments.append(
                    f"maximum {nutrient_key} set to {_fmt(rng['Max'])} (customization maximum; "
                    f"user minimum {_fmt(min_val)} exceeds normal high-nutrient maximum "
                    f"{_fmt(high_max)})"
                )
        elif high_min is not None:
            rng["Min"] = high_min
            rng["Max"] = high_max
            adjustments.append(
                f"minimum {nutrient_key} adjusted from {_fmt(original_min)} to {_fmt(high_min)} "
                f"and maximum set to {_fmt(high_max)} (user target {_fmt(min_val)} exceeds "
                f"high-nutrient maximum; no customization maximum configured)"
            )
    elif max_val is not None and max_val > high_max:
        overflow_cap = band_max if band_max is not None and band_max > high_max else high_max
        if max_val > overflow_cap:
            rng["Max"] = overflow_cap
            cap_label = (
                "customization maximum"
                if overflow_cap == band_max and band_max != high_max
                else "high-nutrient maximum"
            )
            adjustments.append(
                f"maximum {nutrient_key} adjusted from {_fmt(original_max)} to "
                f"{_fmt(overflow_cap)} ({cap_label})"
            )
        else:
            rng["Max"] = max_val
        if min_val is None and high_min is not None:
            rng["Min"] = high_min
            adjustments.append(
                f"minimum {nutrient_key} auto-added as {_fmt(high_min)} "
                f"(high-nutrient minimum — user max above normal high-nutrient maximum)"
            )

    return adjustments


def _canonical_nutrient_key(nutrient_name: str) -> str:
    """Map user or internal nutrient labels to dataframe column keys."""
    n = (nutrient_name or "").strip()
    if not n:
        return ""
    if n in NUTRIENT_COLUMNS:
        return n
    if n in NUTRIENT_NAME_MAP.values():
        return n
    mapped = NUTRIENT_NAME_MAP.get(n)
    if mapped:
        return mapped
    for label, col in NUTRIENT_NAME_MAP.items():
        if label.lower() == n.lower():
            return col
    return n


def _salad_like_align_nutrient_filter(
    merged: Dict[str, Any],
    nutrition_constraints: Dict[str, Dict[str, Any]],
) -> List[str]:
    """
    Align one NutrientFilters row to heybo.nutrition_constraint_information (salad.py
    `align_nutrient_filters_with_guidelines` parity for the default / non–diet-flag paths).
    Mutates merged['Range'] and may set _guidelines_adjusted, _adjustment_messages, _original_range.
    """
    adjustments_made: List[str] = []
    nutrient_key = (merged.get("Nutrient") or "").strip()
    if not nutrient_key or nutrient_key not in nutrition_constraints:
        return adjustments_made

    g = nutrition_constraints[nutrient_key]
    unit = (g.get("unit") or "").strip()
    unit_s = f"{unit}" if unit else ""
    high_min = _safe_float_nutrient(g.get("high_min"))
    high_max = _safe_float_nutrient(g.get("high_max"))
    band_max = _heybo_nutrient_band_max(g)
    band_min = _heybo_nutrient_band_min(g)

    def _fmt(v: Optional[float]) -> str:
        if v is None:
            return ""
        s = f"{v:g}" if isinstance(v, float) and not v.is_integer() else f"{int(v)}"
        return f"{s}{unit_s}" if unit_s else s

    # Explicit user numeric requests: preserve user Min/Max, then apply customization overflow
    # when the target exceeds normal high_max (uses nutrition_constraint_information.customization_max),
    # and apply customization_min floor when the target goes below the absolute minimum.
    if merged.get("_user_original_request"):
        orig = dict(merged.get("_original_range") or {})
        rng: Dict[str, Any] = {}
        min_val = _safe_float_nutrient(orig.get("Min"))
        max_val = _safe_float_nutrient(orig.get("Max"))
        original_min, original_max = min_val, max_val
        if min_val is not None:
            rng["Min"] = min_val
        if max_val is not None:
            rng["Max"] = max_val
        if min_val is not None and max_val is not None and min_val > max_val:
            rng["Min"], rng["Max"] = max_val, min_val
            adjustments_made.append(
                f"Swapped Min and Max values for {nutrient_key} (Min was greater than Max)"
            )
            original_min, original_max = min_val, max_val
            min_val, max_val = _safe_float_nutrient(rng.get("Min")), _safe_float_nutrient(rng.get("Max"))
        equal_target = _equal_min_max_target(rng)
        if equal_target is None:
            equal_target = _equal_min_max_target(orig)
        if equal_target is not None:
            # Exact target first (e.g. protein 50/50); only hard customization clamps here.
            merged["_equal_min_max_target"] = True
            ceiling = _heybo_nutrient_band_max(g)
            if ceiling is not None:
                merged["_equal_max_ceiling"] = ceiling
            equal_adj = _heybo_apply_equal_min_max_target_to_range(
                rng, nutrient_key, g, unit_s
            )
            effective_target = _equal_min_max_target(rng)
            if effective_target is not None:
                merged["_equal_target_value"] = float(effective_target)
            elif rng.get("Min") is not None:
                merged["_equal_target_value"] = float(rng["Min"])
            merged["Range"] = rng
            all_adj = list(adjustments_made) + list(equal_adj)
            if all_adj:
                merged["_guidelines_adjusted"] = True
                merged["_adjustment_messages"] = list(all_adj)
                if not isinstance(merged.get("_original_range"), dict):
                    merged["_original_range"] = {"Min": original_min, "Max": original_max}
            return all_adj
        # Apply customization_min floor: raise the user's Min if it falls below the
        # absolute minimum configured in nutrition_constraint_information.customization_min.
        if band_min is not None:
            cur_min = _safe_float_nutrient(rng.get("Min"))
            if cur_min is not None and cur_min < band_min:
                rng["Min"] = band_min
                adjustments_made.append(
                    f"minimum {nutrient_key} raised from {_fmt(cur_min)} to {_fmt(band_min)} "
                    f"(customization minimum floor — user value is below the absolute minimum allowed)"
                )
                min_val = band_min
        adjustments_made.extend(
            _heybo_apply_customization_overflow_to_range(
                rng,
                nutrient_key=nutrient_key,
                high_min=high_min,
                high_max=high_max,
                band_max=band_max,
                original_min=original_min,
                original_max=original_max,
                unit_s=unit_s,
            )
        )
        merged["Range"] = rng
        if adjustments_made:
            merged["_guidelines_adjusted"] = True
            merged["_adjustment_messages"] = list(adjustments_made)
            if not isinstance(merged.get("_original_range"), dict):
                merged["_original_range"] = {"Min": original_min, "Max": original_max}
        return adjustments_made

    # Pure diet-derived row (no explicit user nutrient row merged): already exported as DB band.
    if merged.get("_diet_converted") and not merged.get("_user_original_request"):
        return adjustments_made

    low_min = _safe_float_nutrient(g.get("low_min"))
    low_max = _safe_float_nutrient(g.get("low_max"))

    rng = merged.setdefault("Range", {})
    min_val = _safe_float_nutrient(rng.get("Min"))
    max_val = _safe_float_nutrient(rng.get("Max"))

    orig = merged.get("_original_range")
    if isinstance(orig, dict) and (orig.get("Min") is not None or orig.get("Max") is not None):
        original_min = _safe_float_nutrient(orig.get("Min"))
        original_max = _safe_float_nutrient(orig.get("Max"))
    else:
        original_min, original_max = min_val, max_val

    # Swap inverted ranges
    if min_val is not None and max_val is not None and min_val > max_val:
        min_val, max_val = max_val, min_val
        rng["Min"], rng["Max"] = min_val, max_val
        adjustments_made.append(
            f"Swapped Min and Max values for {nutrient_key} (Min was greater than Max)"
        )

    min_val = _safe_float_nutrient(rng.get("Min"))
    max_val = _safe_float_nutrient(rng.get("Max"))

    equal_target = _equal_min_max_target(rng)
    if equal_target is None:
        orig_rng = merged.get("_original_range")
        if isinstance(orig_rng, dict):
            equal_target = _equal_min_max_target(orig_rng)

    # --- Minimum alignment (salad.py ~4122–4210, diet flags false) ---
    # Equal Min/Max: user target is Min; Max is always DB high_max (never the duplicated user Max).
    if equal_target is not None:
        target = float(equal_target)
        rng["Min"] = target
        if low_max is not None and low_min is not None and target <= low_max:
            rng["Min"] = low_min
            rng["Max"] = low_max
            adjustments_made.append(
                f"minimum {nutrient_key} set to {_fmt(low_min)} and maximum set to {_fmt(low_max)} (database low-nutrient maximum; user target {_fmt(target)})"
            )
            min_val = low_min
            max_val = low_max
        elif band_max is not None:
            effective_max = band_max if target > (high_max or 0) and band_max > (high_max or 0) else high_max
            if effective_max is not None:
                rng["Max"] = effective_max
                label = (
                    "customization maximum"
                    if effective_max == band_max and band_max != high_max
                    else "database high-nutrient maximum"
                )
                adjustments_made.append(
                    f"maximum {nutrient_key} set to {_fmt(effective_max)} ({label}; user target {_fmt(target)})"
                )
                min_val = target
                max_val = effective_max
    elif high_min is not None and min_val is not None and min_val >= high_min:
        if (
            high_max is not None
            and min_val > high_max
            and band_max is not None
            and band_max > high_max
        ):
            if min_val > band_max:
                user_target = min_val
                rng["Min"] = band_max
                rng["Max"] = user_target
                adjustments_made.append(
                    f"minimum {nutrient_key} set to {_fmt(band_max)} (customization minimum) and "
                    f"maximum set to {_fmt(user_target)} (user target above customization maximum "
                    f"{_fmt(band_max)})"
                )
                min_val = band_max
                max_val = user_target
            else:
                rng["Max"] = band_max
                adjustments_made.append(
                    f"maximum {nutrient_key} set to {_fmt(band_max)} (customization maximum)"
                )
                max_val = band_max
        else:
            if min_val > high_min:
                new_min = high_min
                rng["Min"] = new_min
                adjustments_made.append(
                    f"minimum {nutrient_key} adjusted from {_fmt(original_min)} to {_fmt(new_min)} (capped to high-nutrient minimum)"
                )
                min_val = new_min
            if max_val is None and high_max is not None:
                rng["Max"] = high_max
                adjustments_made.append(
                    f"maximum {nutrient_key} auto-added as {_fmt(high_max)} (high-nutrient range)"
                )
                max_val = high_max
            elif (
                original_max is not None
                and max_val is not None
                and abs(float(original_max) - float(max_val)) > 1e-6
                and high_max is not None
                and max_val == high_max
            ):
                adjustments_made.append(
                    f"maximum {nutrient_key} adjusted from {_fmt(original_max)} to {_fmt(max_val)} (high-nutrient range)"
                )
    elif low_min is not None and min_val is not None and min_val < low_min:
        new_min = low_min
        rng["Min"] = new_min
        adjustments_made.append(
            f"minimum {nutrient_key} adjusted from {_fmt(original_min)} to {_fmt(new_min)} (capped to low-nutrient minimum)"
        )
        min_val = new_min
        if max_val is None and low_max is not None:
            rng["Max"] = low_max
            adjustments_made.append(
                f"maximum {nutrient_key} auto-added as {_fmt(low_max)} (low-nutrient range)"
            )
            max_val = low_max
    elif (
        low_min is not None
        and high_min is not None
        and min_val is not None
        and low_min < min_val < high_min
        and max_val is None
        and high_max is not None
    ):
        rng["Max"] = high_max
        adjustments_made.append(
            f"maximum {nutrient_key} auto-added as {_fmt(high_max)} (middle-range Min — flexibility up to high-nutrient maximum)"
        )
        max_val = high_max
    elif min_val is not None and max_val is None:
        if high_min is not None and min_val >= high_min and high_max is not None:
            rng["Max"] = high_max
            adjustments_made.append(
                f"maximum {nutrient_key} auto-added as {_fmt(high_max)} (edge case — high nutrient Min)"
            )
            max_val = high_max
        elif low_max is not None:
            rng["Max"] = low_max
            adjustments_made.append(
                f"maximum {nutrient_key} auto-added as {_fmt(low_max)} (edge case — low/moderate nutrient Min)"
            )
            max_val = low_max

    min_val = _safe_float_nutrient(rng.get("Min"))
    max_val = _safe_float_nutrient(rng.get("Max"))

    # Min-only + Max user path (salad ~4212–4227)
    if min_val is None and low_min is not None and low_max is not None and max_val is not None:
        if high_max is not None and max_val > high_max:
            new_min = high_min if high_min is not None else low_min
            rng["Min"] = new_min
            adjustments_made.append(
                f"minimum {nutrient_key} auto-added as {_fmt(new_min)} (high-nutrient minimum — user max {_fmt(max_val)} above {_fmt(high_max)})"
            )
            min_val = new_min
        else:
            rng["Min"] = low_min
            adjustments_made.append(
                f"minimum {nutrient_key} auto-added as {_fmt(low_min)} (low-nutrient minimum)"
            )
            min_val = low_min

    min_val = _safe_float_nutrient(rng.get("Min"))
    max_val = _safe_float_nutrient(rng.get("Max"))

    # --- Maximum alignment (salad.py ~4229–4272) ---
    is_high_nutrient_request = False
    if original_min is not None and high_min is not None:
        try:
            is_high_nutrient_request = float(original_min) >= float(high_min)
        except (TypeError, ValueError):
            pass
    if not is_high_nutrient_request and min_val is not None and high_min is not None:
        try:
            is_high_nutrient_request = float(min_val) >= float(high_min)
        except (TypeError, ValueError):
            pass

    cap_max = band_max if is_high_nutrient_request and band_max is not None else high_max
    if max_val is not None and cap_max is not None:
        if is_high_nutrient_request:
            if max_val > cap_max:
                rng["Max"] = cap_max
                cap_label = (
                    "customization maximum"
                    if cap_max == band_max and band_max != high_max
                    else "high-nutrient range"
                )
                adjustments_made.append(
                    f"maximum {nutrient_key} adjusted from "
                    f"{_fmt(original_max if original_max is not None else max_val)} to "
                    f"{_fmt(cap_max)} ({cap_label})"
                )
            elif (
                original_max is not None
                and abs(float(original_max) - float(max_val)) > 1e-6
                and abs(float(max_val) - float(high_max)) < 1e-6
            ):
                adjustments_made.append(
                    f"maximum {nutrient_key} adjusted from {_fmt(original_max)} to {_fmt(max_val)} (high-nutrient range)"
                )
        elif low_max is not None and max_val > low_max:
            rng["Max"] = low_max
            adjustments_made.append(
                f"maximum {nutrient_key} adjusted from {_fmt(original_max if original_max is not None else max_val)} to {_fmt(low_max)} (capped to low-nutrient maximum)"
            )

    if adjustments_made:
        merged["_guidelines_adjusted"] = True
        merged["_adjustment_messages"] = list(adjustments_made)
        if not isinstance(merged.get("_original_range"), dict):
            merged["_original_range"] = {"Min": original_min, "Max": original_max}

    return adjustments_made


def _merge_and_align_nutrient_filters(
    nutrient_filters: List[Dict[str, Any]],
    nutrition_constraints: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    One filter per nutrient: AND semantics — max of all Mins, min of all Maxes.
    Then align ranges to DB bands (Salad-style parity).
    """
    messages: List[str] = []
    if not nutrient_filters:
        return [], messages

    key_order: List[str] = []
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for nf in nutrient_filters:
        raw = (nf.get("Nutrient") or "").strip()
        key = _canonical_nutrient_key(raw)
        if not key:
            messages.append("Skipped nutrient filter with empty Nutrient name")
            continue
        if key not in groups:
            key_order.append(key)
            groups[key] = []
        groups[key].append(nf)

    merged_list: List[Dict[str, Any]] = []
    for key in key_order:
        group = groups[key]
        diet_item = next(
            (x for x in group if x.get("_diet_converted") and x.get("_original_diet")), None
        )
        user_item = next((x for x in group if not x.get("_diet_converted")), None)

        if len(group) == 1:
            item0 = group[0]
            rng = dict(item0.get("Range") or {})
            merged = {"Nutrient": key, "Range": rng}
            if item0.get("_diet_converted") and item0.get("_original_diet"):
                merged["_diet_converted"] = True
                merged["_original_diet"] = item0.get("_original_diet")
            if item0.get("_user_original_request"):
                merged["_user_original_request"] = True
                merged["_original_range"] = dict(
                    item0.get("_original_range") or (item0.get("Range") or {})
                )
            elif not item0.get("_diet_converted"):
                r0 = item0.get("Range") or {}
                if r0.get("Min") is not None or r0.get("Max") is not None:
                    merged["_user_original_request"] = True
                    merged["_original_range"] = dict(r0)
        else:
            mins: List[float] = []
            maxs: List[float] = []
            for item in group:
                r = item.get("Range") or {}
                mn = _safe_float_nutrient(r.get("Min"))
                mx = _safe_float_nutrient(r.get("Max"))
                if mn is not None:
                    mins.append(mn)
                if mx is not None:
                    maxs.append(mx)
            rng = {}
            user_equal_target = None
            user_min_only = False
            user_max_only = False
            user_orig: Dict[str, Any] = {}
            if user_item:
                user_orig = dict(
                    user_item.get("_original_range") or user_item.get("Range") or {}
                )
                user_equal_target = _equal_min_max_target(user_orig)
                user_min_only = (
                    user_orig.get("Min") is not None and user_orig.get("Max") is None
                )
                user_max_only = (
                    user_orig.get("Max") is not None and user_orig.get("Min") is None
                )
            if user_min_only:
                mn = _safe_float_nutrient(user_orig.get("Min"))
                if mn is not None:
                    rng["Min"] = mn
            elif user_max_only:
                mx = _safe_float_nutrient(user_orig.get("Max"))
                if mx is not None:
                    rng["Max"] = mx
            else:
                if mins:
                    rng["Min"] = max(mins)
                # Equal Min/Max from user: align step applies DB high_max.
                if user_equal_target is None and maxs:
                    rng["Max"] = min(maxs)
                elif user_equal_target is not None:
                    rng["Min"] = user_equal_target

            merged = {"Nutrient": key, "Range": rng}
            if diet_item:
                merged["_diet_converted"] = True
                merged["_original_diet"] = diet_item.get("_original_diet")
            if user_item:
                merged["_user_original_request"] = True
                ur = user_item.get("_original_range")
                if isinstance(ur, dict) and (ur.get("Min") is not None or ur.get("Max") is not None):
                    merged["_original_range"] = dict(ur)
                else:
                    merged["_original_range"] = dict(user_item.get("Range") or {})

        _salad_like_align_nutrient_filter(merged, nutrition_constraints)
        merged_list.append(merged)

    return merged_list, messages


def preprocess_heybo_filters(filters: dict) -> dict:
    """
    Merge nutrient-style diet labels into NutrientFilters using heybo.nutrition_constraint_information,
    then dedupe by nutrient (diet + explicit targets → one row per nutrient, Salad-style).
    """
    from .nutrition_constraints import (
        convert_heybo_diet_to_nutrient_filters,
        infer_heybo_diets_from_nutrient_filters,
        load_heybo_nutrition_constraints_from_db,
    )

    filters = filters.copy()
    if filters.get("_heybo_only_mode_generation"):
        _page = filters.get("Page", 1)
        try:
            _page = int(_page)
        except (TypeError, ValueError):
            _page = 1
        if _page < 1:
            _page = 1
        filters["Page"] = _page
        filters["DietFilters"] = []
        filters["NutrientFilters"] = []
        filters["_nutrient_preprocess_messages"] = []
        filters["_heybo_guideline_adjustments_from_preprocess"] = []
        return filters
    # Pagination (Salad parity): Page is 1-based; used for logging and client contract.
    _page = filters.get("Page", 1)
    try:
        _page = int(_page)
    except (TypeError, ValueError):
        _page = 1
    if _page < 1:
        _page = 1
    filters["Page"] = _page

    raw_diet = filters.get("DietFilters")
    if isinstance(raw_diet, dict):
        filters["_heybo_message_diet_filters_dict"] = dict(raw_diet)
    diet_filters = _get_filter_list(filters.get("DietFilters"))
    initial_diets = list(diet_filters)
    nutrient_filters = list(filters.get("NutrientFilters", []))
    for nf in nutrient_filters:
        if isinstance(nf, dict) and not nf.get("_diet_converted"):
            r = nf.get("Range") or {}
            if r.get("Min") is not None or r.get("Max") is not None:
                nf.setdefault("_user_original_request", True)
                nf.setdefault("_original_range", dict(r))
    nutrition_constraints = load_heybo_nutrition_constraints_from_db()
    guideline_adjustments: List[str] = []
    remaining_diets: List[str] = []
    converted_all: List[Dict[str, Any]] = []
    for diet in diet_filters:
        converted = convert_heybo_diet_to_nutrient_filters(
            [diet], nutrition_constraints, silent=True
        )
        if converted:
            converted_all.extend(converted)
        else:
            remaining_diets.append(diet)

    has_explicit_numeric_nutrients = any(
        isinstance(nf, dict)
        and not nf.get("_diet_converted")
        and (
            (nf.get("Range") or {}).get("Min") is not None
            or (nf.get("Range") or {}).get("Max") is not None
        )
        for nf in nutrient_filters
    )
    inferred_diets: List[str] = []
    if not initial_diets and not has_explicit_numeric_nutrients:
        inferred_diets = infer_heybo_diets_from_nutrient_filters(
            nutrient_filters, nutrition_constraints
        )
        if inferred_diets:
            guideline_adjustments.append(
                f"Dynamically inferred diets from nutrient targets: {', '.join(inferred_diets)}"
            )
            for diet in inferred_diets:
                converted = convert_heybo_diet_to_nutrient_filters(
                    [diet], nutrition_constraints, silent=True
                )
                if converted:
                    converted_all.extend(converted)
                else:
                    remaining_diets.append(diet)

    if converted_all:
        guideline_adjustments.insert(
            0,
            f"Converted {len(converted_all)} nutrient-based diet filters to nutrient constraints",
        )

    combined = nutrient_filters + converted_all
    merged, preprocess_messages = _merge_and_align_nutrient_filters(
        combined, nutrition_constraints
    )
    for m in merged:
        for line in m.get("_adjustment_messages") or []:
            if line not in guideline_adjustments:
                guideline_adjustments.append(line)

    filters["DietFilters"] = remaining_diets
    filters["NutrientFilters"] = merged
    filters["_nutrient_preprocess_messages"] = preprocess_messages
    filters["_heybo_guideline_adjustments_from_preprocess"] = guideline_adjustments
    return filters


def apply_allergen_filter(
    df: pd.DataFrame,
    user_input: dict,
    global_validations: dict,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Drop every catalog row for an ingredient if any of its SKUs lists an excluded allergen tag."""
    allergen_filters = _get_filter_list(user_input.get("AllergenFilters"))
    if not allergen_filters:
        global_validations["allergen_exclusions"].append("No allergen filters applied")
        return df, None
    if "allergens" not in df.columns:
        global_validations["allergen_exclusions"].append(
            "Allergen filters requested but ingredients have no allergens column — filter not applied"
        )
        return df, None
    original_count = len(df)
    excluded_names = ingredient_names_excluded_by_allergen(df, allergen_filters)
    row_mask = df["allergens"].apply(
        lambda raw: ingredient_lists_excluded_allergen(raw, allergen_filters)
    )
    tagged_row_count = int(row_mask.sum())
    filtered = df[~df["ingredient_name"].isin(excluded_names)]
    removed = original_count - len(filtered)
    name_note = (
        f" ({len(excluded_names)} ingredient name(s) removed because at least one SKU lists the tag)"
        if excluded_names
        else ""
    )
    global_validations["allergen_exclusions"].append(
        f"Excluded allergen tags: {', '.join(allergen_filters)} — removed {removed} catalog row(s)"
        f"{name_note} ({tagged_row_count} row(s) directly tagged in allergens)"
    )
    if filtered.empty:
        return filtered, "No ingredients left after allergen filtering"
    return filtered, None


def apply_diet_filter(
    df: pd.DataFrame,
    user_input: dict,
    global_validations: dict,
) -> Tuple[pd.DataFrame, List[str], Optional[str]]:
    """Apply diet filters with fallback; returns (filtered_df, fallback_categories, error_or_None)."""
    diet_filters = _get_filter_list(user_input.get("DietFilters"))
    if not diet_filters:
        global_validations["diet_fallbacks"].append("No diet filters applied")
        return df, [], None
    filtered, fallback_categories = apply_heybo_diet_filters_with_fallback(df, diet_filters)
    # Do not log "Applied diet filters: … - Removed N" to diet_fallbacks — that string was copied
    # into message_to_user; Salad-style UX is diet in the opener / "Following dietary preferences"
    # from bowl validations, not raw filter pipeline text.
    if filtered.empty:
        return filtered, fallback_categories, "No ingredients left after diet filtering"
    return filtered, fallback_categories, None


def _ingredient_cuisine_tokens(cuisine_raw) -> List[str]:
    """salad.py apply_filters (~8087–8090): split comma list or list column into lowercase tokens."""
    if cuisine_raw is None or (isinstance(cuisine_raw, float) and pd.isna(cuisine_raw)):
        return []
    if isinstance(cuisine_raw, list):
        return [str(c).strip().lower() for c in cuisine_raw if str(c).strip()]
    s = str(cuisine_raw).strip()
    if not s:
        return []
    return [c.strip().lower() for c in s.split(",") if c.strip()]


def _parse_cuisine_include_exclude(user_input: dict) -> Tuple[List[str], List[str]]:
    """
    salad.py apply_filters (~8063–8064): dict keys with True = include, False = exclude.
    List-style CuisineFilters is treated as included-only (OR).
    """
    cf = user_input.get("CuisineFilters")
    if not cf:
        return [], []
    if isinstance(cf, dict):
        included = [str(k).strip().lower() for k, v in cf.items() if v is True]
        excluded = [str(k).strip().lower() for k, v in cf.items() if v is False]
        return included, excluded
    if isinstance(cf, list):
        return [str(x).strip().lower() for x in cf if str(x).strip()], []
    return [], []


def _row_matches_salad_cuisine(
    tokens: List[str], included_cuisines: List[str], excluded_cuisines: List[str]
) -> bool:
    """salad.py apply_filters (~8092–8103): OR across included; then apply exclusions."""
    should_include = True
    if included_cuisines:
        should_include = any(c in tokens for c in included_cuisines)
    if excluded_cuisines and should_include:
        should_include = not any(c in tokens for c in excluded_cuisines)
    if not included_cuisines and not excluded_cuisines:
        should_include = True
    return should_include

def apply_cuisine_filter(
    df: pd.DataFrame,
    user_input: dict,
    global_validations: dict,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    salad.py apply_filters (~8061–8133): cuisine does **not** remove rows from the dataframe.

    Matching ingredients get tracked for priority during bowl fill; the full filtered catalog
    remains available for structural variety. If nothing matches, relax like Salad (“suggestion
    only”) and continue without priority tags.
    """
    user_input.pop("_cuisine_matched_ingredient_names", None)
    user_input.pop("_heybo_true_cuisine_by_category", None)
    user_input.pop("_heybo_true_cuisine_names", None)
    included, excluded = _parse_cuisine_include_exclude(user_input)
    if not included and not excluded:
        global_validations["cuisine_matches"].append("No cuisine filters applied")
        return df, None
    if user_input.get("_relax_cuisine_filter"):
        global_validations["cuisine_matches"].append(
            "Cuisine filter relaxed by progressive order - using all available ingredients"
        )
        return df, None

    if "cuisine" not in df.columns:
        global_validations["cuisine_matches"].append(
            "No ingredients matched requested cuisine filters — treating as suggestion only (no cuisine column)"
        )
        return df, None

    matched_mask = df["cuisine"].apply(
        lambda raw: _row_matches_salad_cuisine(
            _ingredient_cuisine_tokens(raw), included, excluded
        )
    )
    names = set(
        df.loc[matched_mask, "ingredient_name"].dropna().astype(str).str.strip().unique()
    )
    names.discard("")

    if not names:
        global_validations["cuisine_matches"].append(
            "No ingredients matched requested cuisine filters — treating as suggestion only "
            "(salad.py parity; full catalog used for selection)"
        )
        return df, None

    # Cuisine is not a hard constraint. If a category has too few unique matches,
    # include extra allergen/diet-safe items from that category in the priority set.
    thin_floor = 2
    true_by_cat: Dict[str, List[str]] = {}
    for cat in df["category"].dropna().unique():
        cat_df = df[df["category"] == cat]
        if cat_df.empty:
            continue
        cat_matched = {
            str(x).strip()
            for x in df.loc[
                matched_mask & (df["category"] == cat), "ingredient_name"
            ].dropna().astype(str)
            if str(x).strip()
        }
        _heybo_record_true_matches(true_by_cat, cat, cat_matched)
        cat_all = {
            str(x).strip()
            for x in cat_df["ingredient_name"].dropna().astype(str)
            if str(x).strip()
        }
        if 0 < len(cat_matched) < thin_floor and len(cat_all) > len(cat_matched):
            extra_n = len(cat_all - cat_matched)
            names |= cat_all
            user_input["_heybo_cuisine_pool_topped_up"] = True
            global_validations["cuisine_matches"].append(
                f"Cuisine: only {len(cat_matched)} unique match(es) in category {cat!r} — "
                f"added {extra_n} extra allergen/diet-safe ingredient(s) "
                "(cuisine is not a hard constraint)"
            )

    user_input["_heybo_true_cuisine_by_category"] = true_by_cat
    user_input["_heybo_true_cuisine_names"] = {
        n for vals in true_by_cat.values() for n in vals
    }
    user_input["_cuisine_matched_ingredient_names"] = names
    inc_disp = ", ".join(included) if included else "(none)"
    exc_disp = ", ".join(excluded) if excluded else "(none)"
    global_validations["cuisine_matches"].append(
        f"Cuisine filter (salad-style priority): {len(names)} ingredient(s) match "
        f"(include: {inc_disp}; exclude: {exc_disp}) — fill prefers these; pool unchanged."
    )
    return df, None


def _heybo_row_matches_preparation_method_salad_style(
    prep_raw,
    ingredient_name: str,
    included_methods: List[str],
    exclude_methods: List[str],
) -> bool:
    """
    Match salad.py apply_filters (~7974–8033): comma-split prep tags; include if any tag equals
    a selected method (case-insensitive) OR any selected method appears as substring in ingredient
    name; then apply exclude_methods on prep tags and name.
    """
    ingredient_name_lower = (ingredient_name or "").lower()
    if prep_raw is None or (isinstance(prep_raw, float) and pd.isna(prep_raw)):
        prep_method_list_lower: List[str] = []
    else:
        prep_method_list_lower = [
            m.strip().lower() for m in str(prep_raw).split(",") if m.strip()
        ]

    included_lower = [m.strip().lower() for m in included_methods if m and str(m).strip()]
    exclude_lower = [m.strip().lower() for m in exclude_methods if m and str(m).strip()]

    should_include = True
    if included_lower:
        matches_prep = any(m in included_lower for m in prep_method_list_lower)
        matches_name = any(method_lower in ingredient_name_lower for method_lower in included_lower)
        should_include = matches_prep or matches_name

    if exclude_lower and should_include:
        if any(m in exclude_lower for m in prep_method_list_lower):
            should_include = False
        if should_include:
            for method_lower in exclude_lower: 
                if method_lower in ingredient_name_lower:
                    should_include = False
                    break

    if not included_lower and not exclude_lower:
        should_include = True

    return should_include


def apply_preparation_method_filter(
    df: pd.DataFrame,
    user_input: dict,
    global_validations: dict,
    df_before_prep: pd.DataFrame,
) -> pd.DataFrame:
    """
    Salad CYO–aligned preparation filter (apply_filters ~7954–8059): OR across user-selected methods
    (any tag matches any selection, or method substring in ingredient name). If nothing matches,
    keep the pre-prep dataframe (allergen/diet/cuisine only) — same spirit as salad's
    'treating as suggestion only' without widening past upstream filters.
    """
    preparation_method = user_input.get("PreparationMethod", {})
    if not preparation_method:
        global_validations["preparation_method_matches"].append("No preparation method filters applied")
        return df
    if user_input.get("_relax_preparation_method_filter"):
        global_validations["preparation_method_matches"].append(
            "Preparation method filter relaxed by progressive order - using all available ingredients"
        )
        return df

    included_methods = [str(k).strip() for k, v in preparation_method.items() if v is True]
    exclude_methods = [str(k).strip() for k, v in preparation_method.items() if v is False]
    if not included_methods and not exclude_methods:
        global_validations["preparation_method_matches"].append("No preparation method filters applied")
        return df

    original_count = len(df)
    if "preparation_method" not in df.columns or "ingredient_name" not in df.columns:
        global_validations["preparation_method_matches"].append(
            "Missing preparation_method or ingredient_name column — skipping prep filter"
        )
        return df

    user_input.pop("_heybo_true_prep_by_category", None)
    base_df = df_before_prep
    mask = base_df.apply(
        lambda row: _heybo_row_matches_preparation_method_salad_style(
            row.get("preparation_method"),
            row.get("ingredient_name", ""),
            included_methods,
            exclude_methods,
        ),
        axis=1,
    )
    if not mask.any():
        global_validations["preparation_method_matches"].append(
            f"No ingredients matched preparation methods {included_methods!r} — using pool without prep filter "
            "(Salad-style relaxation; allergen/diet/cuisine still apply)"
        )
        return base_df.copy()

    strict_df = base_df[mask]
    # Per-category relaxation (Salad-style): when user *includes* prep methods, never zero out a
    # category that still has options upstream — keep full category if strict match is empty there.
    # Exclude-only requests never relax (would re-introduce excluded items).
    relax_when_empty = bool(included_methods)
    thin_floor = 2
    parts: List[pd.DataFrame] = []
    true_by_cat: Dict[str, List[str]] = {}
    for cat in base_df["category"].dropna().unique():
        cat_full = base_df[base_df["category"] == cat]
        cat_strict = strict_df[strict_df["category"] == cat]
        if cat_strict.empty:
            if not cat_full.empty and relax_when_empty:
                global_validations["preparation_method_matches"].append(
                    f"Preparation method: no match in category {cat!r} — keeping all {len(cat_full)} "
                    "ingredient(s) there (Salad-style per-category relaxation)"
                )
                parts.append(cat_full)
            continue
        picked_names = set(cat_strict["ingredient_name"].astype(str))
        _heybo_record_true_matches(true_by_cat, cat, picked_names)
        fallback_unique = int(cat_full["ingredient_name"].astype(str).nunique())
        if (
            relax_when_empty
            and len(picked_names) < thin_floor
            and fallback_unique > len(picked_names)
        ):
            extra = cat_full[~cat_full["ingredient_name"].astype(str).isin(picked_names)]
            extra_n = (
                int(extra["ingredient_name"].astype(str).nunique()) if not extra.empty else 0
            )
            global_validations["preparation_method_matches"].append(
                f"Preparation method: only {len(picked_names)} unique match(es) in "
                f"category {cat!r} — added {extra_n} extra allergen/diet-safe ingredient(s) "
                "(prep is not a hard constraint)"
            )
            user_input["_heybo_prep_pool_topped_up"] = True
            parts.append(pd.concat([cat_strict, extra], axis=0) if extra_n else cat_strict)
        else:
            parts.append(cat_strict)

    user_input["_heybo_true_prep_by_category"] = true_by_cat
    result = pd.concat(parts, axis=0) if parts else base_df.iloc[0:0]
    removed = original_count - len(result)
    detail = ", ".join(included_methods) if included_methods else "(excludes only)"
    global_validations["preparation_method_matches"].append(
        f"Applied preparation method filter (per category, Salad-style): {detail} - Removed {removed} ingredients"
    )
    return result


def apply_flavor_filter(
    df: pd.DataFrame,
    user_input: dict,
    global_validations: dict,
    df_fallback: pd.DataFrame,
    heybo_cfg: Optional[dict] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Salad-style per-category flavor matching: OR across active flavors with per-flavor High→Low→No
    fallback on scores. Flavor is not a hard constraint (unlike allergen/diet):
    if a category has zero matches, or too few unique matches to build variety, add extra
    ingredients from the pre-flavor pool (allergen + diet + cuisine + prep still apply).
    """
    flavor_preferences = user_input.get("FlavorPreferences", {})
    if not flavor_preferences:
        global_validations["flavor_adjustments"].append("No flavor preference filters applied")
        return df, None
    if user_input.get("_relax_flavor_filter"):
        global_validations["flavor_adjustments"].append(
            "Flavor filter relaxed by progressive order - using all available ingredients"
        )
        return df, None
    user_input.pop("_heybo_true_flavor_by_category", None)
    flavor_thresholds = (heybo_cfg or {}).get("flavor_thresholds") or {}
    if not flavor_thresholds:
        global_validations["flavor_adjustments"].append(
            "Flavor thresholds not loaded — cannot apply FlavorPreferences"
        )
        return df, "Heybo flavor thresholds missing; check heybo.flavor_profile_details"

    fallback_order = {
        "High": ["High", "Low", "No"],
        "Low": ["Low", "High", "No"],
        "No": ["No", "Low", "High"],
    }

    # One row per (user-facing flavor label, requested intensity) → bands actually used when
    # the requested band had zero rows in that category (Salad-style); summarized once below.
    band_fallback_used: Dict[Tuple[str, str], Set[str]] = {}

    def flavor_requested_band_mask(frame: pd.DataFrame) -> pd.Series:
        """Requested intensity only (e.g. High). Used for FEW/MANY true matches."""
        matched = pd.Series(False, index=frame.index)
        for flavor, intensity in flavor_preferences.items():
            if not intensity or (isinstance(intensity, str) and not intensity.strip()):
                continue
            if isinstance(intensity, str):
                intensity = intensity.strip()
            flavor_col = str(flavor).lower()
            thresholds_for_flavor = flavor_thresholds.get(flavor_col)
            if flavor_col not in frame.columns or not thresholds_for_flavor:
                continue
            if intensity not in thresholds_for_flavor:
                continue
            flavor_series = pd.to_numeric(frame[flavor_col], errors="coerce").fillna(0)
            mn, mx = thresholds_for_flavor[intensity]
            matched = matched | ((flavor_series >= mn) & (flavor_series <= mx))
        return matched

    def flavor_or_mask(frame: pd.DataFrame) -> pd.Series:
        matched = pd.Series(False, index=frame.index)
        for flavor, intensity in flavor_preferences.items():
            if not intensity or (isinstance(intensity, str) and not intensity.strip()):
                continue
            if isinstance(intensity, str):
                intensity = intensity.strip()
            flavor_col = str(flavor).lower()
            thresholds_for_flavor = flavor_thresholds.get(flavor_col)
            if flavor_col not in frame.columns or not thresholds_for_flavor or intensity not in fallback_order:
                continue
            flavor_matched = pd.Series(False, index=frame.index)
            flavor_series = pd.to_numeric(frame[flavor_col], errors="coerce").fillna(0)
            for attempt in fallback_order[intensity]:
                if attempt not in thresholds_for_flavor:
                    continue
                mn, mx = thresholds_for_flavor[attempt]
                current = (flavor_series >= mn) & (flavor_series <= mx)
                if current.any():
                    flavor_matched = current
                    if attempt != intensity:
                        key = (str(flavor), intensity)
                        band_fallback_used.setdefault(key, set()).add(attempt)
                    break
            matched = matched | flavor_matched
        return matched

    applicable = False
    for flavor, intensity in flavor_preferences.items():
        if not intensity or (isinstance(intensity, str) and not intensity.strip()):
            continue
        if isinstance(intensity, str):
            intensity = intensity.strip()
        fc = str(flavor).lower()
        if fc in df.columns and flavor_thresholds.get(fc) and intensity in fallback_order:
            applicable = True
            break
    if not applicable:
        global_validations["flavor_adjustments"].append(
            "Flavor preferences set but no matching flavor columns/thresholds on ingredients — skipping flavor filter"
        )
        return df, None

    original_count = len(df)
    active_flavors = [f"{k}: {v}" for k, v in flavor_preferences.items() if v]
    flavor_summary = ", ".join(active_flavors)

    parts: List[pd.DataFrame] = []
    # Prefer flavor matches; if a category has fewer than this many unique items,
    # top up from the pre-flavor pool so nutrient packing is not stuck on 1 SKU.
    flavor_thin_unique_floor = 2
    true_by_cat: Dict[str, List[str]] = {}
    for cat in df["category"].dropna().unique():
        cat_df = df[df["category"] == cat]
        if cat_df.empty:
            continue
        mask = flavor_or_mask(cat_df)
        picked = cat_df[mask]
        if picked.empty:
            global_validations["flavor_adjustments"].append(
                f"Flavor ({flavor_summary}): no matches in category {cat!r} — keeping all {len(cat_df)} "
                "ingredient(s) there (Salad-style per-category relaxation)"
            )
            parts.append(cat_df)
            continue
        picked_names = set(picked["ingredient_name"].astype(str))
        requested = cat_df[flavor_requested_band_mask(cat_df)]
        requested_names = set(requested["ingredient_name"].astype(str)) if not requested.empty else set()
        _heybo_record_true_matches(true_by_cat, cat, requested_names)
        fallback_unique = int(cat_df["ingredient_name"].astype(str).nunique())
        if len(picked_names) < flavor_thin_unique_floor and fallback_unique > len(picked_names):
            extra = cat_df[~cat_df["ingredient_name"].astype(str).isin(picked_names)]
            extra_n = (
                int(extra["ingredient_name"].astype(str).nunique()) if not extra.empty else 0
            )
            global_validations["flavor_adjustments"].append(
                f"Flavor ({flavor_summary}): only {len(picked_names)} unique match(es) in "
                f"category {cat!r} — added {extra_n} extra allergen/diet-safe ingredient(s) "
                "(flavor is not a hard constraint)"
            )
            user_input["_heybo_flavor_pool_topped_up"] = True
            user_input["flavor_relaxation_enabled"] = True
            parts.append(pd.concat([picked, extra], axis=0) if extra_n else picked)
        else:
            parts.append(picked)

    user_input["_heybo_true_flavor_by_category"] = true_by_cat
    df = pd.concat(parts, axis=0) if parts else df.iloc[0:0]
    if df.empty and original_count > 0:
        df = df_fallback.copy()
        global_validations["flavor_adjustments"].append(
            "No ingredients after per-category flavor merge — using full pre-flavor pool "
            "(allergen/diet/cuisine/prep still apply)"
        )

    for (flav_label, req_int), bands in sorted(band_fallback_used.items()):
        bands_fmt = ", ".join(sorted(bands))
        global_validations["flavor_adjustments"].append(
            f"{flav_label} ({req_int} preferred): in some categories nothing scored in the {req_int} band — "
            f"used {bands_fmt} band(s) there (normal fallback; selections still follow your flavor preferences)."
        )

    if active_flavors:
        removed = original_count - len(df)
        global_validations["flavor_adjustments"].append(
            f"Flavor filter applied for {flavor_summary}: kept flavor-matched ingredients "
            f"({len(df)} in pool vs {original_count} before; {removed} not selected for this pass)."
        )
    if df.empty:
        return df, "No ingredients left after flavor preference filtering"
    return df, None


def build_heybo_strict_safety_catalog(
    df_heybo: pd.DataFrame,
    user_input: dict,
    global_validations: dict,
) -> pd.DataFrame:
    """
    Full location catalog narrowed to allergen + strict diet safety only.

    Used to top up category minimums when flavor/cuisine/prep/nutrient/light-hearty
    filters leave too few ingredients — never bypasses allergen or diet rules.
    """
    from .diet import filter_dataframe_by_strict_diets

    if user_input.get("_heybo_only_mode_generation"):
        return df_heybo.iloc[0:0].copy()
    df = df_heybo.copy()
    df, err = apply_allergen_filter(df, user_input, global_validations)
    if err or df.empty:
        return df.iloc[0:0].copy() if not df.empty else df
    diet_filters = _get_filter_list(user_input.get("DietFilters"))
    if diet_filters:
        df = filter_dataframe_by_strict_diets(df, diet_filters)
        if df.empty:
            return df
        global_validations.setdefault("minimum_fill_catalog", []).append(
            f"Strict safety catalog (allergen + diet: {', '.join(diet_filters)}): "
            f"{len(df)} ingredient row(s) available for minimum-fill top-up"
        )
    else:
        global_validations.setdefault("minimum_fill_catalog", []).append(
            f"Strict safety catalog (allergen only): {len(df)} ingredient row(s) for minimum-fill top-up"
        )
    return df


def minimum_fill_pools_by_category(strict_safety_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Unique ingredient_name per category from the strict-safety catalog."""
    if strict_safety_df is None or strict_safety_df.empty:
        return {}
    out: Dict[str, List[str]] = {}
    for cat, grp in strict_safety_df.groupby("category"):
        out[str(cat)] = list(dict.fromkeys(grp["ingredient_name"].astype(str).tolist()))
    return out


def apply_all_ingredient_filters(
    df_heybo: pd.DataFrame,
    user_input: dict,
    global_validations: dict,
    heybo_cfg: Optional[dict] = None,
) -> Tuple[pd.DataFrame, List[str], Optional[str]]:
    """
    Apply all filters in order: allergen → diet → cuisine (must-include list; pool unchanged) →
    preparation method → flavor → light/hearty.
    Returns (filtered_ingredients, fallback_categories, error_message).
    If error_message is not None, caller should return {"error": error_message}.
    """
    fallback_categories: List[str] = []
    if user_input.get("_heybo_only_mode_generation"):
        global_validations["allergen_exclusions"].append(
            "Only mode: allergen filters not applied to ingredient pool"
        )
        global_validations["diet_fallbacks"].append("Only mode: diet filters not applied to ingredient pool")
        global_validations["filter_summary"].append(
            f"Only mode: using full catalog ({len(df_heybo)} ingredients) for Include/Extra selection"
        )
        return df_heybo.copy(), fallback_categories, None

    df = df_heybo.copy()

    df, err = apply_allergen_filter(df, user_input, global_validations)
    if err:
        return df, fallback_categories, err

    df, fallback_categories, err = apply_diet_filter(df, user_input, global_validations)
    if err:
        return df, fallback_categories, err

    df, err = apply_cuisine_filter(df, user_input, global_validations)
    if err:
        return df, fallback_categories, err

    # Snapshots for Salad-style relaxation: never widen past allergen/diet into raw df_heybo
    # (cuisine does not shrink the dataframe — salad.py parity).
    df_before_prep = df.copy()
    df = apply_preparation_method_filter(df, user_input, global_validations, df_before_prep)

    df_before_flavor = df.copy()
    df, err = apply_flavor_filter(df, user_input, global_validations, df_before_flavor, heybo_cfg)
    if err:
        return df, fallback_categories, err

    cat_limits = (heybo_cfg or {}).get("category_limits") or {}
    df = apply_light_hearty_filter(df, user_input, global_validations, cat_limits)

    global_validations["filter_summary"].append(f"Filtered to {len(df)} available ingredients")
    return df, fallback_categories, None
