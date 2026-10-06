"""
Heybo diet: apply diet filters with fallback, check nutrient filters.
"""
import pandas as pd

from .diet_constants import (
    DIET_HIERARCHY,
    HEYBO_NUTRIENT_RELAXATION_PRIORITY_ORDER,
    NUTRIENT_COLUMNS,
    NUTRIENT_NAME_MAP,
)
from .nutrition_constraints import (
    load_heybo_nutrition_constraints_from_db,
    load_heybo_nutrient_priority_order_from_db,
)

RELAXATION_PERCENTAGES = [0, 15, 30, 50, 70, 85]

# Bowl totals vs heybo.nutrition_constraint_information balanced_min / balanced_max (Salad parity).
BALANCED_DIET_NUTRIENTS = ("calories_kCal", "carbs_g", "protein_g", "total_fat_g")

# Nutrients enforced by Balanced bowl validation — Light/Hearty-style overlap suppress.
BALANCED_CRITERIA_NUTRIENT_KEYS = frozenset(BALANCED_DIET_NUTRIENTS)


def balanced_criteria_nutrient_overlap(nutrient_filters) -> list:
    """
    Return Balanced criteria nutrient keys that also appear in NutrientFilters
    (calories_kCal, carbs_g, protein_g, total_fat_g). Empty ⇒ no overlap.
    """
    active = set(heybo_active_nutrient_filter_keys(nutrient_filters or []))
    return sorted(active & BALANCED_CRITERIA_NUTRIENT_KEYS)


def balanced_suppressed_by_nutrient_filters(nutrient_filters) -> bool:
    """
    True when any NutrientFilters nutrient is part of Balanced diet criteria.

    In that case Balanced is ignored entirely so the explicit nutrient filter
    owns those targets (same priority rule as Light/Hearty vs NutrientFilters).
    Other preference filters (price, cuisine, diet, …) do not suppress Balanced.
    """
    return bool(balanced_criteria_nutrient_overlap(nutrient_filters))


# When user sends Min == Max (point target), treat as "at least Min" and open Max by this
# absolute slack before generation. Exact bowl totals are discrete; requiring == wastes attempts.
EQUAL_NUTRIENT_POINT_TARGET_SLACK_DEFAULT = 15.0
EQUAL_NUTRIENT_POINT_TARGET_SLACK_BY_KEY = {
    "protein_g": 15.0,
    "carbs_g": 15.0,
    "total_fat_g": 15.0,
    "saturated_fat_g": 5.0,
    "fiber_g": 5.0,
    "sugar_g": 10.0,
    "added_sugar_g": 10.0,
    "calories_kCal": 75.0,
    "sodium_mg": 150.0,
    "calcium_mg": 100.0,
    "iron_mg": 5.0,
    "potassium_mg": 200.0,
    "phosphorus_mg": 100.0,
    "cholesterol_mg": 25.0,
    "vitamin_d_mcg": 5.0,
}


def _heybo_equal_nutrient_point_target_slack(nutrient_key: str) -> float:
    """Absolute Max headroom when Min == Max for ``nutrient_key``."""
    key = (nutrient_key or "").strip()
    if key in EQUAL_NUTRIENT_POINT_TARGET_SLACK_BY_KEY:
        return float(EQUAL_NUTRIENT_POINT_TARGET_SLACK_BY_KEY[key])
    low = key.lower()
    if "kcal" in low or low.startswith("calorie"):
        return 75.0
    if low.endswith("_mg"):
        return 150.0
    if low.endswith("_mcg"):
        return 5.0
    if low.endswith("_g"):
        return 15.0
    return float(EQUAL_NUTRIENT_POINT_TARGET_SLACK_DEFAULT)


def _heybo_safe_float_nutrient(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _heybo_equal_min_max_target(rng):
    """When Min and Max are the same, return that single target value."""
    mn = _heybo_safe_float_nutrient((rng or {}).get("Min"))
    mx = _heybo_safe_float_nutrient((rng or {}).get("Max"))
    if mn is not None and mx is not None and abs(mn - mx) < 1e-6:
        return mn
    return None


def _heybo_filter_is_equal_min_max_target(nutrient_filter) -> bool:
    """True when this filter started as an equal Min/Max user target (Salad parity)."""
    if not nutrient_filter:
        return False
    if nutrient_filter.get("_equal_min_max_target"):
        return True
    if _heybo_equal_min_max_target(nutrient_filter.get("_original_range")) is not None:
        return True
    return _heybo_equal_min_max_target(nutrient_filter.get("Range")) is not None


def _heybo_equal_min_max_target_value(nutrient_filter):
    """Resolve the single equal Min/Max target; Min is held during relaxation."""
    if not nutrient_filter:
        return None
    stored = _heybo_safe_float_nutrient(nutrient_filter.get("_equal_target_value"))
    if stored is not None:
        return float(stored)
    for source in (
        nutrient_filter.get("_original_range"),
        nutrient_filter.get("Range"),
    ):
        target = _heybo_equal_min_max_target(source)
        if target is not None:
            return float(target)
    return None


def open_heybo_equal_nutrient_point_targets(nutrient_filters: list) -> list:
    """
    When Min and Max are the same value, users almost always mean "at least X"
    (UI often sends a single target as Min=Max). Open Max to Min+slack up front so
    generation does not burn the attempt budget hunting an exact bowl total.

    Mutates each matching filter's Range in place. Preserves ``_original_range`` for
    messaging. Sets ``_equal_point_target_opened`` on changed rows.
    Caps soft Max at customization / ``_equal_max_ceiling`` when present (Salad parity).
    Returns human-readable notices for filter_summary.
    """
    notices: list = []
    if not isinstance(nutrient_filters, list):
        return notices
    try:
        constraints = load_heybo_nutrition_constraints_from_db()
    except Exception:
        constraints = {}
    for nf in nutrient_filters:
        if not isinstance(nf, dict):
            continue
        if nf.get("_equal_point_target_opened"):
            continue
        rng = nf.get("Range")
        if not isinstance(rng, dict):
            continue
        raw_min, raw_max = rng.get("Min"), rng.get("Max")
        if raw_min is None or raw_max is None:
            continue
        try:
            mn = float(raw_min)
            mx = float(raw_max)
        except (TypeError, ValueError):
            continue
        # Treat near-equal floats as a point target (UI / JSON often echo the same number).
        if abs(mn - mx) > 1e-6:
            continue
        key = _resolve_nutrient_filter_key(str(nf.get("Nutrient") or "").strip())
        slack = _heybo_equal_nutrient_point_target_slack(key or str(nf.get("Nutrient") or ""))
        new_max = mn + slack
        ceiling = _heybo_safe_float_nutrient(nf.get("_equal_max_ceiling"))
        if ceiling is None:
            constraint = (constraints or {}).get(key or "", {}) or {}
            cust_max = _heybo_safe_float_nutrient(constraint.get("customization_max"))
            high_max = _heybo_safe_float_nutrient(constraint.get("high_max"))
            if cust_max is not None and (high_max is None or cust_max > high_max):
                ceiling = cust_max
            else:
                ceiling = high_max
        if ceiling is not None and new_max > float(ceiling):
            new_max = float(ceiling)
        if new_max + 1e-9 < mn:
            new_max = mn
        if not isinstance(nf.get("_original_range"), dict):
            nf["_original_range"] = {"Min": mn, "Max": mx}
        rng["Min"] = mn
        rng["Max"] = new_max
        nf["Range"] = rng
        nf["_equal_point_target_opened"] = True
        nf["_equal_point_target_slack"] = slack
        nf["_equal_min_max_target"] = True
        nf["_equal_target_value"] = mn
        if ceiling is not None:
            nf["_equal_max_ceiling"] = float(ceiling)
        label = key or str(nf.get("Nutrient") or "nutrient")
        notices.append(
            f"Equal {label} target {mn:g}–{mx:g} treated as at least {mn:g} "
            f"(soft Max {new_max:g}, +{slack:g})"
        )
    return notices


def _resolve_nutrient_filter_key(nutrient_name: str) -> str:
    """Map a NutrientFilters Nutrient value to an internal column key."""
    n = (nutrient_name or "").strip()
    if not n:
        return ""
    if n in NUTRIENT_COLUMNS:
        return n
    mapped = NUTRIENT_NAME_MAP.get(n)
    if mapped:
        return mapped
    for label, col in NUTRIENT_NAME_MAP.items():
        if label.lower() == n.lower():
            return col
    return n


def _nutrient_relaxation_priority_rank(nutrient_key: str) -> int:
    """Lower rank = higher priority (kept strict longer). Unknown keys relax first.

    Priority order is loaded from heybo.nutrition_constraint_information.priority_order
    (1 = highest priority).  Falls back to the hardcoded
    ``HEYBO_NUTRIENT_RELAXATION_PRIORITY_ORDER`` constant when the DB column is absent.
    """
    key = (nutrient_key or "").strip()
    db_order = load_heybo_nutrient_priority_order_from_db()
    order = db_order if db_order else HEYBO_NUTRIENT_RELAXATION_PRIORITY_ORDER
    try:
        return order.index(key)
    except ValueError:
        return len(order)


def heybo_active_nutrient_filter_keys(nutrient_filters) -> list:
    """Canonical nutrient keys with an active NutrientFilters row."""
    keys = []
    seen = set()
    for nf in nutrient_filters or []:
        if not isinstance(nf, dict):
            continue
        key = _resolve_nutrient_filter_key(nf.get("Nutrient"))
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def heybo_max_nutrient_relaxation_level(relaxation_levels: dict) -> int:
    if not relaxation_levels:
        return 0
    return max(int(v or 0) for v in relaxation_levels.values())


def heybo_advance_priority_nutrient_relaxation(
    active_nutrient_keys,
    current_levels: dict,
) -> tuple:
    """
    Relax one nutrient step: lowest-priority active filter first.
    Returns (new_levels, relaxed_key or None).
    """
    max_level = len(RELAXATION_PERCENTAGES) - 1
    candidates = [
        key
        for key in (active_nutrient_keys or [])
        if int((current_levels or {}).get(key, 0) or 0) < max_level
    ]
    if not candidates:
        return dict(current_levels or {}), None
    candidates.sort(key=_nutrient_relaxation_priority_rank, reverse=True)
    target = candidates[0]
    new_levels = dict(current_levels or {})
    new_levels[target] = int(new_levels.get(target, 0) or 0) + 1
    return new_levels, target


# Plant/pescatarian chain: a looser request includes every stricter tag.
# Vegetarian includes Vegan; Pescatarian includes Vegetarian and Vegan.
_STRUCTURAL_DIET_TAGS = ("Vegan", "Vegetarian", "Pescatarian")


def _diet_exact_token_mask(series, diet_name: str) -> pd.Series:
    """True where diet_parameters lists the diet as an exact token (not substring)."""
    needle = str(diet_name).strip().lower()
    if not needle:
        return pd.Series(False, index=series.index)
    return series.fillna("").apply(
        lambda x: needle
        in [
            t.strip().lower()
            for t in str(x).replace("/", "|").replace(";", "|").replace(",", "|").split("|")
        ]
    )


def heybo_compatible_diet_tags(user_diet: str) -> list:
    """
    Diet tags that satisfy a requested ingredient diet.

    Vegetarian → Vegetarian + Vegan. Pescatarian → those plus Pescatarian.
    Vegan stays Vegan-only. Other hierarchy diets keep a single exact tag.
    """
    name = (user_diet or "").strip()
    if not name:
        return []
    if name not in DIET_HIERARCHY:
        return [name]
    idx = DIET_HIERARCHY.index(name)
    if name in _STRUCTURAL_DIET_TAGS:
        return list(DIET_HIERARCHY[: idx + 1])
    return [name]


def _heybo_row_matches_compatible_diets(series, user_diet: str) -> pd.Series:
    """True when a row's diet_parameters matches any compatible tag for user_diet."""
    tags = heybo_compatible_diet_tags(user_diet)
    if not tags:
        return pd.Series(False, index=series.index)
    mask = pd.Series(False, index=series.index)
    for tag in tags:
        mask |= _diet_exact_token_mask(series, tag)
    if user_diet in ("Vegan", "Vegetarian"):
        mask &= ~_diet_exact_token_mask(series, "Non-Vegetarian")
    elif user_diet == "Pescatarian":
        non_veg = _diet_exact_token_mask(series, "Non-Vegetarian")
        pesc = _diet_exact_token_mask(series, "Pescatarian")
        mask &= ~(non_veg & ~pesc)
    return mask


def filter_dataframe_by_strict_diets(df: pd.DataFrame, diet_filters) -> pd.DataFrame:
    """
    Keep rows that satisfy every requested diet (structural diets include stricter tags).
    Used when topping up bowls after flavor/cuisine/prep shrink the preference pool.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df.iloc[0:0] if isinstance(df, pd.DataFrame) else df
    if not diet_filters:
        return df
    if isinstance(diet_filters, list):
        names = [str(d).strip() for d in diet_filters if str(d).strip()]
    else:
        names = [str(diet_filters).strip()] if str(diet_filters).strip() else []
    if not names or "diet_parameters" not in df.columns:
        return df.iloc[0:0]
    mask = pd.Series(True, index=df.index)
    for diet_name in names:
        mask &= _heybo_row_matches_compatible_diets(df["diet_parameters"], diet_name)
    return df[mask]


def apply_heybo_diet_filters_with_fallback(df, diet_filters):
    """Apply diet filters with fallback logic similar to SaladStop."""
    if not isinstance(df, pd.DataFrame) or df.empty or not isinstance(diet_filters, list) or not diet_filters:
        return df, []
    if isinstance(diet_filters, list):
        diet_filters = [str(diet) if not isinstance(diet, str) else diet for diet in diet_filters]
    else:
        diet_filters = [str(diet_filters)]
    user_diet = diet_filters[0]
    try:
        idx = DIET_HIERARCHY.index(user_diet)
    except ValueError:
        idx = len(DIET_HIERARCHY) - 1
    matched_dfs = []
    fallback_categories = []

    for category in df['category'].unique():
        cat_df = df[df['category'] == category]
        if user_diet in _STRUCTURAL_DIET_TAGS:
            # Union all compatible tags (Vegetarian keeps Vegan sauces/proteins too).
            pool = cat_df[_heybo_row_matches_compatible_diets(cat_df["diet_parameters"], user_diet)]
            if not pool.empty:
                matched_dfs.append(pool)
                tags = ", ".join(heybo_compatible_diet_tags(user_diet))
                print(
                    f"Category: {category}, Diet: {user_diet} (includes {tags}), "
                    f"Matched: {len(pool)} items"
                )
                continue
        else:
            found = False
            for fallback_diet in DIET_HIERARCHY[:idx + 1][::-1]:
                pool = df[
                    (df['category'] == category) &
                    (_diet_exact_token_mask(df['diet_parameters'], fallback_diet))
                ]
                if not pool.empty:
                    matched_dfs.append(pool)
                    found = True
                    print(f"Category: {category}, Diet: {fallback_diet}, Matched: {len(pool)} items")
                    break
            if found:
                continue
        fallback_categories.append(category)
    return pd.concat(matched_dfs) if matched_dfs else df.iloc[0:0], fallback_categories


def meets_heybo_nutrient_filters(total_nutrients, nutrient_filters):
    """Check if bowl meets nutrient filter requirements."""
    if not nutrient_filters:
        return True
    for nutrient_filter in nutrient_filters:
        try:
            input_name = nutrient_filter.get('Nutrient', '').strip()
            range_filter = nutrient_filter.get('Range', {})
            min_val = range_filter.get('Min')
            max_val = range_filter.get('Max')
            stored_name = _resolve_nutrient_filter_key(input_name)
            if not stored_name:
                print(f"Warning: Unrecognized nutrient filter '{input_name}'")
                continue
            actual_value = total_nutrients.get(stored_name, 0.0)
            print(f"Checking {input_name} ({stored_name}): {actual_value}, range: {min_val}-{max_val}")
            if min_val is not None and actual_value < min_val:
                print(f"Failed minimum check: {actual_value} < {min_val}")
                return False
            if max_val is not None and actual_value > max_val:
                print(f"Failed maximum check: {actual_value} > {max_val}")
                return False
        except Exception as e:
            print(f"Error processing nutrient filter {nutrient_filter}: {e}")
            continue
    return True


def get_heybo_relaxed_nutrient_filters(
    nutrient_filters,
    relaxation_level=0,
    relaxation_levels=None,
):
    """
    Return nutrient filters relaxed by level (0..5) per nutrient or uniformly.
    When relaxation_levels is set, each filter uses its own level (priority-order relaxation).
    """
    if not nutrient_filters:
        return []
    max_level = len(RELAXATION_PERCENTAGES) - 1
    relaxed = []
    for nf in nutrient_filters:
        key = _resolve_nutrient_filter_key((nf or {}).get("Nutrient", ""))
        if relaxation_levels is not None and key:
            level = int((relaxation_levels or {}).get(key, 0) or 0)
        else:
            level = int(relaxation_level or 0)
        level = max(0, min(level, max_level))
        pct = RELAXATION_PERCENTAGES[level] / 100.0
        rng = dict((nf or {}).get("Range", {}))
        # Equal Min/Max: hold Min at target; only expand Max as relaxation increases.
        if _heybo_filter_is_equal_min_max_target(nf):
            target = _heybo_equal_min_max_target_value(nf)
            if target is not None:
                rng["Min"] = target
                opened_max = _heybo_safe_float_nutrient((nf.get("Range") or {}).get("Max"))
                base_max = float(target)
                if nf.get("_equal_point_target_opened") and opened_max is not None:
                    base_max = max(base_max, float(opened_max))
                relaxed_max = base_max * (1 + pct) if pct > 0 else base_max
                ceiling = _heybo_safe_float_nutrient((nf or {}).get("_equal_max_ceiling"))
                if ceiling is not None:
                    relaxed_max = min(float(relaxed_max), ceiling)
                rng["Max"] = relaxed_max
                out = dict(nf or {})
                out["Nutrient"] = (nf or {}).get("Nutrient")
                out["Range"] = rng
                relaxed.append(out)
                continue
        min_val = rng.get("Min")
        max_val = rng.get("Max")
        if min_val is not None:
            rng["Min"] = max(0, float(min_val) * (1 - pct))
        if max_val is not None:
            rng["Max"] = float(max_val) * (1 + pct)
        relaxed.append({
            "Nutrient": (nf or {}).get("Nutrient"),
            "Range": rng,
        })
    return relaxed


def meets_heybo_nutrient_filters_with_relaxation(
    total_nutrients,
    nutrient_filters,
    relaxation_level=0,
    relaxation_levels=None,
):
    """Check nutrients against progressively relaxed ranges (per-nutrient or uniform)."""
    relaxed_filters = get_heybo_relaxed_nutrient_filters(
        nutrient_filters,
        relaxation_level=relaxation_level,
        relaxation_levels=relaxation_levels,
    )
    return meets_heybo_nutrient_filters(total_nutrients, relaxed_filters), relaxed_filters


def heybo_meets_balanced_diet(total_nutrients: dict, relaxation_level: int = 0) -> bool:
    """
    True if calories, carbs, protein, and fat fall within DB balanced_min/balanced_max per nutrient_key,
    widened using the same RELAXATION_PERCENTAGES as nutrient relaxation. If no balanced columns exist
    for those keys, returns True.
    """
    try:
        guidelines = load_heybo_nutrition_constraints_from_db()
    except Exception:
        guidelines = {}
    level = max(0, min(int(relaxation_level), len(RELAXATION_PERCENTAGES) - 1))
    relaxation_pct = RELAXATION_PERCENTAGES[level] / 100.0
    nutrients_checked = 0

    for nutrient_key in BALANCED_DIET_NUTRIENTS:
        nutrient_guidelines = guidelines.get(nutrient_key) or {}
        balanced_min = nutrient_guidelines.get("balanced_min")
        balanced_max = nutrient_guidelines.get("balanced_max")
        if balanced_min is None or balanced_max is None:
            continue

        nutrients_checked += 1
        actual_val = total_nutrients.get(nutrient_key, 0) or 0
        try:
            bmin = float(balanced_min)
            bmax = float(balanced_max)
            actual_val = float(actual_val)
        except (TypeError, ValueError):
            return False

        if relaxation_pct > 0:
            width = bmax - bmin
            widen = width * relaxation_pct
            check_min, check_max = bmin - widen, bmax + widen
        else:
            check_min, check_max = bmin, bmax

        if not (check_min <= actual_val <= check_max):
            return False

    return True


def balanced_diet_nutrient_filters(relaxation_level: int = 0) -> list:
    """
    Build NutrientFilters-shaped ranges from DB balanced_min/balanced_max.

    Same widening as ``heybo_meets_balanced_diet`` when relaxation > 0 (Salad parity).
    """
    try:
        guidelines = load_heybo_nutrition_constraints_from_db()
    except Exception:
        guidelines = {}
    level = max(0, min(int(relaxation_level or 0), len(RELAXATION_PERCENTAGES) - 1))
    relaxation_pct = RELAXATION_PERCENTAGES[level] / 100.0
    out: list = []
    for nutrient_key in BALANCED_DIET_NUTRIENTS:
        nutrient_guidelines = guidelines.get(nutrient_key) or {}
        balanced_min = nutrient_guidelines.get("balanced_min")
        balanced_max = nutrient_guidelines.get("balanced_max")
        if balanced_min is None or balanced_max is None:
            continue
        try:
            lo = float(balanced_min)
            hi = float(balanced_max)
        except (TypeError, ValueError):
            continue
        if relaxation_pct > 0:
            widening = (hi - lo) * relaxation_pct
            lo = max(0.0, lo - widening)
            hi = hi + widening
        out.append({
            "Nutrient": nutrient_key,
            "Range": {"Min": lo, "Max": hi},
            "_cpsat_source": "balanced",
        })
    return out
