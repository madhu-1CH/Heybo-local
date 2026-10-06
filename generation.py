"""
Heybo bowl generation: main loop to build and validate Heybo BYB bowls.

  1. PREPARE   — filters, pools, Apriori, category limits
  2. CP-SAT    — ``_heybo_try_cpsat_fill`` → ``cpsat_runner.run_heybo_cpsat_fill``
  3. RANDOM    — fallback while ``len(heybo_bowls) < 5`` (relaxation re-runs CP-SAT)
  4. RESPOND   — names, images, validations, ``message_to_user``
"""
import copy
import random
from collections import Counter
from typing import Optional

import pandas as pd

from .config import CPSAT_ENABLED
from .config_loader import REQUIRED_BOWL_CATEGORIES, get_heybo_config
from .cpsat_runner import CpsatFillState, run_heybo_cpsat_fill
from .diet_constants import (
    HEYBO_BOWL_COMPONENT_KEYS,
    HEYBO_EXTRA_CATEGORIES,
    HEYBO_NUTRIENT_RELAXATION_PRIORITY_ORDER,
    NUTRIENT_COLUMNS,
)
from .nutrition_constraints import load_heybo_nutrient_priority_order_from_db
from .db import (
    load_heybo_data_from_db,
    load_heybo_apriori_rules,
    preprocess_heybo_dataframe,
)
from .filters import (
    preprocess_heybo_filters,
    apply_all_ingredient_filters,
    build_heybo_strict_safety_catalog,
    minimum_fill_pools_by_category,
)
from .diet import (
    BALANCED_DIET_NUTRIENTS,
    balanced_criteria_nutrient_overlap,
    balanced_suppressed_by_nutrient_filters,
    heybo_meets_balanced_diet,
    meets_heybo_nutrient_filters,
    meets_heybo_nutrient_filters_with_relaxation,
    heybo_active_nutrient_filter_keys,
    heybo_advance_priority_nutrient_relaxation,
    heybo_max_nutrient_relaxation_level,
    open_heybo_equal_nutrient_point_targets,
    RELAXATION_PERCENTAGES,
)
from .pricing import (
    apply_pricing_tier_split_to_bowl,
    calculate_heybo_bowl_cost_with_breakdown,
    normalize_heybo_price_filters,
    pop_pricing_metadata,
    _tier_default_price,
)
from .bowl_utils import (
    calculate_heybo_bowl_weight,
    calculate_heybo_total_nutrients,
    calculate_heybo_total_co2e,
    get_heybo_incompatible_pairs,
    is_heybo_bowl_compatible,
)
from .validation import (
    validate_heybo_preparation_method_matches,
    validate_heybo_cuisine_matches,
    validate_heybo_ingredient_selections,
    validate_heybo_diet_compatibility,
    validate_heybo_allergen_safety,
    validate_heybo_nutritional_targets,
    validate_heybo_bowl_composition,
    validate_heybo_price,
    extract_heybo_user_warnings,
)
from .user_message import build_heybo_message_to_user
from .light_hearty import (
    evaluate_light_hearty_bowl,
    format_light_hearty_failure,
    light_hearty_criteria_nutrient_overlap,
    light_hearty_suppressed_by_nutrient_filters,
    prime_light_hearty_cache_from_cfg,
)
from .nutrition_constraints import load_heybo_nutrition_constraints_from_db
from .image_details import build_heybo_dynamic_grid_image_details, build_heybo_image_url_map

# Stages that use the same 40-failure threshold. The loop below only acts on price + prep/cuisine/flavor
# (ingredient pool / price band). Nutrient, light_hearty, balanced_diet, co2 are validated after a bowl exists.
# Match salad RELAXATION_ORDER (relax first → last).
RELAXATION_ORDER = [
    "price",
    "balanced_diet",
    "light_hearty",
    "co2",
    "prep",
    "cuisine",
    "flavor",
    "nutrient",
]

BALANCED_DIET_FAILURES_BEFORE_RELAX = 20

# Constraint-first: spend most of max_attempts (200) building bowls that jointly match
# all active filters. Only then allow price/nutrient relaxation (last 25% of the budget).
# Example: attempts 1–150 = generate feasible bowls; attempts 151–200 = may relax.
JOINT_CONSTRAINT_RELAX_AFTER_ATTEMPT_FRACTION = 0.75

# Always enforce at least this many warm + cold sides combined (within per-category DB limits).
MIN_TOTAL_SIDES_NORMAL = 3

# Bowls returned per request when not in Include/Extra-only mode (Salad / pagination parity).
HEYBO_BOWLS_PER_PAGE = 5


def _heybo_detect_full_category_excludes(
    category_to_ingredients: dict,
    exclude_list: list,
) -> set:
    """Categories where every post-filter eligible ingredient is in Exclude."""
    exc = set(exclude_list or [])
    fully_excluded: set = set()
    for cat in REQUIRED_BOWL_CATEGORIES:
        eligible = set(category_to_ingredients.get(cat) or [])
        if not eligible:
            continue
        if not (eligible - exc):
            fully_excluded.add(cat)
    return fully_excluded


def _heybo_full_exclude_conflict_min(
    cat: str,
    category_limits: dict,
    category_limits_customization: dict,
) -> int:
    """Customization minimum for a fully excluded category (default 1)."""
    normal_min, normal_max = category_limits.get(cat, (0, 1))
    cmin, _ = category_limits_customization.get(cat, (normal_min, normal_max))
    try:
        return int(cmin)
    except (TypeError, ValueError):
        try:
            return int(normal_min)
        except (TypeError, ValueError):
            return 1


def _heybo_full_exclude_conflict_requirement_phrase(cat: str, cmin: int) -> str:
    cat_lower = str(cat).lower()
    if cmin == 1:
        return f"at least one {cat_lower} ingredient"
    return f"at least {cmin} {cat_lower} ingredients"


def _heybo_full_exclude_conflict_user_notice(cat: str, cmin: int) -> str:
    """Single user-facing Note when Exclude cleared a category that still has a DB minimum."""
    cat_lower = str(cat).lower()
    if cmin == 1:
        return (
            f"We added a {cat_lower} ingredient to each bowl because "
            f"a minimum of 1 {cat_lower} is required per bowl "
            f"(you excluded all available {cat_lower} options)"
        )
    return (
        f"We added {cat_lower} ingredients to each bowl because "
        f"a minimum of {cmin} {cat_lower} is required per bowl "
        f"(you excluded all available {cat_lower} options)"
    )


def _heybo_category_exclude_set(
    category: str,
    exclude_list,
    ignore_exclude_categories: set,
) -> set:
    """User Exclude applies per category unless the category is a full-exclude conflict."""
    if category in ignore_exclude_categories:
        return set()
    return set(exclude_list or [])


def _heybo_apply_full_exclude_category_policy(
    fully_excluded: set,
    category_limits: dict,
    category_limits_customization: dict,
) -> tuple:
    """
    Split fully excluded categories:
    - customization min 0 → omit category (0, 0)
    - customization min > 0 → conflict (fill from catalog despite Exclude; warn user)
    """
    omit: set = set()
    conflict: set = set()
    for cat in fully_excluded:
        normal_min, normal_max = category_limits.get(cat, (0, 1))
        cmin, _ = category_limits_customization.get(cat, (normal_min, normal_max))
        try:
            cmin_int = int(cmin) if cmin is not None else int(normal_min)
        except (TypeError, ValueError):
            cmin_int = int(normal_min)
        if cmin_int <= 0:
            omit.add(cat)
        else:
            conflict.add(cat)
    return omit, conflict


def _heybo_sample_pool_cuisine_first(pool, needed, cuisine_names):
    """
    salad.generate_bowl (~5198, ~5276–5306): when cuisine filter is active, prefer
    cuisine-matched picks from this pool, then fill from the rest.
    """
    if needed <= 0 or not pool:
        return []
    pool = list(dict.fromkeys(pool))
    cn = cuisine_names if isinstance(cuisine_names, set) else set(cuisine_names or [])
    first = [x for x in pool if x in cn]
    rest = [x for x in pool if x not in cn]
    out = []
    remain = needed
    if first:
        k = min(remain, len(first))
        out.extend(random.sample(first, k))
        remain -= k
    if remain > 0 and rest:
        k = min(remain, len(rest))
        out.extend(random.sample(rest, k))
    return out


def _heybo_few_true_match_additions(
    user_input: dict,
    category: str,
    max_count: int,
    eligible,
    already,
    *,
    cuisine_active: bool,
    flavor_active: bool,
    prep_active: bool,
) -> list:
    """
    Salad FEW vs MANY: if true matches in this category are ≤ max × 1.5, add them
    to every bowl (capped at remaining slots). MANY → leave sampling to rotate.
    """
    already_list = list(already or [])
    already_set = set(already_list)
    eligible_set = set(eligible or []) | already_set
    try:
        cap = int(max_count)
    except (TypeError, ValueError):
        cap = 1
    room = max(0, cap - len(already_list))
    if room <= 0:
        return []
    mapped = HEYBO_EXTRA_CATEGORIES.get(category, category)
    sources = []
    if cuisine_active:
        sources.append("_heybo_true_cuisine_by_category")
    if flavor_active:
        sources.append("_heybo_true_flavor_by_category")
    if prep_active:
        sources.append("_heybo_true_prep_by_category")
    forced: list = []
    for key in sources:
        by_cat = (user_input or {}).get(key) or {}
        uniq = list(
            dict.fromkeys(
                str(n).strip()
                for n in (by_cat.get(mapped) or by_cat.get(category) or [])
                if n and str(n).strip()
            )
        )
        if not uniq:
            continue
        if len(uniq) > cap * 1.5:
            continue
        for name in uniq:
            if room <= 0:
                break
            if name in already_set or name in forced:
                continue
            if name not in eligible_set:
                continue
            forced.append(name)
            room -= 1
    return forced


def _is_truthy_flag(val) -> bool:
    if val is True:
        return True
    if isinstance(val, str) and val.strip().lower() in ("true", "1", "yes", "on"):
        return True
    if isinstance(val, (int, float)) and val == 1:
        return True
    return False


def _heybo_ingredients_only_mode(ingredients: dict, user_input: dict) -> bool:
    """True when payload requests Only BYB (bowl 1 exact Include∪Extra; bowls 2–5 variety)."""
    ing = ingredients or {}
    if _is_truthy_flag(ing.get("Only")) or _is_truthy_flag(ing.get("only")):
        return True
    return _is_truthy_flag((user_input or {}).get("Only")) or _is_truthy_flag(
        (user_input or {}).get("only")
    )


def _heybo_categories_with_request_overflow(
    requested_count_by_category: dict,
    category_limits: dict,
) -> set:
    """
    Categories where Include+Extra count exceeds the normal max.

    Only those categories use customization max as a **ceiling** so user-asked
    extras can fit. Other categories stay on normal limits. Fill must not pack
    random items up to customization max — only the user-requested extras.
    """
    overflow: set = set()
    for category, req_count in (requested_count_by_category or {}).items():
        normal_max = category_limits.get(category, (0, 0))[1]
        if req_count > normal_max:
            overflow.add(category)
    return overflow


def _heybo_ask_aware_category_limit(
    cat: str,
    normal_min: int,
    normal_max: int,
    high_limits: dict,
    requested_count_by_category: dict,
    *,
    use_high_min: bool = False,
) -> tuple:
    """
    Ceiling for Extra / Include overflow: high enough for user-asked counts, never
    above the high (customization/extras) max, and not a fill-to-max target.

    Example: normal Proteins max=1, user Include+Extra=2, customization max=5
    → effective max=2 (not 5).
    """
    hmin, hmax = high_limits.get(cat, (normal_min, normal_max))
    try:
        hmax_i = int(hmax)
    except (TypeError, ValueError):
        hmax_i = int(normal_max)
    try:
        hmin_i = int(hmin)
    except (TypeError, ValueError):
        hmin_i = int(normal_min)
    req = int((requested_count_by_category or {}).get(cat, 0) or 0)
    eff_max = min(hmax_i, max(int(normal_max), req))
    eff_min = hmin_i if use_high_min else int(normal_min)
    if eff_min > eff_max:
        eff_min = eff_max
    return (eff_min, eff_max)


def _heybo_has_active_numeric_nutrient_filters(nutrient_filters) -> bool:
    """True when any NutrientFilters row has a numeric Min or Max in Range."""
    if not isinstance(nutrient_filters, list):
        return False
    for nf in nutrient_filters:
        if not isinstance(nf, dict):
            continue
        rng = nf.get("Range") or {}
        if not isinstance(rng, dict):
            continue
        if rng.get("Min") is not None or rng.get("Max") is not None:
            return True
    return False


def _heybo_nutrient_filters_need_more(nutrient_filters) -> bool:
    """
    Return True when at least one active nutrient filter has a Min that exceeds
    ``low_max`` for that nutrient in the DB.

    When the user's Min is above the Low band's ceiling they are already asking
    for more than normal Low-diet bowls can reliably deliver, so switching to
    customization category limits (more ingredient slots) after 20 failed
    attempts can actually help reach that minimum.

    Return False when every active filter is Max-only (e.g. Low Calorie, explicit
    Max cap) or has a Min that sits within the Low band (Min ≤ low_max).  In
    those cases adding extra slots only increases nutrient totals and bowl price.

    Fallback: if nutrition_constraint_information is unavailable, falls back to
    the simple "any non-zero Min" heuristic so behaviour degrades gracefully.
    """
    if not isinstance(nutrient_filters, list):
        return False
    try:
        nutrition_constraints = load_heybo_nutrition_constraints_from_db()
    except Exception:
        nutrition_constraints = {}

    for nf in nutrient_filters:
        if not isinstance(nf, dict):
            continue
        rng = nf.get("Range") or {}
        if not isinstance(rng, dict):
            continue
        try:
            min_val = float(rng["Min"]) if rng.get("Min") is not None else 0.0
        except (TypeError, ValueError):
            min_val = 0.0
        if min_val <= 0:
            continue  # Max-only or zero Min — no expansion needed

        from .diet import _resolve_nutrient_filter_key as _rkey

        nutrient_key = _rkey(str(nf.get("Nutrient") or "").strip()) or str(
            nf.get("Nutrient") or ""
        )
        constraint = nutrition_constraints.get(nutrient_key, {})
        if not constraint and nutrient_key:
            # DB rows are keyed by column name; also try raw label.
            constraint = nutrition_constraints.get(str(nf.get("Nutrient") or ""), {})

        low_max = constraint.get("low_max")
        if low_max is None:
            # No DB info for this nutrient — fall back: any non-zero Min triggers expand.
            return True
        try:
            if min_val > float(low_max):
                return True   # user Min exceeds Low band ceiling → expand helps
        except (TypeError, ValueError):
            return True       # can't compare → safe fallback: expand

    return False  # all filters are Max-only or Min within the Low band → keep normal limits


def _heybo_effective_numeric_nutrient_filters(nutrient_filters):
    """
    Prefer explicit user-provided numeric nutrient ranges when present.
    This keeps numeric NutrientFilters as governing constraints even when
    DietFilters were converted to nutrient ranges during preprocessing.
    """
    if not isinstance(nutrient_filters, list) or not nutrient_filters:
        return []
    explicit = []
    for nf in nutrient_filters:
        if not isinstance(nf, dict):
            continue
        if not nf.get("_user_original_request"):
            continue
        rng = nf.get("Range") or {}
        if not isinstance(rng, dict):
            continue
        if rng.get("Min") is None and rng.get("Max") is None:
            continue
        explicit.append(nf)
    return explicit if explicit else list(nutrient_filters)


def _heybo_strip_non_only_filters_for_generation(user_input: dict) -> None:
    """
    Only BYB: use Include / Extra / Exclude only — do not apply diet, allergen, cuisine,
    flavor, price, prep, nutrient, Light/Hearty, Sustainable, or Balanced during generation.
    """
    user_input["DietFilters"] = []
    user_input["NutrientFilters"] = []
    user_input["CuisineFilters"] = []
    user_input["FlavorPreferences"] = {}
    user_input["Price"] = {}
    user_input["PreparationMethod"] = {}
    user_input["AllergenFilters"] = []
    for flag in ("Light", "Hearty", "Sustainable", "Balanced", "Signatures"):
        user_input[flag] = False
    for key in (
        "_heybo_message_diet_filters_dict",
        "_nutrient_preprocess_messages",
        "_heybo_guideline_adjustments_from_preprocess",
        "_relax_preparation_method_filter",
        "_relax_cuisine_filter",
        "_relax_flavor_filter",
        "_cuisine_matched_ingredient_names",
    ):
        user_input.pop(key, None)


def _heybo_apply_balanced_override(user_input: dict, global_validations: dict) -> bool:
    """
    Decide whether Balanced should run. Returns True if Balanced remains active.

    Salad parity: ignore Balanced only when NutrientFilters overlap Balanced
    criteria nutrients (cal/carb/protein/fat). Other filters keep Balanced.
    """
    if not _is_truthy_flag(user_input.get("Balanced")):
        return False

    _nf = user_input.get("NutrientFilters") or []
    _overlap = balanced_criteria_nutrient_overlap(_nf)
    if _overlap:
        user_input["_balanced_suppressed_by_nutrients"] = True
        user_input["_balanced_suppressed_nutrient_keys"] = list(_overlap)
        global_validations["balanced_override"] = True
        _msg = (
            "Balanced ignored — NutrientFilters already set "
            + ", ".join(_overlap)
            + " (same nutrients as Balanced criteria)"
        )
        global_validations["balanced_override_message"] = _msg
        if not any(
            isinstance(x, str) and "Balanced ignored" in x
            for x in (global_validations.get("filter_summary") or [])
        ):
            global_validations.setdefault("filter_summary", []).append(_msg)
        if not any(
            isinstance(x, str) and "Balanced ignored" in x
            for x in (global_validations.get("balanced_diet_matches") or [])
        ):
            global_validations.setdefault("balanced_diet_matches", []).append(_msg)
        return False

    return True


# Price filter: if Min and Max are the same dollar amount, treat as a target and allow ±slack (discrete bowl costs).
# After 40 misses, relaxation_idx steps up and slack grows. Same slacks widen a real range (min−slack, max+slack).
PRICE_RELAXATION_SLACKS = (0.5, 1.0, 2.0, 5.0)


def _heybo_user_requested_price_bounds(user_input):
    """Min/Max the user asked for (before any MaxBowlPrice costlier-bowls cap)."""
    orig = user_input.get("_price_original_before_snap")
    if isinstance(orig, dict) and (orig.get("Min") is not None or orig.get("Max") is not None):
        return orig.get("Min"), orig.get("Max")
    price_filter = user_input.get("Price") or {}
    return price_filter.get("Min"), price_filter.get("Max")


def _heybo_flag_most_bowls_outside_price_range(heybo_bowls, user_input, price_relaxation_level):
    """Set ``_most_bowls_outside_price_range`` when most bowls miss the user's requested Price bounds."""
    user_input["_most_bowls_outside_price_range"] = False
    if not heybo_bowls:
        return
    gv = user_input.get("global_validations") or {}
    if gv.get("price_minimum_violation"):
        return
    min_price_val, max_price_val = _heybo_user_requested_price_bounds(user_input)
    if min_price_val is None and max_price_val is None:
        return
    bowls_outside = 0
    for bowl in heybo_bowls:
        raw = bowl.get("Total Cost")
        if raw is None:
            bowls_outside += 1
            continue
        try:
            cost_value = float(str(raw).replace("$", "").replace(",", "").strip())
        except (ValueError, TypeError):
            bowls_outside += 1
            continue
        if min_price_val is not None and max_price_val is not None:
            if cost_value < float(min_price_val) or cost_value > float(max_price_val):
                bowls_outside += 1
        elif min_price_val is not None:
            if cost_value < float(min_price_val):
                bowls_outside += 1
        elif max_price_val is not None:
            if cost_value > float(max_price_val):
                bowls_outside += 1
    if bowls_outside > len(heybo_bowls) / 2:
        user_input["_most_bowls_outside_price_range"] = True


def _effective_price_bounds(min_p, max_p, relaxation_idx):
    """
    Return (eff_min, eff_max) for generation filtering.
    Heybo has no .50/.90 price rounding — only ±slack from ``PRICE_RELAXATION_SLACKS``.
    Equal Min/Max is treated as a target: always ±slack at the current relaxation step.
    """
    if min_p is None and max_p is None:
        return None, None
    a = float(min_p) if min_p is not None else None
    b = float(max_p) if max_p is not None else None
    idx = min(relaxation_idx, len(PRICE_RELAXATION_SLACKS) - 1)
    slack = PRICE_RELAXATION_SLACKS[idx]

    if a is not None and b is not None and round(a, 2) == round(b, 2):
        t = a
        return t - slack, t + slack

    lo, hi = a, b
    if lo is not None:
        lo -= slack
    if hi is not None:
        hi += slack
    return lo, hi


def _has_sustainable_co2_data(df: pd.DataFrame) -> bool:
    """True when ingredient data has meaningful CO2 values for Sustainable filtering."""
    if "co2e_values_per_serving" not in df.columns:
        return False
    vals = pd.to_numeric(df["co2e_values_per_serving"], errors="coerce")
    if not bool(vals.notna().any()):
        return False
    # During rollout, DB NULLs are normalized to 0. Treat all-zero CO2 as "no data yet".
    non_null = vals.dropna()
    return bool((non_null > 0).any())


def _sustainable_co2_bounds(heybo_cfg: dict):
    cfg = (heybo_cfg or {}).get("co2_config") or {}
    rng = cfg.get("Sustainable") or {}
    lo = pd.to_numeric(rng.get("min"), errors="coerce")
    hi = pd.to_numeric(rng.get("max"), errors="coerce")
    if pd.notna(lo) and pd.notna(hi):
        return float(lo), float(hi)
    return None, None


# Salad parity: Sustainable CO2e max widens  DB-max → 1.80 → unlimited.
CO2_RELAXATION_LEVEL_1_MAX = 1.80
CO2_MAX_RELAXATION_LEVEL = 2


def _effective_sustainable_co2_bounds(heybo_cfg: dict, co2_relaxation_level: int = 0):
    """Sustainable min/max honoring ``co2_relaxation_level`` (Salad random-loop parity)."""
    lo, hi = _sustainable_co2_bounds(heybo_cfg)
    if lo is None or hi is None:
        return None, None
    level = int(co2_relaxation_level or 0)
    if level <= 0:
        return lo, hi
    if level == 1:
        return lo, CO2_RELAXATION_LEVEL_1_MAX
    return lo, float("inf")


def _heybo_combined_count_for_category(bowl: dict, cat: str) -> int:
    """Total line items for limits: main + Extra * bucket (Proteins, Warm sides, Cold sides)."""
    if cat == "Proteins":
        return len(bowl.get("Proteins") or []) + len(bowl.get("Extra Proteins") or [])
    if cat == "Warm sides":
        return len(bowl.get("Warm sides") or []) + len(bowl.get("Extra Warm sides") or [])
    if cat == "Cold sides":
        return len(bowl.get("Cold sides") or []) + len(bowl.get("Extra Cold sides") or [])
    return 0


def _heybo_combined_warm_cold_side_count(bowl: dict) -> int:
    """Warm + cold side line items across main and Extra * lists (pricing may move items between them)."""
    return _heybo_combined_count_for_category(bowl, "Warm sides") + _heybo_combined_count_for_category(
        bowl, "Cold sides"
    )


def _heybo_merge_preference_and_minimum_fill_pool(
    preference_names: list,
    minimum_fill_names: list,
    *,
    expand_when_unique_below: int,
    exclude_set: set,
) -> tuple:
    """
    Prefer preference-filtered names; add strict-safety catalog names only when the
    preference pool is smaller than expand_when_unique_below.
    """
    pref = list(dict.fromkeys([t for t in (preference_names or []) if t and t not in exclude_set]))
    if expand_when_unique_below <= 0 or len(pref) >= expand_when_unique_below:
        return pref, False
    merged = list(
        dict.fromkeys(
            pref + [t for t in (minimum_fill_names or []) if t and t not in exclude_set and t not in pref]
        )
    )
    return merged, len(merged) > len(pref)


def _heybo_minimum_fill_names_for_category(category: str, minimum_fill_pools: dict) -> list:
    if category == "Proteins":
        return list(
            dict.fromkeys(
                (minimum_fill_pools.get("Proteins") or [])
                + (minimum_fill_pools.get("Extra Proteins") or [])
            )
        )
    if category == "Warm sides":
        return list(
            dict.fromkeys(
                (minimum_fill_pools.get("Warm sides") or [])
                + (minimum_fill_pools.get("Extra Warm sides") or [])
            )
        )
    if category == "Cold sides":
        return list(
            dict.fromkeys(
                (minimum_fill_pools.get("Cold sides") or [])
                + (minimum_fill_pools.get("Extra Cold sides") or [])
            )
        )
    return list(minimum_fill_pools.get(category) or [])


def _heybo_side_fill_pools(
    warm_sides,
    cold_sides,
    extra_warm_sides,
    extra_cold_sides,
    minimum_fill_pools: dict,
    exclude_list: list,
    required_sides: int,
) -> tuple:
    """Warm/cold pools for structural side minimum; may widen via strict-safety catalog."""
    exc = set(exclude_list or [])
    warm_p = list(dict.fromkeys([t for t in (warm_sides or []) + (extra_warm_sides or []) if t not in exc]))
    cold_p = list(dict.fromkeys([t for t in (cold_sides or []) + (extra_cold_sides or []) if t not in exc]))
    combined = len(set(warm_p + cold_p))
    if required_sides <= 0 or combined >= required_sides:
        return warm_p, cold_p, False
    warm_mf = _heybo_minimum_fill_names_for_category("Warm sides", minimum_fill_pools)
    cold_mf = _heybo_minimum_fill_names_for_category("Cold sides", minimum_fill_pools)
    warm_merged = list(dict.fromkeys(warm_p + [t for t in warm_mf if t not in exc]))
    cold_merged = list(dict.fromkeys(cold_p + [t for t in cold_mf if t not in exc]))
    return warm_merged, cold_merged, len(set(warm_merged + cold_merged)) > combined


def _heybo_bowl_ingredient_names(bowl: dict) -> set:
    names = set()
    for key in HEYBO_BOWL_COMPONENT_KEYS:
        names.update(bowl.get(key) or [])
    return names


def _heybo_ingredient_set_compatible(
    ingredient_names: set,
    incompatible_pairs: dict,
    user_included: set,
) -> bool:
    """Same rules as is_heybo_bowl_compatible, for a flat ingredient-name set."""
    user_included = user_included or set()
    for ingredient in ingredient_names:
        if ingredient not in incompatible_pairs:
            continue
        conflicts = ingredient_names & incompatible_pairs[ingredient]
        conflicts = {c for c in conflicts if not (ingredient in user_included and c in user_included)}
        if conflicts:
            return False
    return True


def _heybo_candidate_compatible(
    candidate: str,
    bowl_ingredient_names: set,
    incompatible_pairs: dict,
    user_included: set,
) -> bool:
    return _heybo_ingredient_set_compatible(
        bowl_ingredient_names | {candidate}, incompatible_pairs, user_included
    )


def _heybo_pick_compatible_from_pool(
    pool: list,
    bowl: dict,
    incompatible_pairs: dict,
    user_included: set,
    exclude_set: set,
    in_bowl: set,
) -> Optional[str]:
    bowl_names = _heybo_bowl_ingredient_names(bowl)
    eligible = [
        t
        for t in pool
        if t not in exclude_set and t not in in_bowl
        and _heybo_candidate_compatible(t, bowl_names, incompatible_pairs, user_included)
    ]
    if not eligible:
        return None
    return random.choice(eligible)


# ---------------------------------------------------------------------------
# Budget-aware picking helpers
# ---------------------------------------------------------------------------

def _heybo_ingredient_nutrient_val(
    ingredient: str,
    nutrient_key: str,
    df: "pd.DataFrame",
    cache: dict,
) -> float:
    """Cached per-ingredient nutrient lookup for budget tracking."""
    if ingredient not in cache:
        cache[ingredient] = {}
    if nutrient_key not in cache[ingredient]:
        try:
            col = df.loc[df["ingredient_name"] == ingredient, nutrient_key]
            cache[ingredient][nutrient_key] = float(col.iloc[0]) if not col.empty else 0.0
        except Exception:
            cache[ingredient][nutrient_key] = 0.0
    return cache[ingredient][nutrient_key]


def _heybo_sort_pool_by_nutrient(
    pool: list,
    numeric_nutrient_filters: list,
    df: "pd.DataFrame",
    cache: dict,
) -> list:
    """
    Sort pool so nutrient-appropriate ingredients appear first, biasing random
    picks toward values that satisfy the active numeric nutrient filters.

    Two cases (lower score = better fit, sorted ascending):

      1. Any filter with a Max bound (with or without a Min):
             score += +nutrient_value       → low values first

      2. Min-only (binding Min > 0, no Max at all):
             score += -nutrient_value       → high values first

    When BOTH Min > 0 and Max are present (range or equal-target constraint),
    Max takes priority and we sort low-first (case 1).  A midpoint/abs-
    deviation approach is NOT used because the target is a bowl-level SUM,
    not a per-ingredient value.  For a bowl-total target of e.g. 9 g fat,
    each individual ingredient should contribute ~1 g (bowl_total / N), so
    low-fat ingredients are always the right choice to stay within the Max
    while accumulating toward the Min.  Sorting by abs(fat − 9) would instead
    favour 9 g-per-ingredient items, pushing bowl totals to 72 g+.

    The old dual-directive (+1 and -1 for the same key) approach cancelled to
    score=0 for every ingredient, neutralising the sort entirely.  This fix
    avoids that by ignoring the Min directive whenever Max is present.

    A Min of 0 is treated as non-binding (trivially satisfied) and falls
    into case 1 as well.

    For multiple independent nutrients the per-nutrient scores are summed.
    Returns the original pool unchanged on any error or if no signal applies.
    """
    if not pool or not numeric_nutrient_filters:
        return pool
    from .diet import _resolve_nutrient_filter_key as _rkey
    try:
        nutrition_constraints = load_heybo_nutrition_constraints_from_db()
    except Exception:
        nutrition_constraints = {}
    directives: list = []
    for nf in numeric_nutrient_filters:
        raw = (nf.get("Nutrient") or "").strip()
        key = _rkey(raw) if raw else raw
        if not key or key not in df.columns:
            continue
        rng = nf.get("Range") or {}
        max_raw = rng.get("Max")
        min_raw = rng.get("Min")
        has_max = max_raw is not None
        try:
            min_val = float(min_raw) if min_raw is not None else 0.0
            is_binding_min = min_val > 0
        except (TypeError, ValueError):
            min_val = 0.0
            is_binding_min = False

        if is_binding_min:
            # Only sort HIGH-first when Min exceeds the low band for this nutrient
            # (e.g. High Protein Min: 45g where low_max ≈ 20g).
            # When Min is just the system floor within the low band (e.g. Low Calorie
            # Min: 350 kcal), the Max is the primary constraint → sort LOW-first so
            # the bowl stays under the calorie ceiling.
            low_max = None
            try:
                low_max = nutrition_constraints.get(key, {}).get("low_max")
            except Exception:
                pass
            if low_max is not None:
                min_exceeds_low_band = min_val > float(low_max)
            else:
                # No band info: if Max is also present assume Max is the primary
                # constraint and prefer low-first; otherwise go high-first.
                min_exceeds_low_band = not has_max
            if min_exceeds_low_band:
                # Case 1a: ambitious Min (above the low band) → high values first.
                directives.append((key, -1))
            elif has_max:
                # Case 1b: Min is within the low band AND a Max cap exists → low-first.
                directives.append((key, +1))
        elif has_max:
            # Case 2: Max-only (no binding Min, or Min == 0) → low values first.
            # The user needs LESS of this nutrient (e.g. Low Fat Max: 8g,
            # Low Calorie Max: 340 kcal). Sorting low values first keeps the
            # bowl under the ceiling.
            directives.append((key, +1))

    if not directives:
        return pool

    def _score(ing: str) -> float:
        total = 0.0
        for key, direction in directives:
            total += direction * _heybo_ingredient_nutrient_val(ing, key, df, cache)
        return total

    try:
        return sorted(pool, key=_score)
    except Exception:
        return pool


def _heybo_has_price_constraint(user_input) -> bool:
    """True when the user set a Price Min and/or Max."""
    price = (user_input or {}).get("Price") or {}
    return isinstance(price, dict) and (
        price.get("Min") is not None or price.get("Max") is not None
    )


def _heybo_hard_user_max_nutrient_keys(nutrient_filters: Optional[list] = None) -> set:
    """
    Nutrient keys with a real user Max (not soft Max from equal Min==Max open).
    Used to detect plain Min+Max bands when price is absent.
    """
    from .diet import _resolve_nutrient_filter_key as _rkey

    keys: set = set()
    for nf in nutrient_filters or []:
        if not isinstance(nf, dict):
            continue
        if nf.get("_equal_point_target_opened"):
            continue
        rng = nf.get("Range") or {}
        if not isinstance(rng, dict) or rng.get("Max") is None:
            continue
        key = _rkey(str(nf.get("Nutrient") or "").strip())
        if key:
            keys.add(key)
    return keys


def _heybo_hard_constraint_mode(
    user_input,
    numeric_nutrient_mode: bool,
    *,
    nutrient_filters: Optional[list] = None,
) -> bool:
    """
    True when generation should prioritize joint feasibility-first picking.

    ON when:
      • any Price Min/Max is set (price headroom + RELAXATION_ORDER widen), or
      • binding Mins that need constructive Min-chase (ambitious / Min-only).

    Price never changes the nutrient fill strategy by itself — see
    ``_heybo_should_chase_binding_mins``. Max-first bands (e.g. calories
    250–300) stay on normal fill whether or not price is present.
    """
    if _heybo_has_price_constraint(user_input):
        return True
    if not numeric_nutrient_mode:
        return False
    mins = _heybo_binding_min_nutrient_targets(nutrient_filters or [])
    return _heybo_should_chase_binding_mins(
        user_input, mins, nutrient_filters or []
    )


def _heybo_should_chase_binding_mins(
    user_input,
    binding_min_targets: dict,
    nutrient_filters: Optional[list] = None,
) -> bool:
    """
    Min-chase / joint-fill when Mins need constructive high packing.

    Global rule — price does NOT flip the nutrient path:
      • Chase when at least one Min is ambitious (Min > DB ``low_max``), e.g.
        protein 60–75 / protein ≥60 — same with or without price.
      • Chase for Min-only filters (no hard Max; soft equal-point Max ignored).
      • Do NOT chase Max-first / low-band Mins (e.g. calories 250–300). Use
        the same normal low-first fill as without price; if the bowl cannot
        meet Price, RELAXATION_ORDER widens price (before nutrient).

    Soft equal-point Max is not a hard Max. Price on ``user_input`` is ignored
    here on purpose (call-site keeps the same signature).
    """
    if not binding_min_targets:
        return False
    filters = list(nutrient_filters or [])
    # Ambitious Mins (above Low-band ceiling) need Extra slots / joint-fill.
    if _heybo_nutrient_filters_need_more(filters):
        return True
    # Min-only (no hard Max on that key), including soft equal-point Max.
    hard_maxes = _heybo_hard_user_max_nutrient_keys(filters)
    return bool(set(binding_min_targets) - hard_maxes)

def _heybo_price_ingredient_headroom(user_input, price_config, price_relaxation_level: int = 0) -> Optional[float]:
    """
    Remaining spend allowed for ingredient ai_prices after BYB_Min_Price (platform base).
    None = no price Max constraint (unlimited headroom for constructive filtering).
    """
    if not isinstance(price_config, dict):
        return None
    base = float(price_config.get("base_price") or 0.0)
    price = (user_input or {}).get("Price") or {}
    if not isinstance(price, dict) or price.get("Max") is None:
        return None
    try:
        _, eff_max = _effective_price_bounds(price.get("Min"), price.get("Max"), price_relaxation_level)
        if eff_max is None:
            return None
        return max(0.0, float(eff_max) - base)
    except (TypeError, ValueError):
        return None


def _heybo_pool_affordable(
    pool: list,
    df: "pd.DataFrame",
    category: str,
    remaining_headroom: Optional[float],
) -> list:
    """Keep ingredients whose default-category ai_price fits remaining price headroom."""
    if remaining_headroom is None or not pool:
        return list(pool)
    out = []
    for ing in pool:
        try:
            p = float(_tier_default_price(ing, df, category) or 0.0)
        except Exception:
            p = 0.0
        if p <= remaining_headroom + 1e-9:
            out.append(ing)
    return out if out else list(pool)


def _heybo_constructive_sample(
    pool: list,
    n: int,
    *,
    df: "pd.DataFrame",
    category: str,
    numeric_nutrient_filters: list,
    cache: dict,
    remaining_headroom: Optional[float],
    prefer_names: Optional[list] = None,
    exclude_names: Optional[set] = None,
) -> list:
    """
    Constraint-first pick: sort by nutrient fitness, keep only affordable items, then take the
    best-fit names (light shuffle among the top tier for mild variety). Prefer ``prefer_names``
    when provided (e.g. unused feasible proteins).
    """
    if n <= 0 or not pool:
        return []
    exclude_names = exclude_names or set()
    working = [x for x in pool if x not in exclude_names]
    if not working:
        working = list(pool)
    working = _heybo_pool_affordable(working, df, category, remaining_headroom)
    if prefer_names:
        pref = [x for x in prefer_names if x in working]
        if pref:
            working = pref + [x for x in working if x not in pref]
    ranked = _heybo_sort_pool_by_nutrient(
        working, numeric_nutrient_filters or [], df, cache
    )
    if not ranked:
        ranked = working
    take = min(n, len(ranked))
    if take <= 0:
        return []
    # Among the best ``take`` items, allow a small shuffle so bowls are not identical clones
    # when several ingredients share similar nutrient scores — still never drop below top tier.
    top_tier = ranked[: max(take, min(len(ranked), take + 2))]
    if len(top_tier) <= take:
        return list(top_tier)
    # Always keep the absolute best first when chasing Mins; fill remaining from next-best.
    best = top_tier[0]
    rest = top_tier[1:]
    random.shuffle(rest)
    return [best] + rest[: take - 1]


def _heybo_binding_min_nutrient_targets(numeric_nutrient_filters: list) -> dict:
    """Map resolved nutrient column -> Min value for binding Mins (> 0)."""
    from .diet import _resolve_nutrient_filter_key as _rkey

    out = {}
    for nf in numeric_nutrient_filters or []:
        if not isinstance(nf, dict):
            continue
        rng = nf.get("Range") or {}
        if not isinstance(rng, dict) or rng.get("Min") is None:
            continue
        try:
            mn = float(rng["Min"])
        except (TypeError, ValueError):
            continue
        if mn <= 0:
            continue
        key = _rkey(str(nf.get("Nutrient") or "").strip())
        if key:
            out[key] = mn
    return out


def _heybo_binding_max_nutrient_targets(numeric_nutrient_filters: list) -> dict:
    """Map resolved nutrient column -> Max value when a Max is set."""
    from .diet import _resolve_nutrient_filter_key as _rkey

    out = {}
    for nf in numeric_nutrient_filters or []:
        if not isinstance(nf, dict):
            continue
        rng = nf.get("Range") or {}
        if not isinstance(rng, dict) or rng.get("Max") is None:
            continue
        try:
            mx = float(rng["Max"])
        except (TypeError, ValueError):
            continue
        key = _rkey(str(nf.get("Nutrient") or "").strip())
        if key:
            out[key] = mx
    return out


def _heybo_estimate_feasible_proteins(
    proteins: list,
    *,
    df: "pd.DataFrame",
    cache: dict,
    numeric_nutrient_filters: list,
    remaining_headroom: Optional[float],
    warm_pool: list,
    cold_pool: list,
    warm_max: int,
    cold_max: int,
    base_pool: list,
    dip_pool: list,
    garnish_pool: list,
    sauce_pool: list,
    protein_max: int = 1,
) -> list:
    """
    Proteins that can reach binding nutrient Mins without exceeding Max caps, under
    the same slot/headroom caps used by joint-fill. Empty ⇒ Mins unreachable under
    current slots (do not fall back to the full list).
    """
    targets = _heybo_binding_min_nutrient_targets(numeric_nutrient_filters)
    max_targets = _heybo_binding_max_nutrient_targets(numeric_nutrient_filters)
    if not targets or not proteins:
        return list(proteins)
    try:
        p_slots = max(1, int(protein_max or 1))
    except (TypeError, ValueError):
        p_slots = 1
    tracked = set(targets) | set(max_targets)

    def _val(ing, k):
        return _heybo_ingredient_nutrient_val(ing, k, df, cache)

    def _pick_fitting(pool, category, k, headroom, used, totals):
        """Pick up to k items that help Mins without blowing Max; stop when Mins met."""
        if k <= 0 or not pool:
            return [], headroom, totals
        available = [x for x in pool if x not in used]
        affordable = _heybo_pool_affordable(available, df, category, headroom)
        if not affordable:
            return [], headroom, totals
        need_min = any(totals.get(key, 0.0) + 1e-9 < mn for key, mn in targets.items())
        if need_min:
            ranked = sorted(
                affordable,
                key=lambda ing: -sum(
                    _val(ing, key)
                    for key, mn in targets.items()
                    if totals.get(key, 0.0) + 1e-9 < mn
                ),
            )
        else:
            ranked = sorted(
                affordable,
                key=lambda ing: sum(_val(ing, key) for key in max_targets) if max_targets else 0.0,
            )
        picked = []
        for ing in ranked:
            if len(picked) >= k:
                break
            if max_targets and any(
                totals.get(key, 0.0) + _val(ing, key) > float(mx) + 1e-9
                for key, mx in max_targets.items()
            ):
                continue
            if not need_min and max_targets:
                break
            picked.append(ing)
            used.add(ing)
            for key in tracked:
                totals[key] = totals.get(key, 0.0) + _val(ing, key)
            try:
                spent = float(_tier_default_price(ing, df, category) or 0.0)
            except Exception:
                spent = 0.0
            if headroom is not None:
                headroom = max(0.0, headroom - spent)
            need_min = any(totals.get(key, 0.0) + 1e-9 < mn for key, mn in targets.items())
            if not need_min and category in ("Proteins", "Warm sides", "Cold sides", "Bases"):
                break
        return picked, headroom, totals

    feasible = []
    for protein in proteins:
        head = remaining_headroom
        used: set = set()
        try:
            p_cost = float(_tier_default_price(protein, df, "Proteins") or 0.0)
        except Exception:
            p_cost = 0.0
        if head is not None and p_cost > head + 1e-9:
            continue
        if head is not None:
            head = max(0.0, head - p_cost)
        used.add(protein)
        totals = {k: _val(protein, k) for k in tracked}
        if max_targets and any(
            totals.get(k, 0.0) > float(mx) + 1e-9 for k, mx in max_targets.items()
        ):
            continue
        extra_needed = max(0, p_slots - 1)
        if extra_needed > 0:
            _, head, totals = _pick_fitting(
                [p for p in proteins if p != protein],
                "Proteins",
                extra_needed,
                head,
                used,
                totals,
            )
        for pool, cat, kmax in (
            (base_pool, "Bases", 1),
            (warm_pool, "Warm sides", max(0, int(warm_max or 0))),
            (cold_pool, "Cold sides", max(0, int(cold_max or 0))),
            (dip_pool, "Dips", 1),
            (garnish_pool, "Garnish", 1),
            (sauce_pool, "Sauces", 1),
        ):
            _, head, totals = _pick_fitting(pool, cat, kmax, head, used, totals)
        if all(totals.get(k, 0.0) + 1e-9 >= mn for k, mn in targets.items()) and (
            not max_targets
            or all(totals.get(k, 0.0) <= float(mx) + 1e-9 for k, mx in max_targets.items())
        ):
            feasible.append(protein)
    if not feasible:
        return []
    return _heybo_sort_pool_by_nutrient(feasible, numeric_nutrient_filters, df, cache)


def _heybo_cheapest_extra_protein_cost(
    proteins: list,
    df: "pd.DataFrame",
) -> float:
    """Lowest Proteins-row ai_price among candidates (proxy for Extra Proteins surcharge)."""
    best = None
    for ing in proteins or []:
        try:
            c = float(_tier_default_price(ing, df, "Proteins") or 0.0)
        except Exception:
            continue
        if c <= 0:
            continue
        if best is None or c < best:
            best = c
    return float(best) if best is not None else 2.0


def _heybo_price_level_for_needed_headroom(needed: float) -> int:
    """Smallest PRICE_RELAXATION_SLACKS index whose slack covers ``needed`` dollars."""
    need = max(0.0, float(needed or 0.0))
    for i, slack in enumerate(PRICE_RELAXATION_SLACKS):
        if float(slack) + 1e-9 >= need:
            return i
    return len(PRICE_RELAXATION_SLACKS) - 1


def _heybo_fill_bowl_joint_constraints(
    bowl: dict,
    *,
    df: "pd.DataFrame",
    cache: dict,
    pools_by_category: dict,
    limits_by_category: dict,
    numeric_nutrient_filters: list,
    remaining_headroom: Optional[float],
    primary_protein: Optional[str] = None,
    prefer_sauce: Optional[str] = None,
    exclude_set: Optional[set] = None,
    used_combos: Optional[set] = None,
    sauce_pool: Optional[list] = None,
    must_include_by_category: Optional[dict] = None,
    incompatible_pairs: Optional[dict] = None,
    user_included: Optional[set] = None,
) -> Optional[float]:
    """
    Fill ``bowl`` to hit binding nutrient Mins without exceeding Max caps (e.g. protein
    70–80). Always respects Max (including category-min slots); stops packing once Mins
    are met; prefers Extra protein over high-protein sides to close Min gaps.
    ``must_include_by_category`` forces user Include/Extra picks into the bowl first
    (same as the normal fill path) — joint-fill must not drop requested ingredients.
    Also skips incompatible pairs (same as normal pick path).
    Returns remaining price headroom, or None if unrestricted.
    """
    exclude_set = exclude_set or set()
    must_include_by_category = must_include_by_category or {}
    incompatible_pairs = incompatible_pairs or {}
    user_included = user_included or set()
    head = remaining_headroom
    used_names: set = set()
    min_targets = _heybo_binding_min_nutrient_targets(numeric_nutrient_filters or [])
    max_targets = _heybo_binding_max_nutrient_targets(numeric_nutrient_filters or [])
    tracked = set(min_targets) | set(max_targets)
    running = {k: 0.0 for k in tracked}

    def _mins_met() -> bool:
        if not min_targets:
            return True
        return all(running.get(k, 0.0) + 1e-9 >= mn for k, mn in min_targets.items())

    def _would_exceed_max(ing: str, reserve: float = 0.0) -> bool:
        if not max_targets:
            return False
        for k, mx in max_targets.items():
            add = _heybo_ingredient_nutrient_val(ing, k, df, cache)
            # Keep a small buffer for Dip/Garnish/Sauce when packing proteins/sides.
            cap = float(mx) - max(0.0, float(reserve))
            if running.get(k, 0.0) + add > cap + 1e-9:
                return True
        return False

    def _contrib(ing: str) -> float:
        keys = max_targets or min_targets
        return sum(_heybo_ingredient_nutrient_val(ing, k, df, cache) for k in keys)

    def _gap_close(ing: str) -> float:
        return sum(
            _heybo_ingredient_nutrient_val(ing, k, df, cache)
            for k, mn in min_targets.items()
            if running.get(k, 0.0) + 1e-9 < mn
        )

    def _finisher_reserve(category: str) -> float:
        """Leave Max room for low Dip/Garnish/Sauce after proteins/sides."""
        if not max_targets or category in ("Dips", "Garnish", "Sauces"):
            return 0.0
        # ~3–4g protein buffer covers typical low finishers (not Avocado Edamame).
        return 4.0 if "protein_g" in max_targets else 0.0

    def _commit(ing: str, category: str) -> None:
        nonlocal head
        used_names.add(ing)
        for k in tracked:
            running[k] = running.get(k, 0.0) + _heybo_ingredient_nutrient_val(
                ing, k, df, cache
            )
        try:
            cost = float(_tier_default_price(ing, df, category) or 0.0)
        except Exception:
            cost = 0.0
        if head is not None:
            head = max(0.0, head - cost)

    def _compatible(ing: str) -> bool:
        if not incompatible_pairs:
            return True
        bowl_names = set(used_names) | set(user_included)
        return _heybo_candidate_compatible(
            ing, bowl_names, incompatible_pairs, user_included
        )

    def _rank_pool(pool: list, category: str) -> list:
        affordable = _heybo_pool_affordable(pool, df, category, head)
        if not affordable:
            affordable = list(pool)
        reserve = _finisher_reserve(category)
        # Drop Max-blowers first so category-min slots cannot force Basil Tofu /
        # Avocado Edamame / Spiced Peanuts past protein Max.
        if max_targets:
            safe = [x for x in affordable if not _would_exceed_max(x, reserve)]
            if safe:
                affordable = safe
        if min_targets and not _mins_met():
            # Close Min with Max-safe items; Proteins preferred over sides via
            # category order. Prefer largest gap-close that still fits Max.
            return sorted(affordable, key=lambda ing: (-_gap_close(ing), _contrib(ing)))
        if max_targets:
            # Mins met (or Min-only absent): lowest contribution to stay under Max.
            return sorted(affordable, key=lambda ing: (_contrib(ing), ing))
        return _heybo_sort_pool_by_nutrient(
            affordable, numeric_nutrient_filters or [], df, cache
        )

    for category in REQUIRED_BOWL_CATEGORIES:
        lim = limits_by_category.get(category) or (0, 0)
        try:
            cat_min, cat_max = int(lim[0]), int(lim[1])
        except (TypeError, ValueError, IndexError):
            cat_min, cat_max = 0, 0
        if cat_max <= 0:
            bowl[category] = []
            continue
        raw_pool = [
            x
            for x in (pools_by_category.get(category) or [])
            if x not in exclude_set and x not in used_names
        ]
        prefer: Optional[list] = None
        if category == "Proteins" and primary_protein and primary_protein in raw_pool:
            prefer = [primary_protein]
        elif category == "Sauces":
            if prefer_sauce and prefer_sauce in raw_pool:
                prefer = [prefer_sauce]
            elif primary_protein and used_combos is not None:
                _user_sauces = [
                    x
                    for x in (must_include_by_category.get(category) or [])
                    if x in raw_pool
                ]
                if _user_sauces:
                    prefer = list(dict.fromkeys(_user_sauces))
                else:
                    sp = sauce_pool or raw_pool
                    unused = [
                        s
                        for s in sp
                        if s in raw_pool
                        and _heybo_primary_bowl_key(primary_protein, s) not in used_combos
                    ]
                    if unused:
                        prefer = unused

        ranked = _rank_pool(raw_pool, category)
        if prefer:
            ranked = list(dict.fromkeys([x for x in prefer if x in raw_pool] + ranked))

        # Force primary protein into slot 1 when possible (variety / sauce keys).
        if (
            category == "Proteins"
            and primary_protein
            and primary_protein in raw_pool
            and primary_protein in ranked
        ):
            ranked = [primary_protein] + [x for x in ranked if x != primary_protein]

        # User Include/Extra first — same contract as the normal fill path.
        forced = [
            x
            for x in (must_include_by_category.get(category) or [])
            if x not in exclude_set
            and x not in used_names
            and (
                x in (pools_by_category.get(category) or [])
                or x in raw_pool
            )
        ]
        # Also accept forced items that are in the catalog pool even if already filtered.
        if category == "Proteins":
            for x in must_include_by_category.get(category) or []:
                if (
                    x not in forced
                    and x not in exclude_set
                    and x not in used_names
                    and x in (pools_by_category.get(category) or [])
                ):
                    forced.append(x)

        reserve = _finisher_reserve(category)
        picks: list = []
        for ing in forced:
            if len(picks) >= cat_max:
                break
            # Include always wins over incompatibility with non-user items; still
            # skip if incompatible with another user Include (same as normal path).
            if not _compatible(ing) and ing not in user_included:
                continue
            picks.append(ing)
            _commit(ing, category)

        for ing in ranked:
            if len(picks) >= cat_max:
                break
            if ing in picks or ing in used_names:
                continue
            if not _compatible(ing):
                continue
            # ALWAYS respect Max — never force a category-min item past the band.
            # Exceptions: primary protein slot 1, and user Include already committed above.
            exceeds = _would_exceed_max(ing, reserve)
            if exceeds:
                if not (
                    category == "Proteins"
                    and not picks
                    and primary_protein
                    and ing == primary_protein
                ):
                    continue
            must_take = len(picks) < cat_min
            # Mins met: stop packing Proteins; light structural sides/finishers only.
            if (
                not must_take
                and min_targets
                and _mins_met()
                and category == "Proteins"
            ):
                break
            if (
                not must_take
                and min_targets
                and _mins_met()
                and category == "Bases"
                and picks
            ):
                break
            if (
                min_targets
                and _mins_met()
                and category == "Warm sides"
                and len(picks) >= 2
            ):
                break
            if (
                min_targets
                and _mins_met()
                and category == "Cold sides"
                and len(bowl.get("Warm sides") or []) + len(picks) >= 3
            ):
                break
            # After Mins met, finishers (Dip/Garnish/Sauce) take at most one low item.
            if (
                not must_take
                and min_targets
                and _mins_met()
                and category in ("Dips", "Garnish", "Sauces")
                and picks
            ):
                break
            picks.append(ing)
            _commit(ing, category)
            if (
                category == "Proteins"
                and min_targets
                and _mins_met()
                and len(picks) >= max(cat_min, 1)
            ):
                break
            # Warm/Cold: stop as soon as Mins are met (don't keep packing high sides).
            if (
                category in ("Warm sides", "Cold sides")
                and min_targets
                and _mins_met()
                and len(picks) >= max(cat_min, 1)
            ):
                break

        bowl[category] = list(picks)
    return head


def _heybo_init_nutrient_budget(
    nutrient_filters: list,
    relaxation_levels: dict,
    df: "pd.DataFrame",
) -> dict:
    """
    Build a per-nutrient remaining budget from the (already relaxed) nutrient filters.
    Only nutrients with a Max constraint that exist as a df column are tracked.
    Returns {col_name: {"remaining_max": float, "min": float|None}}
    """
    from .diet import get_heybo_relaxed_nutrient_filters, _resolve_nutrient_filter_key as _rkey
    if not nutrient_filters:
        return {}
    relaxed = get_heybo_relaxed_nutrient_filters(
        nutrient_filters, relaxation_levels=relaxation_levels or {}
    )
    budget: dict = {}
    for nf in relaxed:
        raw = (nf.get("Nutrient") or "").strip()
        key = _rkey(raw) if raw else raw
        if not key or key not in df.columns:
            continue
        rng = nf.get("Range") or {}
        mx = rng.get("Max")
        mn = rng.get("Min")
        if mx is None:
            continue
        budget[key] = {
            "remaining_max": float(mx),
            "min": float(mn) if mn is not None else None,
        }
    return budget


def _heybo_budget_filter_pool(
    pool: list,
    nutrient_budget: dict,
    df: "pd.DataFrame",
    cache: dict,
    needed: int,
) -> list:
    """
    Return items from pool whose individual nutrient value fits within the remaining budget.
    If nothing fits the per-item budget, returns the full pool unchanged so that
    random.sample retains variety — post-assembly validation handles over-budget bowls.
    """
    if not nutrient_budget or not pool:
        return pool
    fitting = [
        ing for ing in pool
        if all(
            _heybo_ingredient_nutrient_val(ing, k, df, cache) <= bud["remaining_max"]
            for k, bud in nutrient_budget.items()
        )
    ]
    # When items fit: use only the budget-safe subset (calorie-aware picking).
    # When nothing fits: return the full pool so variety is not artificially reduced;
    # the bowl will either pass or fail post-assembly nutrient validation as before.
    return fitting if fitting else pool


def _heybo_budget_deduct(
    nutrient_budget: dict,
    ingredients: list,
    df: "pd.DataFrame",
    cache: dict,
) -> None:
    """Subtract each ingredient's nutrient value from the remaining budget."""
    for ing in ingredients:
        for key, bud in nutrient_budget.items():
            bud["remaining_max"] -= _heybo_ingredient_nutrient_val(ing, key, df, cache)


# ---------------------------------------------------------------------------


def _heybo_build_generation_catalog_pools(
    *,
    bases,
    proteins,
    extra_proteins,
    warm_sides,
    cold_sides,
    extra_warm_sides,
    extra_cold_sides,
    dips,
    garnishes,
    sauces,
    sauces_for_attempt,
    exclude_list,
    minimum_fill_pools,
    category_limits,
    only_mode: bool,
    exclude_ignore_categories=None,
) -> dict:
    """Preference pools merged with strict-safety top-up when a category is below DB minimum."""
    if only_mode or not minimum_fill_pools:
        return _heybo_attempt_catalog_pools(
            bases=bases,
            proteins=proteins,
            extra_proteins=extra_proteins,
            warm_sides=warm_sides,
            cold_sides=cold_sides,
            extra_warm_sides=extra_warm_sides,
            extra_cold_sides=extra_cold_sides,
            dips=dips,
            garnishes=garnishes,
            sauces=sauces,
            sauces_for_attempt=sauces_for_attempt,
            exclude_list=exclude_list,
            exclude_ignore_categories=exclude_ignore_categories,
        )
    exc = set(exclude_list or [])
    ignore_exc = set(exclude_ignore_categories or [])
    out = {}
    for cat, pref, extras in (
        ("Bases", bases, []),
        (
            "Proteins",
            list(dict.fromkeys((proteins or []) + (extra_proteins or []))),
            ["Extra Proteins"],
        ),
        (
            "Warm sides",
            list(dict.fromkeys((warm_sides or []) + (extra_warm_sides or []))),
            ["Extra Warm sides"],
        ),
        (
            "Cold sides",
            list(dict.fromkeys((cold_sides or []) + (extra_cold_sides or []))),
            ["Extra Cold sides"],
        ),
        ("Dips", dips, []),
        ("Garnish", garnishes, []),
        ("Sauces", list(dict.fromkeys((sauces_for_attempt or []) + list(sauces or []))), []),
    ):
        db_min = int((category_limits.get(cat) or (0, 0))[0] or 0)
        cat_exc = _heybo_category_exclude_set(cat, exclude_list, ignore_exc)
        mf = _heybo_minimum_fill_names_for_category(cat, minimum_fill_pools)
        merged, _ = _heybo_merge_preference_and_minimum_fill_pool(
            [t for t in pref if t not in cat_exc],
            mf,
            expand_when_unique_below=max(db_min, 1) if db_min > 0 else 0,
            exclude_set=cat_exc,
        )
        out[cat] = merged
    return out


def _heybo_count_for_db_limit(bowl: dict, category: str) -> int:
    """Count bowl items against heybo.ingredient_wise_count for one display category."""
    if category in ("Proteins", "Warm sides", "Cold sides"):
        return _heybo_combined_count_for_category(bowl, category)
    return len(bowl.get(category) or [])


def _heybo_attempt_catalog_pools(
    *,
    bases,
    proteins,
    extra_proteins,
    warm_sides,
    cold_sides,
    extra_warm_sides,
    extra_cold_sides,
    dips,
    garnishes,
    sauces,
    sauces_for_attempt,
    exclude_list,
    exclude_ignore_categories=None,
) -> dict:
    """Full filtered catalog per category (DB mins must be met from these pools)."""
    ignore_exc = set(exclude_ignore_categories or [])
    sauce_src = list(dict.fromkeys((sauces_for_attempt or []) + list(sauces or [])))
    return {
        "Bases": [
            t for t in (bases or [])
            if t not in _heybo_category_exclude_set("Bases", exclude_list, ignore_exc)
        ],
        "Proteins": list(
            dict.fromkeys(
                [
                    t for t in (proteins or []) + (extra_proteins or [])
                    if t not in _heybo_category_exclude_set("Proteins", exclude_list, ignore_exc)
                ]
            )
        ),
        "Warm sides": list(
            dict.fromkeys(
                [
                    t for t in (warm_sides or []) + (extra_warm_sides or [])
                    if t not in _heybo_category_exclude_set("Warm sides", exclude_list, ignore_exc)
                ]
            )
        ),
        "Cold sides": list(
            dict.fromkeys(
                [
                    t for t in (cold_sides or []) + (extra_cold_sides or [])
                    if t not in _heybo_category_exclude_set("Cold sides", exclude_list, ignore_exc)
                ]
            )
        ),
        "Dips": [
            t for t in (dips or [])
            if t not in _heybo_category_exclude_set("Dips", exclude_list, ignore_exc)
        ],
        "Garnish": [
            t for t in (garnishes or [])
            if t not in _heybo_category_exclude_set("Garnish", exclude_list, ignore_exc)
        ],
        "Sauces": [
            t for t in sauce_src
            if t not in _heybo_category_exclude_set("Sauces", exclude_list, ignore_exc)
        ],
    }


def _heybo_refill_category_to_db_min(
    bowl: dict,
    category: str,
    db_min: int,
    catalog_pool: list,
    exclude_set: set,
    bowl_validations: dict,
    *,
    incompatible_pairs=None,
    user_included=None,
    nutrient_budget: Optional[dict] = None,
    df: Optional["pd.DataFrame"] = None,
    cache: Optional[dict] = None,
) -> bool:
    """Add catalog picks until category count reaches db_min. Returns True if satisfied."""
    if _heybo_count_for_db_limit(bowl, category) >= db_min:
        return True
    in_bowl = set(bowl.get(category) or [])
    if category == "Proteins":
        in_bowl |= set(bowl.get("Extra Proteins") or [])
    elif category == "Warm sides":
        in_bowl |= set(bowl.get("Extra Warm sides") or [])
    elif category == "Cold sides":
        in_bowl |= set(bowl.get("Extra Cold sides") or [])
    eligible = [t for t in catalog_pool if t not in exclude_set and t not in in_bowl]
    if incompatible_pairs:
        bowl_names = _heybo_bowl_ingredient_names(bowl)
        eligible = [
            t
            for t in eligible
            if _heybo_candidate_compatible(t, bowl_names, incompatible_pairs, user_included or set())
        ]
    # Do not refill past nutrient Max (e.g. protein 70–80) — random high-protein
    # Dip/Garnish refill was undoing joint-fill Max guards.
    if nutrient_budget and df is not None and cache is not None and eligible:
        _fit = [
            t
            for t in eligible
            if all(
                _heybo_ingredient_nutrient_val(t, k, df, cache) <= bud["remaining_max"]
                for k, bud in nutrient_budget.items()
            )
        ]
        if _fit:
            eligible = _fit
        else:
            return False
    if not eligible:
        return False
    need = db_min - _heybo_count_for_db_limit(bowl, category)
    # Prefer lowest Max contribution when a nutrient budget is active.
    if nutrient_budget and df is not None and cache is not None:
        eligible = sorted(
            eligible,
            key=lambda t: sum(
                _heybo_ingredient_nutrient_val(t, k, df, cache) for k in nutrient_budget
            ),
        )
        picks = eligible[: min(need, len(eligible))]
    else:
        picks = random.sample(eligible, min(need, len(eligible)))
    if category == "Proteins":
        bowl["Proteins"] = list(bowl.get("Proteins") or []) + picks
    else:
        bowl[category] = list(bowl.get(category) or []) + picks
    for pick in picks:
        bowl_validations["bowl_composition"].append(
            f"{category}: added {pick} to meet DB minimum {db_min} (strict-safety pool)"
        )
        if nutrient_budget and df is not None and cache is not None:
            _heybo_budget_deduct(nutrient_budget, [pick], df, cache)
    return _heybo_count_for_db_limit(bowl, category) >= db_min


def _heybo_bowl_meets_db_limits(
    bowl: dict,
    effective_limit_for_cat,
    catalog_by_category: dict,
    exclude_set: set,
    bowl_validations: dict,
    *,
    try_refill: bool = True,
    only_allowed_names=None,
    min_total_sides: Optional[int] = None,
    incompatible_pairs=None,
    user_included=None,
    full_exclude_conflict_categories=None,
    nutrient_budget: Optional[dict] = None,
    df: Optional["pd.DataFrame"] = None,
    cache: Optional[dict] = None,
) -> bool:
    """
    Enforce ingredient_wise_count min/max for every REQUIRED_BOWL_CATEGORIES row.
    Never accept a bowl below DB min when the filtered catalog could supply one.
    In Only mode, refills use Include ∪ Extra only; structural catalog fills (e.g. base)
    may already satisfy DB minimum even when they are outside only_allowed_names.
    When min_total_sides is set, also require that many warm+cold items (main + Extra *).
    """
    if min_total_sides is not None and min_total_sides > 0:
        side_total = _heybo_combined_warm_cold_side_count(bowl)
        if side_total < min_total_sides:
            bowl_validations["bowl_composition"].append(
                f"Warm+cold sides: {side_total} below structural minimum {min_total_sides} — skipped"
            )
            return False
    for category in REQUIRED_BOWL_CATEGORIES:
        lim = effective_limit_for_cat(category)
        if not lim or len(lim) < 2:
            continue
        db_min, db_max = int(lim[0]), int(lim[1])
        catalog_pool = list(catalog_by_category.get(category) or [])
        if only_allowed_names is not None:
            refill_pool = [t for t in catalog_pool if t in only_allowed_names]
        else:
            refill_pool = catalog_pool
        count = _heybo_count_for_db_limit(bowl, category)
        # Only mode may auto-fill Bases (etc.) from the full catalog even when not in Include/Extra.
        if db_min > 0 and count < db_min and len(refill_pool) < db_min:
            bowl_validations["bowl_composition"].append(
                f"{category}: catalog has {len(refill_pool)} allowed items "
                f"but DB minimum is {db_min} — skipped"
            )
            return False
        if db_max >= 0 and count > db_max:
            bowl_validations["bowl_composition"].append(
                f"{category}: {count} exceeds DB maximum {db_max} — skipped"
            )
            return False
        if db_min > 0 and count < db_min:
            if try_refill and refill_pool:
                cat_exclude = (
                    set()
                    if category in (full_exclude_conflict_categories or set())
                    else exclude_set
                )
                _heybo_refill_category_to_db_min(
                    bowl,
                    category,
                    db_min,
                    refill_pool,
                    cat_exclude,
                    bowl_validations,
                    incompatible_pairs=incompatible_pairs,
                    user_included=user_included,
                    nutrient_budget=nutrient_budget,
                    df=df,
                    cache=cache,
                )
                count = _heybo_count_for_db_limit(bowl, category)
            if count < db_min:
                bowl_validations["bowl_composition"].append(
                    f"{category}: {count} below DB minimum {db_min} — skipped"
                )
                return False
    return True


def _cap_heybo_main_extra_pair(bowl: dict, normal_key: str, extra_key: str, max_total: int) -> None:
    """Trim Extra * list first, then main list, so total items never exceeds max_total."""
    if max_total is None or max_total < 0:
        return
    main = list(bowl.get(normal_key) or [])
    ext = list(bowl.get(extra_key) or [])
    total = len(main) + len(ext)
    if total <= max_total:
        return
    over = total - max_total
    while over > 0 and ext:
        ext.pop()
        over -= 1
    while over > 0 and main:
        main.pop()
        over -= 1
    bowl[normal_key] = main
    bowl[extra_key] = ext


def _ordered_proteins_for_bowl(bowl):
    return list(bowl.get("Proteins") or []) + list(bowl.get("Extra Proteins") or [])


def _set_bowl_proteins_from_ordered(bowl, ordered):
    """Keep full order in Proteins; Extra Proteins is filled only by user Include/Extra redistribution."""
    bowl["Proteins"] = list(ordered)
    bowl["Extra Proteins"] = []


def _primary_protein_name(bowl):
    ordered = _ordered_proteins_for_bowl(bowl)
    return ordered[0] if ordered else None


def _heybo_primary_bowl_key(primary_protein, primary_sauce) -> str:
    """
    Canonical primary identity string for diversity checks; matches ``Bowl Name`` base
    (``Protein_Sauce``, sauce-only, or Lulu-BYB) used after a bowl is finalized.
    """
    if primary_protein and primary_sauce:
        return f"{primary_protein}_{primary_sauce}"
    if primary_sauce:
        return str(primary_sauce)
    return "Lulu-BYB"


def _heybo_user_requested_ingredient_set(include_list, extra_list):
    """Include ∪ Extra names the customer explicitly asked for."""
    return set(include_list or []) | set(extra_list or [])


def _heybo_enforce_primary_combo_diversity(primary_protein, primary_sauce, user_requested):
    """
    Protein+sauce uniqueness applies only when both primary ingredients were system-added.
    If either is Include/Extra, keep the customer's choice even when the pair repeats.
    """
    if not primary_protein or not primary_sauce:
        return False
    user_requested = user_requested or set()
    return (
        primary_protein not in user_requested
        and primary_sauce not in user_requested
    )


def _df_has_ingredient_category(df, ingredient_name, category):
    if df is None or df.empty or not ingredient_name:
        return False
    m = (df["ingredient_name"] == ingredient_name) & (df["category"] == category)
    return bool(m.any())


def _per_category_include_extra_order(include_list, extra_list, ingredient_category_map, required_categories):
    """Preserve Include-then-Extra order per base category (for mapping user 'Extra' to Extra * SKUs)."""
    out = {cat: {"inc": [], "ext": []} for cat in required_categories}
    for ing in include_list:
        cat_val = ingredient_category_map.get(ing)
        if not cat_val:
            continue
        mapped = HEYBO_EXTRA_CATEGORIES.get(cat_val, cat_val)
        if mapped in out:
            out[mapped]["inc"].append(ing)
    for ing in extra_list:
        cat_val = ingredient_category_map.get(ing)
        if not cat_val:
            continue
        mapped = HEYBO_EXTRA_CATEGORIES.get(cat_val, cat_val)
        if mapped in out:
            out[mapped]["ext"].append(ing)
    return out


def _redistribute_proteins_user_extra_slots(bowl, df, per_category_include_extra):
    """Re-assign Proteins vs Extra Proteins using Include/Extra list counts (run again after sauce diversity reorder)."""
    normal_cat, extra_cat = "Proteins", "Extra Proteins"
    items = list(bowl.get(normal_cat, [])) + list(bowl.get(extra_cat, []))
    slots = per_category_include_extra.get(normal_cat) or {"inc": [], "ext": []}
    inc_left = Counter(slots["inc"])
    ext_left = Counter(slots["ext"])
    new_norm = []
    new_extra = []
    for ing in items:
        if inc_left.get(ing, 0) > 0:
            inc_left[ing] -= 1
            new_norm.append(ing)
        elif ext_left.get(ing, 0) > 0 and _df_has_ingredient_category(df, ing, extra_cat):
            ext_left[ing] -= 1
            new_extra.append(ing)
        else:
            new_norm.append(ing)
    bowl[normal_cat] = new_norm
    bowl[extra_cat] = new_extra


def _redistribute_user_extra_slots_to_extra_categories(
    bowl, df, per_category_include_extra, user_prefix_lengths, required_categories
):
    """
    User Ingredients.Extra means an additional portion: put those slots in Extra Proteins / Warm / Cold
    when the dataframe has that ingredient under the Extra * category (distinct SKU / price row).
    Warm/Cold: leading user-requested slots (before random fill) stay in order Include then Extra.
    Proteins: full list is walked with counters (order may change when Sauces reorders proteins).
    """
    for extra_cat, normal_cat in HEYBO_EXTRA_CATEGORIES.items():
        if normal_cat not in required_categories:
            continue
        if normal_cat == "Proteins":
            _redistribute_proteins_user_extra_slots(bowl, df, per_category_include_extra)
            continue

        items = list(bowl.get(normal_cat, []))
        slots = per_category_include_extra.get(normal_cat) or {"inc": [], "ext": []}
        inc_left = Counter(slots["inc"])
        ext_left = Counter(slots["ext"])
        new_extra = list(bowl.get(extra_cat, []))
        prefix_n = user_prefix_lengths.get(normal_cat, 0)
        if prefix_n <= 0:
            continue
        user_part = items[:prefix_n]
        tail = items[prefix_n:]
        new_norm = []
        for ing in user_part:
            if inc_left.get(ing, 0) > 0:
                inc_left[ing] -= 1
                new_norm.append(ing)
            elif ext_left.get(ing, 0) > 0 and _df_has_ingredient_category(df, ing, extra_cat):
                ext_left[ing] -= 1
                new_extra.append(ing)
            else:
                new_norm.append(ing)
        bowl[normal_cat] = new_norm + tail
        bowl[extra_cat] = new_extra


def prefer_unused_protein_sauce_combos(
    available_proteins,
    available_sauces,
    used_combos,
    include_list,
    extra_list,
):
    """
    Reorder protein and sauce pools to prefer primary bowl keys (``Protein_Sauce`` style) not
    already in ``used_combos`` (set of those key strings).
    User-requested proteins/sauces are left untouched in meaning; pools still contain them.
    """
    if not available_proteins or not available_sauces:
        return available_proteins, available_sauces
    include_set = set(include_list or [])
    extra_set = set(extra_list or [])
    user_requested_proteins = [p for p in available_proteins if p in include_set or p in extra_set]
    user_requested_sauces = [s for s in available_sauces if s in include_set or s in extra_set]

    if user_requested_proteins:
        preferred_proteins = list(user_requested_proteins)
    else:
        dressings_to_check = user_requested_sauces if user_requested_sauces else available_sauces
        protein_scores = {}
        for protein in available_proteins:
            unused_count = sum(
                1 for s in dressings_to_check if _heybo_primary_bowl_key(protein, s) not in used_combos
            )
            protein_scores[protein] = unused_count
        sorted_proteins = sorted(available_proteins, key=lambda p: protein_scores[p], reverse=True)
        proteins_with_unused = [p for p in sorted_proteins if protein_scores[p] > 0]
        preferred_proteins = proteins_with_unused if proteins_with_unused else list(available_proteins)

    if user_requested_sauces:
        preferred_sauces = list(user_requested_sauces)
    else:
        proteins_to_check = user_requested_proteins if user_requested_proteins else available_proteins
        sauce_scores = {}
        for sauce in available_sauces:
            unused_count = sum(
                1 for p in proteins_to_check if _heybo_primary_bowl_key(p, sauce) not in used_combos
            )
            sauce_scores[sauce] = unused_count
        sorted_sauces = sorted(available_sauces, key=lambda s: sauce_scores[s], reverse=True)
        sauces_with_unused = [s for s in sorted_sauces if sauce_scores[s] > 0]
        preferred_sauces = sauces_with_unused if sauces_with_unused else list(available_sauces)

    return preferred_proteins, preferred_sauces


def _unused_protein_sauce_pairs_exist(proteins_list, sauces_list, used_combos):
    if not proteins_list or not sauces_list:
        return False
    for p in proteins_list:
        for s in sauces_list:
            if _heybo_primary_bowl_key(p, s) not in used_combos:
                return True
    return False


def _try_fix_duplicate_primary_combo(
    bowl,
    protein_pool,
    sauce_pool,
    used_combos,
    bowl_validations,
    max_proteins_for_bowl=None,
    user_requested=None,
):
    """
    If the primary pair's bowl key is already in ``used_combos`` but another unused (p, s) exists in
    the filtered pools, swap sauce, reorder proteins, or add a protein (when under max) so the
    primary bowl key is fresh. Primary protein is first in ordered proteins (before user-extra split).
    Returns True if the bowl was modified.
    Never changes a user-requested (Include/Extra) primary protein or sauce.
    """
    user_requested = user_requested or set()
    prots = _ordered_proteins_for_bowl(bowl)
    sauces_in_bowl = bowl.get("Sauces") or []
    if not prots or not sauces_in_bowl:
        return False
    if not protein_pool or not sauce_pool:
        return False
    if not _unused_protein_sauce_pairs_exist(protein_pool, sauce_pool, used_combos):
        return False
    primary_p, primary_s = prots[0], sauces_in_bowl[0]
    if _heybo_primary_bowl_key(primary_p, primary_s) not in used_combos:
        return False
    if not _heybo_enforce_primary_combo_diversity(primary_p, primary_s, user_requested):
        return False
    for alt_s in sauce_pool:
        if alt_s == primary_s or alt_s in user_requested:
            continue
        if _heybo_primary_bowl_key(primary_p, alt_s) not in used_combos:
            bowl["Sauces"] = [alt_s]
            bowl_validations["bowl_composition"].append(
                f"Diversity: swapped sauce to {alt_s} to avoid repeating primary pair"
            )
            return True
    for alt_p in prots[1:]:
        if alt_p in user_requested:
            continue
        if _heybo_primary_bowl_key(alt_p, primary_s) not in used_combos:
            new_order = [alt_p] + [x for x in prots if x != alt_p]
            _set_bowl_proteins_from_ordered(bowl, new_order)
            bowl_validations["bowl_composition"].append(
                f"Diversity: reordered proteins to lead with {alt_p} for unused primary+sauce pair"
            )
            return True
    for alt_s in sauce_pool:
        if alt_s in user_requested:
            continue
        for alt_p in prots:
            if alt_p in user_requested:
                continue
            if _heybo_primary_bowl_key(alt_p, alt_s) not in used_combos:
                bowl["Sauces"] = [alt_s]
                new_order = [alt_p] + [x for x in prots if x != alt_p]
                _set_bowl_proteins_from_ordered(bowl, new_order)
                bowl_validations["bowl_composition"].append(
                    f"Diversity: set primary pair to ({alt_p}, {alt_s}) to avoid repeat"
                )
                return True
    cap = max_proteins_for_bowl if max_proteins_for_bowl is not None else len(prots)
    if len(prots) < cap:
        for alt_p in protein_pool:
            if alt_p in prots or alt_p in user_requested:
                continue
            for alt_s in sauce_pool:
                if alt_s in user_requested:
                    continue
                if _heybo_primary_bowl_key(alt_p, alt_s) not in used_combos:
                    new_order = [alt_p] + prots
                    _set_bowl_proteins_from_ordered(bowl, new_order)
                    bowl["Sauces"] = [alt_s]
                    bowl_validations["bowl_composition"].append(
                        f"Diversity: added {alt_p} as primary to unlock unused pair ({alt_p}, {alt_s})"
                    )
                    return True
    return False


def _build_heybo_ingredient_mapping(df):
    """sku_code -> {'name','category'} for Apriori suggestion resolution."""
    out = {}
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        sku = row.get("sku_code")
        if pd.isna(sku):
            continue
        s = str(sku).strip()
        if not s:
            continue
        out[s] = {
            "name": row.get("ingredient_name", "Unknown"),
            "category": row.get("category", "Unknown"),
        }
    return out


def _find_heybo_apriori_enhancements(selected_ingredients, apriori_rules, ingredient_mapping):
    """Salad `find_apriori_enhancements`: top 10 unique partners, overlap then confidence."""
    if apriori_rules is None or getattr(apriori_rules, "empty", True):
        return []
    selected = [str(x).strip() for x in (selected_ingredients or []) if x and str(x).strip()]
    selected_set = set(selected)
    if not selected_set:
        return []
    sku_to_name = {sku: details.get("name") for sku, details in ingredient_mapping.items()}
    col_names = list(apriori_rules.columns)
    antecedent_idx = col_names.index("antecedent")
    consequent_idx = col_names.index("consequent")
    confidence_idx = col_names.index("confidence") if "confidence" in col_names else None
    lift_idx = col_names.index("lift") if "lift" in col_names else None
    support_idx = col_names.index("support") if "support" in col_names else None
    enhancements = []
    for rule in apriori_rules.itertuples(index=False):
        antecedent_raw = str(rule[antecedent_idx]).strip('"\'')
        consequent_raw = str(rule[consequent_idx]).strip('"\'')
        antecedent_skus = (
            [x.strip() for x in antecedent_raw.split(",")] if "," in antecedent_raw else [antecedent_raw]
        )
        consequent_skus = (
            [x.strip() for x in consequent_raw.split(",")] if "," in consequent_raw else [consequent_raw]
        )
        antecedent_matched_names = []
        for ant_sku in antecedent_skus:
            if ant_sku in sku_to_name:
                ant_name = sku_to_name[ant_sku]
                if ant_name in selected_set:
                    antecedent_matched_names.append(ant_name)
            elif ant_sku in selected_set:
                antecedent_matched_names.append(ant_sku)
        if not antecedent_matched_names:
            continue
        for cons_sku in consequent_skus:
            cons_name = sku_to_name.get(cons_sku, cons_sku)
            if cons_name in selected_set:
                continue
            details = None
            sku_code = cons_sku
            for mapped_sku, mapped_details in ingredient_mapping.items():
                if mapped_details.get("name") == cons_name:
                    details = mapped_details
                    sku_code = mapped_sku
                    break
            if not details:
                continue
            enhancements.append(
                {
                    "ingredient_name": cons_name,
                    "sku_code": sku_code,
                    "category": details.get("category", "Unknown"),
                    "confidence": rule[confidence_idx] if confidence_idx is not None else 0.0,
                    "lift": rule[lift_idx] if lift_idx is not None else 0.0,
                    "support": rule[support_idx] if support_idx is not None else 0.0,
                    "paired_with": list(antecedent_matched_names),
                    "reason": f"Frequently paired with {', '.join(antecedent_matched_names)}",
                }
            )
    enhancements.sort(
        key=lambda x: (len(x.get("paired_with") or []), x.get("confidence") or 0),
        reverse=True,
    )
    unique = []
    seen = set()
    for e in enhancements:
        nm = e["ingredient_name"]
        if nm in seen:
            continue
        seen.add(nm)
        unique.append(e)
    return unique[:10]


def _suggest_heybo_apriori_combinations(selected_ingredients, apriori_rules, ingredient_mapping):
    """Salad `suggest_apriori_combinations`: keep paired_with for CP-SAT overlap ranking."""
    enhancements = _find_heybo_apriori_enhancements(
        selected_ingredients, apriori_rules, ingredient_mapping
    )
    suggestions = []
    for e in enhancements:
        suggestions.append(
            {
                "sku_code": e["sku_code"],
                "ingredient_name": e["ingredient_name"],
                "category": e["category"],
                "reason": e["reason"],
                "confidence": e["confidence"],
                "frequency": e["support"],
                "paired_with": list(e.get("paired_with") or []),
            }
        )
    suggestions.sort(key=lambda x: (x["confidence"], x["frequency"]), reverse=True)
    return suggestions


def _heybo_load_apriori_suggestions(include_list, df):
    if not include_list:
        return []
    apriori_rules = load_heybo_apriori_rules()
    ingredient_mapping = _build_heybo_ingredient_mapping(df)
    if apriori_rules is None or apriori_rules.empty or not ingredient_mapping:
        return []
    return _suggest_heybo_apriori_combinations(include_list, apriori_rules, ingredient_mapping)


def _apriori_category_match(target_category, suggestion_category):
    if target_category == "Proteins":
        return suggestion_category in {"Proteins", "Extra Proteins"}
    if target_category == "Warm sides":
        return suggestion_category in {"Warm sides", "Extra Warm sides"}
    if target_category == "Cold sides":
        return suggestion_category in {"Cold sides", "Extra Cold sides"}
    return suggestion_category == target_category


def _apply_heybo_apriori_enhancement_to_pool(
    category,
    pool,
    global_validations,
    user_included_ingredients=None,
    incompatible_pairs=None,
):
    """
    Salad `apply_apriori_enhancement_to_pool`: reorder pool when the user asked
    for one or more Include/Extra items. Random 0..min(3, n) partners first.
    """
    user_included_ingredients = {
        str(x).strip()
        for x in (user_included_ingredients or [])
        if x and str(x).strip()
    }
    if not user_included_ingredients:
        return pool
    suggestions = global_validations.get("apriori_suggestions") or []
    if not suggestions:
        return pool
    incompatible_pairs = incompatible_pairs or {}
    category_suggestions = [s for s in suggestions if _apriori_category_match(category, s.get("category"))]
    if not category_suggestions:
        return pool
    category_suggestions.sort(
        key=lambda x: (
            len(set(x.get("paired_with") or []) & user_included_ingredients),
            x.get("confidence") or 0,
            x.get("frequency") or 0,
        ),
        reverse=True,
    )
    compatible_suggestions = []
    for s in category_suggestions:
        nm = s.get("ingredient_name")
        if not nm:
            continue
        conflict = False
        if nm in incompatible_pairs and (set(incompatible_pairs[nm]) & set(user_included_ingredients)):
            conflict = True
        if not conflict:
            for ui in user_included_ingredients:
                if ui in incompatible_pairs and nm in incompatible_pairs[ui]:
                    conflict = True
                    break
        if not conflict:
            compatible_suggestions.append(s)
    in_pool_names = []
    seen_names = set()
    for s in compatible_suggestions:
        nm = s.get("ingredient_name")
        if nm and nm in pool and nm not in seen_names:
            seen_names.add(nm)
            in_pool_names.append(nm)
    k = random.randint(0, min(3, len(in_pool_names))) if in_pool_names else 0
    top = random.sample(in_pool_names, k) if k else []
    if not top:
        return pool
    apriori_pool = [ing for ing in pool if ing in top]
    non_apriori = []
    for ing in pool:
        if ing in top:
            continue
        conflict = False
        if ing in incompatible_pairs and (set(incompatible_pairs[ing]) & set(user_included_ingredients)):
            conflict = True
        if not conflict:
            for ui in user_included_ingredients:
                if ui in incompatible_pairs and ing in incompatible_pairs[ui]:
                    conflict = True
                    break
        if not conflict:
            non_apriori.append(ing)
    return apriori_pool + non_apriori


def generate_heybo_bowls(user_input):
    """Main function to generate Heybo BYB bowls."""
    try:
        df_heybo = load_heybo_data_from_db(user_input)
        print("[HEYBO] generate_heybo_bowls loaded data — constraint-first build active in this binary")
        if not isinstance(df_heybo, pd.DataFrame) or df_heybo.empty:
            raise ValueError("Heybo ingredient data not loaded properly")
        df_heybo = preprocess_heybo_dataframe(df_heybo)
        ingredient_category_map = (
            df_heybo.drop_duplicates(subset=["ingredient_name"])
            .set_index("ingredient_name")["category"]
            .to_dict()
        )
        ingredient_records = (
            df_heybo.drop_duplicates(subset=["ingredient_name"])
            .set_index("ingredient_name")
            .to_dict("index")
        )
        image_url_by_name = build_heybo_image_url_map(df_heybo)
        _raw_page_id = (
            (user_input.get("recommend_page_id") or user_input.get("RecommendPageId") or user_input.get("recommendation_page_id") or "")
            .strip()
        )
        _raw_session = (user_input.get("session_id") or user_input.get("SessionId") or "").strip()
        heybo_cfg = get_heybo_config(_raw_page_id, _raw_session, page=user_input.get("Page"))
        # Pre-warm Light/Hearty cache from already-fetched config — avoids a separate
        # DB round-trip the first time evaluate_light_hearty_bowl is called per request.
        prime_light_hearty_cache_from_cfg(heybo_cfg)
        ingredients_early = user_input.get("Ingredients") or {}
        only_mode = _heybo_ingredients_only_mode(ingredients_early, user_input)
        if only_mode:
            user_input["_heybo_full_user_input_snapshot"] = copy.deepcopy(user_input)
            user_input["_heybo_only_mode_generation"] = True
            _heybo_strip_non_only_filters_for_generation(user_input)
        user_input = preprocess_heybo_filters(user_input)
        sustainable_on = _is_truthy_flag(user_input.get("Sustainable"))
        if sustainable_on and not _has_sustainable_co2_data(df_heybo):
            user_input["_heybo_sustainable_no_data"] = True
            return {
                "error": "No sustainable data found",
                "message_to_user": build_heybo_message_to_user(
                    [], None, user_input=user_input
                ),
            }
        ingredients = user_input.get("Ingredients", {})
        extra_list = list(ingredients.get('Extra', []))
        extras_requested = len(extra_list) > 0
        categories_with_extras = set()
        for extra_ingredient in extra_list:
            ingredient_category = ingredient_category_map.get(extra_ingredient)
            if ingredient_category in HEYBO_EXTRA_CATEGORIES:
                normal_category = HEYBO_EXTRA_CATEGORIES[ingredient_category]
                categories_with_extras.add(normal_category)
        print(f"Categories with extras requested: {categories_with_extras}")
        global_validations = {
            "flavor_adjustments": [],
            "diet_fallbacks": [],
            "allergen_exclusions": [],
            "data_loading": [f"Successfully loaded {len(df_heybo)} Heybo ingredients from database"],
            "filter_summary": (
                [
                    "Only mode: bowl 1 uses Include, Extra, and Exclude only (base added if needed); "
                    "bowls 2–5 apply your other filters and may add ingredients for variety"
                ]
                if only_mode
                else []
            ),
            "extras_logic": [],
            "cuisine_matches": [],
            "preparation_method_matches": [],
            "category_limit_notices": [],
            "light_hearty_matches": [],
            "apriori_suggestions": [],
            "guideline_adjustments": [],
            "category_limits_used": "normal",
        }
        _pre_ga = user_input.pop("_heybo_guideline_adjustments_from_preprocess", None)
        if _pre_ga:
            global_validations["guideline_adjustments"].extend(_pre_ga)
        is_balanced_requested = _heybo_apply_balanced_override(
            user_input, global_validations
        )
        for _msg in user_input.pop("_nutrient_preprocess_messages", []):
            global_validations["diet_fallbacks"].append(_msg)
        if categories_with_extras:
            global_validations["extras_logic"].append(
                "Extra requested for: "
                + ", ".join(sorted(categories_with_extras))
                + " — raised category ceiling only as needed for your Extra portions "
                "(not full customization fill)"
            )
        else:
            global_validations["extras_logic"].append("No extra ingredients requested - using normal category limits")
        category_limits = heybo_cfg["category_limits"]
        category_limits_customization = heybo_cfg.get("category_limits_customization", category_limits)
        category_limits_extras = heybo_cfg["category_limits_extras"]
        previous_bowls = heybo_cfg["previous_bowls"]
        max_bowl_weight = heybo_cfg["max_bowl_weight"]
        if not only_mode:
            normalize_heybo_price_filters(
                user_input, heybo_cfg["price_config"], global_validations
            )
        incompatible_pairs = get_heybo_incompatible_pairs(df_heybo)
        required_categories = ["Bases", "Proteins", "Warm sides", "Cold sides", "Dips", "Garnish", "Sauces"]
        prep_active = bool(
            isinstance(user_input.get("PreparationMethod"), dict)
            and any(v is True for v in user_input.get("PreparationMethod", {}).values())
        )
        cuisine_filters_val = user_input.get("CuisineFilters")
        cuisine_active = (
            (isinstance(cuisine_filters_val, list) and len(cuisine_filters_val) > 0)
            or (isinstance(cuisine_filters_val, dict) and any(v is True for v in cuisine_filters_val.values()))
        )
        diet_active = bool(user_input.get("DietFilters"))
        allergen_active = bool(user_input.get("AllergenFilters"))
        flavor_active = bool(
            isinstance(user_input.get("FlavorPreferences"), dict)
            and any(v for v in user_input.get("FlavorPreferences", {}).values())
        )
        flavor_thresholds = (heybo_cfg or {}).get("flavor_thresholds") or {}

        def rebuild_filtered_context():
            filtered, fb_categories, err = apply_all_ingredient_filters(
                df_heybo, user_input, global_validations, heybo_cfg=heybo_cfg
            )
            if err:
                return None, fb_categories, err, {}, [], [], [], [], [], [], [], [], [], []
            empty_cats_local = []
            category_counts_local = {}
            for cat in required_categories:
                safe = filtered[filtered['category'] == cat]
                category_counts_local[cat] = len(safe)
                print(f"Category: {cat}, Safe count: {len(safe)}")
                if safe.empty:
                    empty_cats_local.append(cat)
            if filtered.empty:
                active_filter_labels = []
                if allergen_active:
                    active_filter_labels.append("allergen")
                if diet_active:
                    active_filter_labels.append("diet")
                if cuisine_active:
                    active_filter_labels.append("cuisine")
                if prep_active:
                    active_filter_labels.append("preparation")
                if flavor_active:
                    active_filter_labels.append("flavor")
                if active_filter_labels:
                    filter_text = "with current " + ", ".join(active_filter_labels) + " filters."
                else:
                    filter_text = "with current filters."
                return None, fb_categories, (
                    "No safe ingredients available "
                    f"{filter_text}"
                ), {}, [], [], [], [], [], [], [], [], [], []
            if empty_cats_local:
                global_validations["filter_summary"].append(
                    "Categories unavailable after filters: " + ", ".join(empty_cats_local)
                )
            global_validations["filter_summary"].append(
                f"Category availability: {', '.join([f'{cat}: {count}' for cat, count in category_counts_local.items()])}"
            )
            category_to_ingredients_local = {
                cat: grp["ingredient_name"].tolist()
                for cat, grp in filtered.groupby("category")
            }
            return (
                filtered,
                fb_categories,
                None,
                category_to_ingredients_local,
                category_to_ingredients_local.get("Bases", []),
                category_to_ingredients_local.get("Proteins", []),
                category_to_ingredients_local.get("Warm sides", []),
                category_to_ingredients_local.get("Cold sides", []),
                category_to_ingredients_local.get("Dips", []),
                category_to_ingredients_local.get("Garnish", []),
                category_to_ingredients_local.get("Sauces", []),
                category_to_ingredients_local.get("Extra Proteins", []),
                category_to_ingredients_local.get("Extra Warm sides", []),
                category_to_ingredients_local.get("Extra Cold sides", []),
            )

        (
            filtered_ingredients,
            fallback_categories,
            filter_error,
            category_to_ingredients,
            bases,
            proteins,
            warm_sides,
            cold_sides,
            dips,
            garnishes,
            sauces,
            extra_proteins,
            extra_warm_sides,
            extra_cold_sides,
        ) = rebuild_filtered_context()
        if filter_error:
            return {
                "error": filter_error,
                "message_to_user": build_heybo_message_to_user(
                    [], None, user_input=user_input, error=filter_error
                ),
            }
        _strict_safety_df = (
            build_heybo_strict_safety_catalog(df_heybo, user_input, global_validations)
            if not only_mode
            else df_heybo.iloc[0:0]
        )
        minimum_fill_pools = (
            minimum_fill_pools_by_category(_strict_safety_df) if not only_mode else {}
        )
        ingredient_repeat_count = {}
        include_list = list(ingredients.get('Include', []))
        exclude_list = list(ingredients.get('Exclude', []))
        available_names = set(df_heybo['ingredient_name'])
        include_list = [i for i in include_list if i in available_names]
        exclude_list = [i for i in exclude_list if i in available_names]
        for ing in include_list:
            ingredient_repeat_count[ing] = ingredient_repeat_count.get(ing, 0) + 1
        extra_ingredients_added = []
        for extra_ingredient in extra_list:
            if extra_ingredient in available_names:
                ingredient_category = ingredient_category_map.get(extra_ingredient)
                if ingredient_category:
                    ingredient_repeat_count[extra_ingredient] = ingredient_repeat_count.get(extra_ingredient, 0) + 1
                    if ingredient_category in HEYBO_EXTRA_CATEGORIES:
                        extra_ingredients_added.append(f"{extra_ingredient} ({ingredient_category})")
                        print(f"Added extra ingredient: {extra_ingredient} from category: {ingredient_category}")
                    else:
                        extra_ingredients_added.append(f"{extra_ingredient} (regular)")
                        print(f"Added regular ingredient as extra: {extra_ingredient} from category: {ingredient_category}")
        if extra_ingredients_added:
            global_validations["extras_logic"].append(f"Added extra ingredients: {', '.join(extra_ingredients_added)}")
        else:
            global_validations["extras_logic"].append("No extra ingredients added")

        # Per-category: if Include/Extra exceeds normal max, that category alone may
        # use customization max as a ceiling for user-asked items (not a fill target).
        requested_count_by_category = {}
        for ing, count in ingredient_repeat_count.items():
            cat_val = ingredient_category_map.get(ing)
            if not cat_val:
                continue
            mapped_category = HEYBO_EXTRA_CATEGORIES.get(cat_val, cat_val)
            if mapped_category in category_limits:
                requested_count_by_category[mapped_category] = requested_count_by_category.get(mapped_category, 0) + count
        categories_with_customization = _heybo_categories_with_request_overflow(
            requested_count_by_category, category_limits
        )
        if include_list:
            try:
                global_validations["apriori_suggestions"] = _heybo_load_apriori_suggestions(
                    include_list, df_heybo
                )
            except Exception as e:
                print(f"Apriori enhancement skipped: {e}")
                global_validations["apriori_suggestions"] = []
        else:
            global_validations["apriori_suggestions"] = []
        only_allowed_names = set()
        if only_mode:
            only_allowed_names = (set(include_list) | set(extra_list)) & available_names
            if not only_allowed_names:
                oerr = (
                    "Only mode requires at least one ingredient in Include or Extra that exists "
                    "in the Heybo catalog after filters."
                )
                return {
                    "error": oerr,
                    "message_to_user": build_heybo_message_to_user(
                        [], global_validations, user_input=user_input, error=oerr
                    ),
                }
            categories_with_customization = set(REQUIRED_BOWL_CATEGORIES)
            global_validations["filter_summary"].append(
                "Only mode: bowl 1 uses customization limits with Include ∪ Extra only — "
                "a base may be added from the catalog if you did not request one; "
                f"bowls 2–{HEYBO_BOWLS_PER_PAGE} may add ingredients within category limits."
            )
        elif categories_with_customization:
            global_validations["extras_logic"].append(
                "Applied customization limits due to requested count overflow in: "
                + ", ".join(sorted(categories_with_customization))
            )
        requested_by_category = {cat: [] for cat in required_categories}
        for ing in include_list:
            cat_val = ingredient_category_map.get(ing)
            if not cat_val:
                continue
            mapped_category = HEYBO_EXTRA_CATEGORIES.get(cat_val, cat_val)
            if mapped_category in requested_by_category:
                requested_by_category[mapped_category].append(ing)
        for ing in extra_list:
            cat_val = ingredient_category_map.get(ing)
            if not cat_val:
                continue
            mapped_category = HEYBO_EXTRA_CATEGORIES.get(cat_val, cat_val)
            if mapped_category in requested_by_category:
                requested_by_category[mapped_category].append(ing)
        user_prefix_lengths = {c: len(requested_by_category[c]) for c in required_categories}
        per_category_include_extra = _per_category_include_extra_order(
            include_list, extra_list, ingredient_category_map, required_categories
        )

        categories_ignored_full_exclude: set = set()
        categories_full_exclude_conflict: set = set()
        _full_exclude_notices_added: set = set()

        def _refresh_full_exclude_category_sets():
            nonlocal categories_ignored_full_exclude, categories_full_exclude_conflict
            fully = _heybo_detect_full_category_excludes(
                category_to_ingredients, exclude_list
            )
            categories_ignored_full_exclude, categories_full_exclude_conflict = (
                _heybo_apply_full_exclude_category_policy(
                    fully, category_limits, category_limits_customization
                )
            )

        def _sync_full_exclude_global_state():
            """Expose full-exclude outcome for user_message (omit vs required-min conflict)."""
            global_validations["full_exclude_omit_categories"] = sorted(
                categories_ignored_full_exclude
            )
            global_validations["full_exclude_conflict_categories"] = sorted(
                categories_full_exclude_conflict
            )
            covered = set()
            for cat in categories_ignored_full_exclude:
                covered.update(category_to_ingredients.get(cat) or [])
            global_validations["full_exclude_covered_ingredient_names"] = sorted(covered)
            conflict_covered = set()
            for cat in categories_full_exclude_conflict:
                conflict_covered.update(category_to_ingredients.get(cat) or [])
            global_validations["full_exclude_conflict_ingredient_names"] = sorted(
                conflict_covered
            )

        def _append_full_exclude_notices():
            for cat in sorted(categories_ignored_full_exclude):
                note = (
                    f"You excluded all available {cat.lower()} options — "
                    f"{cat.lower()} omitted from your bowls"
                )
                if note not in _full_exclude_notices_added:
                    _full_exclude_notices_added.add(note)
                    global_validations["category_limit_notices"].append(note)
                    global_validations["filter_summary"].append(
                        f"Full exclude: {cat} omitted (customization min 0)"
                    )
            for cat in sorted(categories_full_exclude_conflict):
                cmin_disp = _heybo_full_exclude_conflict_min(
                    cat, category_limits, category_limits_customization
                )
                note = _heybo_full_exclude_conflict_user_notice(cat, cmin_disp)
                req = _heybo_full_exclude_conflict_requirement_phrase(cat, cmin_disp)
                if note not in _full_exclude_notices_added:
                    _full_exclude_notices_added.add(note)
                    global_validations["category_limit_notices"].append(note)
                global_validations["filter_summary"].append(
                    f"Full exclude conflict: {cat} filled from catalog despite Exclude ({req})"
                )

        _refresh_full_exclude_category_sets()
        _sync_full_exclude_global_state()
        _append_full_exclude_notices()

        nutrient_filters_all = user_input.get("NutrientFilters", [])
        # Min==Max point targets (e.g. protein 60–60) → treat as "at least Min" with soft Max.
        _eq_notices = open_heybo_equal_nutrient_point_targets(nutrient_filters_all)
        for _n in _eq_notices:
            global_validations["filter_summary"].append(_n)
            print(f"[CONSTRAINT-FIRST] {_n}")
        user_input["NutrientFilters"] = nutrient_filters_all
        _active_nutrient_keys = heybo_active_nutrient_filter_keys(nutrient_filters_all)
        global_validations["nutrient_relaxation_active_keys"] = _active_nutrient_keys
        user_input["nutrient_relaxation_active_keys"] = _active_nutrient_keys
        numeric_nutrient_mode = _heybo_has_active_numeric_nutrient_filters(
            nutrient_filters_all
        )
        numeric_nutrient_filters = _heybo_effective_numeric_nutrient_filters(
            nutrient_filters_all
        )
        # True only when at least one nutrient filter has a binding Min (e.g. High Protein).
        # In that case, the customization-stage expansion (more ingredient slots) can help
        # reach the minimum.  When all filters are Max-only (e.g. Low Calorie), expansion
        # adds more ingredients → more nutrients → harder to stay under the cap, and higher price.
        _numeric_expand_for_min = _heybo_nutrient_filters_need_more(numeric_nutrient_filters)
        # Binding Mins (e.g. protein ≥ 60): detect from effective OR raw NutrientFilters.
        # Do not gate on numeric_nutrient_mode — messaging already sees Mins when filters exist.
        _binding_min_targets = _heybo_binding_min_nutrient_targets(
            list(numeric_nutrient_filters or []) or list(nutrient_filters_all or [])
        )
        if not _binding_min_targets and nutrient_filters_all:
            # Fallback: accept Range keys Min/min and Nutrient labels loosely.
            from .diet import _resolve_nutrient_filter_key as _rkey_fallback

            for _nf in nutrient_filters_all:
                if not isinstance(_nf, dict):
                    continue
                _rng = _nf.get("Range") or {}
                if not isinstance(_rng, dict):
                    continue
                _raw_min = _rng.get("Min", _rng.get("min"))
                if _raw_min is None:
                    continue
                try:
                    _mn = float(_raw_min)
                except (TypeError, ValueError):
                    continue
                if _mn <= 0:
                    continue
                _key = _rkey_fallback(str(_nf.get("Nutrient") or "").strip())
                if _key:
                    _binding_min_targets[_key] = _mn
        _chase_binding_mins = _heybo_should_chase_binding_mins(
            user_input,
            _binding_min_targets,
            list(numeric_nutrient_filters or []) or list(nutrient_filters_all or []),
        )
        NUTRIENT_NUMERIC_NORMAL_LIMIT_ATTEMPTS = 20
        constraint_first_mode = _heybo_hard_constraint_mode(
            user_input,
            numeric_nutrient_mode or _chase_binding_mins,
            nutrient_filters=list(numeric_nutrient_filters or [])
            or list(nutrient_filters_all or []),
        )
        print(
            f"[CONSTRAINT-FIRST] mode={constraint_first_mode} chase_mins={_chase_binding_mins} "
            f"targets={_binding_min_targets} numeric_mode={numeric_nutrient_mode} "
            f"nf={len(numeric_nutrient_filters or [])} all_nf={len(nutrient_filters_all or [])}"
        )
        if constraint_first_mode:
            global_validations["filter_summary"].append(
                "Constraint-first mode: use most of the attempt budget to build bowls that "
                "jointly satisfy active price/nutrient filters; relax only in the last "
                f"{int((1 - JOINT_CONSTRAINT_RELAX_AFTER_ATTEMPT_FRACTION) * 100)}% of attempts "
                "if no joint match; variety only when multiple feasible proteins exist"
            )
            if _chase_binding_mins:
                global_validations["filter_summary"].append(
                    "Constraint-first Min chase: first "
                    f"{NUTRIENT_NUMERIC_NORMAL_LIMIT_ATTEMPTS} attempts use normal slot "
                    "limits; if Mins still miss, expand to normal min + customization max "
                    f"(targets: {', '.join(f'{k}>={v:g}' for k, v in _binding_min_targets.items())})"
                )
        if numeric_nutrient_mode:
            global_validations["filter_summary"].append(
                "Numeric nutrient filters detected — first 20 attempts use normal limits; "
                "then normal minimums + customization maximums"
            )
            _priority_order = (
                load_heybo_nutrient_priority_order_from_db()
                or HEYBO_NUTRIENT_RELAXATION_PRIORITY_ORDER
            )
            global_validations["filter_summary"].append(
                "Nutrient relaxation uses priority order "
                f"({', '.join(_priority_order[:4])}, …) — "
                "lowest-priority active nutrient relaxes first"
            )

        # Pre-assign closure cells so _effective_limit_for_cat can be called safely
        # before the main generation loop (e.g. category_limit_notices).
        strict_only_bowl = only_mode
        variety_context_active = False

        def _effective_limit_for_cat(cat, attempt_no=None):
            if cat not in category_limits:
                return None
            if cat in categories_ignored_full_exclude:
                return (0, 0)
            if cat in categories_full_exclude_conflict:
                normal_min, normal_max = category_limits[cat]
                cmin, _ = category_limits_customization.get(cat, (normal_min, normal_max))
                try:
                    cmin_int = int(cmin)
                except (TypeError, ValueError):
                    cmin_int = int(normal_min)
                return (cmin_int, int(normal_max))
            # Only-mode bowls 2–5: normal menu limits. Spread user Include/Extra
            # within those caps and fill remaining slots from the catalog — do not
            # reuse bowl-1 customization / extras ceilings (avoids 4–5 proteins).
            if variety_context_active:
                return category_limits[cat]
            if strict_only_bowl:
                return category_limits_customization.get(cat, category_limits.get(cat))
            normal_min, normal_max = category_limits[cat]
            if (
                numeric_nutrient_mode
                and isinstance(attempt_no, int)
                and attempt_no > NUTRIENT_NUMERIC_NORMAL_LIMIT_ATTEMPTS
            ):
                # Staged search: first N attempts use normal slot limits so we prefer
                # ordinary ingredient combos. Only then expand to customization max when
                # Mins need more slots (High Protein, etc.). Max-only filters stay on
                # normal limits so we don't add items that push calories/price up.
                # Exception: user-explicit extras/customization counts always honoured.
                if (
                    not _numeric_expand_for_min
                    and not _chase_binding_mins
                    and (
                        cat not in categories_with_customization
                        and cat not in categories_with_extras
                    )
                ):
                    return (normal_min, normal_max)
                _cmin, _cmax = category_limits_customization.get(
                    cat, (normal_min, normal_max)
                )
                return (normal_min, _cmax)
            mn, mx = normal_min, normal_max
            if cat in categories_with_customization:
                # Overflow: customization max is a ceiling for user asks only.
                return _heybo_ask_aware_category_limit(
                    cat,
                    mn,
                    mx,
                    category_limits_customization,
                    requested_count_by_category,
                    use_high_min=True,
                )
            if cat in categories_with_extras:
                # Extra requested: raise ceiling only as needed for Extra portions —
                # do not open full customization max as a random fill target.
                return _heybo_ask_aware_category_limit(
                    cat,
                    mn,
                    mx,
                    category_limits_extras,
                    requested_count_by_category,
                    use_high_min=False,
                )
            return (mn, mx)

        # message_to_user: mirror salad.process_validation_messages_for_user (~9918–9987).
        # Heybo bowls do not set bowl_specific.category_limits, so salad never emits these.
        # category_limit_notices are joined under " - Note: " in user_message._append_category_limit_notices
        # — do not prefix "Note: " on each line.
        if categories_with_customization and not only_mode:
            for cat in sorted(categories_with_customization):
                lim = _effective_limit_for_cat(cat)
                if not lim or len(lim) < 2:
                    continue
                mx = lim[1]
                req_total = requested_count_by_category.get(cat, 0)
                if req_total > mx:
                    if mx == 1:
                        note = (
                            f"You requested {req_total} {cat.lower()} ingredients, but only 1 can be "
                            f"included per bowl. Each bowl includes one of your requested options."
                        )
                    else:
                        note = (
                            f"You requested {req_total} {cat.lower()} ingredients, but only {mx} can be "
                            f"included per bowl. Each bowl includes up to {mx} of your requested options."
                        )
                    if note not in global_validations["category_limit_notices"]:
                        global_validations["category_limit_notices"].append(note)

        def _balanced_category_limit_violations(bowl_dict):
            violations = []
            skip_keys = {
                "Extra Proteins",
                "Extra Warm sides",
                "Extra Cold sides",
                "PriceBreakdown",
            }
            for cat, items in bowl_dict.items():
                if cat in (
                    "message", "Validations", "Shop Name", "Bowl Type", "Bowl Name",
                    "Total Cost", "Total Nutrients", "Total Weight", "Total_CO2e_g",
                ):
                    continue
                if cat in skip_keys:
                    continue
                lim = _effective_limit_for_cat(cat, attempts)
                if not lim or len(lim) < 2:
                    continue
                max_count = lim[1]
                if cat in ("Proteins", "Warm sides", "Cold sides"):
                    count = _heybo_combined_count_for_category(bowl_dict, cat)
                else:
                    if not isinstance(items, list):
                        continue
                    count = len(items)
                if count > max_count:
                    violations.append(f"{cat}: {count} > {max_count}")
            return violations

        def _apply_combined_category_caps(bowl_dict):
            """Enforce ingredient_wise_count max on main + Extra * combined (customization parity)."""
            for nk, ek, c in (
                ("Proteins", "Extra Proteins", "Proteins"),
                ("Warm sides", "Extra Warm sides", "Warm sides"),
                ("Cold sides", "Extra Cold sides", "Cold sides"),
            ):
                lim = _effective_limit_for_cat(c, attempts)
                if lim and len(lim) >= 2:
                    _cap_heybo_main_extra_pair(bowl_dict, nk, ek, lim[1])

        target_bowl_count = HEYBO_BOWLS_PER_PAGE
        variety_context_active = False
        strict_only_bowl = only_mode
        heybo_bowls = []
        used_protein_sauce_combos = set(heybo_cfg.get("previous_protein_sauce_combos") or set())
        primary_combo_fallback_logged = False
        lh_small_pool_dedup_logged = False
        max_attempts = 200
        attempts = 0
        nutrient_relaxation_levels: dict = {}
        nutrient_relaxation_level = 0
        failed_prep_attempts = 0
        failed_cuisine_attempts = 0
        failed_flavor_attempts = 0
        failed_nutrient_attempts = 0
        failed_price_attempts = 0
        # Binding Mins missed while bowl still fits price → price is the real blocker
        # (need Extra protein / more slots). Count these so price relaxes before nutrient.
        failed_min_under_budget_attempts = 0
        price_relaxation_level = 0
        failed_light_hearty_attempts = 0
        light_hearty_relaxation_level = user_input.get("light_hearty_relaxation_level", 0)
        if not isinstance(light_hearty_relaxation_level, int) or light_hearty_relaxation_level < 0:
            light_hearty_relaxation_level = 0
        if light_hearty_relaxation_level > 5:
            light_hearty_relaxation_level = 5
        user_input["light_hearty_relaxation_level"] = light_hearty_relaxation_level
        _lh_overlap = light_hearty_criteria_nutrient_overlap(
            list(nutrient_filters_all or [])
        )
        _lh_flag_on = _is_truthy_flag(user_input.get("Light")) or _is_truthy_flag(
            user_input.get("Hearty")
        )
        _lh_suppressed = bool(_lh_overlap) and _lh_flag_on
        if _lh_suppressed:
            user_input["_light_hearty_suppressed_by_nutrients"] = True
            user_input["_light_hearty_suppressed_nutrient_keys"] = list(_lh_overlap)
            if not any(
                isinstance(x, str) and "Light/Hearty ignored" in x
                for x in (global_validations.get("light_hearty_matches") or [])
            ):
                global_validations.setdefault("light_hearty_matches", []).append(
                    "Light/Hearty ignored — NutrientFilters already set "
                    + ", ".join(_lh_overlap)
                    + " (same nutrients as Light/Hearty criteria)"
                )
        _lh_active = _lh_flag_on and not _lh_suppressed
        failed_balanced_attempts = 0
        balanced_relaxation_level = user_input.get("balanced_relaxation_level", 0)
        if not isinstance(balanced_relaxation_level, int) or balanced_relaxation_level < 0:
            balanced_relaxation_level = 0
        if balanced_relaxation_level > 5:
            balanced_relaxation_level = 5
        user_input["balanced_relaxation_level"] = balanced_relaxation_level
        failed_co2_attempts = 0
        co2_relaxation_level = user_input.get("co2_relaxation_level", 0)
        if not isinstance(co2_relaxation_level, int) or co2_relaxation_level < 0:
            co2_relaxation_level = 0
        if co2_relaxation_level > CO2_MAX_RELAXATION_LEVEL:
            co2_relaxation_level = CO2_MAX_RELAXATION_LEVEL
        user_input["co2_relaxation_level"] = co2_relaxation_level
        prep_relaxation_enabled = bool(user_input.get("_relax_preparation_method_filter"))
        cuisine_relaxation_enabled = bool(user_input.get("_relax_cuisine_filter"))
        flavor_relaxation_enabled = bool(user_input.get("_relax_flavor_filter"))
        # Nutrient misses while prep/cuisine/flavor still prefer a thin matched pool.
        # Do not reset these when a preferred-only bowl "matches" that filter.
        failed_prep_pool_attempts = 0
        failed_cuisine_pool_attempts = 0
        failed_flavor_pool_attempts = 0
        _page_n = user_input.get("Page", 1)
        _rp = (
            (user_input.get("recommend_page_id") or user_input.get("RecommendPageId") or user_input.get("recommendation_page_id") or "")
            .strip()
        )
        _sid = (user_input.get("session_id") or user_input.get("SessionId") or "").strip()
        _pag_parts = [
            f"Page {_page_n}",
            (
                f"up to {HEYBO_BOWLS_PER_PAGE} bowls per request"
                if not only_mode
                else f"Only mode: 1 exact-ingredient bowl + {HEYBO_BOWLS_PER_PAGE - 1} variety bowls"
            ),
        ]
        def _activate_heybo_variety_context():
            nonlocal variety_context_active, only_allowed_names, filtered_ingredients
            nonlocal fallback_categories, category_to_ingredients, bases, proteins
            nonlocal warm_sides, cold_sides, dips, garnishes, sauces
            nonlocal extra_proteins, extra_warm_sides, extra_cold_sides
            nonlocal minimum_fill_pools, _strict_safety_df, categories_with_customization
            nonlocal categories_with_extras
            nonlocal categories_ignored_full_exclude, categories_full_exclude_conflict
            nonlocal nutrient_filters_all, numeric_nutrient_filters, numeric_nutrient_mode
            nonlocal prep_active, cuisine_active, diet_active, allergen_active, flavor_active
            nonlocal is_balanced_requested, sustainable_on, user_input
            nonlocal nutrient_relaxation_levels, failed_nutrient_attempts
            nonlocal failed_light_hearty_attempts, light_hearty_relaxation_level
            nonlocal failed_balanced_attempts, balanced_relaxation_level
            nonlocal failed_co2_attempts, co2_relaxation_level
            nonlocal failed_price_attempts, price_relaxation_level
            nonlocal failed_min_under_budget_attempts
            nonlocal failed_prep_pool_attempts, failed_cuisine_pool_attempts, failed_flavor_pool_attempts
            nonlocal prep_relaxation_enabled, cuisine_relaxation_enabled, flavor_relaxation_enabled
            nonlocal _numeric_expand_for_min, _chase_binding_mins, _binding_min_targets
            nonlocal constraint_first_mode
            if variety_context_active:
                return
            snap = user_input.get("_heybo_full_user_input_snapshot")
            if not isinstance(snap, dict):
                variety_context_active = True
                return
            for key in (
                "DietFilters",
                "NutrientFilters",
                "CuisineFilters",
                "FlavorPreferences",
                "Price",
                "PreparationMethod",
                "AllergenFilters",
                "Light",
                "Hearty",
                "Sustainable",
                "Balanced",
                "Signatures",
                "_heybo_message_diet_filters_dict",
                "_relax_preparation_method_filter",
                "_relax_cuisine_filter",
                "_relax_flavor_filter",
                "_cuisine_matched_ingredient_names",
                "_heybo_true_cuisine_by_category",
                "_heybo_true_cuisine_names",
                "_heybo_true_flavor_by_category",
                "_heybo_true_prep_by_category",
            ):
                if key in snap:
                    user_input[key] = copy.deepcopy(snap[key])
            user_input.pop("_heybo_only_mode_generation", None)
            user_input["_heybo_variety_bowls_active"] = True
            for _msg in user_input.pop("_nutrient_preprocess_messages", []):
                global_validations["diet_fallbacks"].append(_msg)
            _pre_ga_var = user_input.pop("_heybo_guideline_adjustments_from_preprocess", None)
            if _pre_ga_var:
                global_validations["guideline_adjustments"].extend(_pre_ga_var)
            user_input = preprocess_heybo_filters(user_input)
            sustainable_on = _is_truthy_flag(user_input.get("Sustainable"))
            normalize_heybo_price_filters(
                user_input, heybo_cfg["price_config"], global_validations
            )
            is_balanced_requested = _heybo_apply_balanced_override(
                user_input, global_validations
            )
            prep_active = bool(
                isinstance(user_input.get("PreparationMethod"), dict)
                and any(v is True for v in user_input.get("PreparationMethod", {}).values())
            )
            cuisine_filters_val = user_input.get("CuisineFilters")
            cuisine_active = (
                (isinstance(cuisine_filters_val, list) and len(cuisine_filters_val) > 0)
                or (
                    isinstance(cuisine_filters_val, dict)
                    and any(v is True for v in cuisine_filters_val.values())
                )
            )
            diet_active = bool(user_input.get("DietFilters"))
            allergen_active = bool(user_input.get("AllergenFilters"))
            flavor_active = bool(
                isinstance(user_input.get("FlavorPreferences"), dict)
                and any(v for v in user_input.get("FlavorPreferences", {}).values())
            )
            (
                filtered_ingredients,
                fallback_categories,
                filter_error,
                category_to_ingredients,
                bases,
                proteins,
                warm_sides,
                cold_sides,
                dips,
                garnishes,
                sauces,
                extra_proteins,
                extra_warm_sides,
                extra_cold_sides,
            ) = rebuild_filtered_context()
            if filter_error:
                global_validations["filter_summary"].append(
                    f"Only mode: variety bowls skipped — {filter_error}"
                )
                user_input["_heybo_variety_bowls_unavailable"] = True
                variety_context_active = True
                return
            _strict_safety_df = build_heybo_strict_safety_catalog(
                df_heybo, user_input, global_validations
            )
            minimum_fill_pools = minimum_fill_pools_by_category(_strict_safety_df)
            # Bowls 2–5 are variety under normal limits — do not carry bowl-1
            # customization / extras ceilings (those exist only to fit exact Include+Extra).
            categories_with_customization = set()
            categories_with_extras = set()
            only_allowed_names = set()
            nutrient_filters_all = user_input.get("NutrientFilters", [])
            _eq_notices = open_heybo_equal_nutrient_point_targets(nutrient_filters_all)
            for _n in _eq_notices:
                global_validations["filter_summary"].append(_n)
            user_input["NutrientFilters"] = nutrient_filters_all
            _active_nutrient_keys = heybo_active_nutrient_filter_keys(nutrient_filters_all)
            global_validations["nutrient_relaxation_active_keys"] = _active_nutrient_keys
            user_input["nutrient_relaxation_active_keys"] = _active_nutrient_keys
            numeric_nutrient_mode = _heybo_has_active_numeric_nutrient_filters(
                nutrient_filters_all
            )
            numeric_nutrient_filters = _heybo_effective_numeric_nutrient_filters(
                nutrient_filters_all
            )
            _numeric_expand_for_min = _heybo_nutrient_filters_need_more(numeric_nutrient_filters)
            _binding_min_targets = _heybo_binding_min_nutrient_targets(
                numeric_nutrient_filters if numeric_nutrient_mode else []
            )
            _chase_binding_mins = _heybo_should_chase_binding_mins(
                user_input,
                _binding_min_targets,
                numeric_nutrient_filters if numeric_nutrient_mode else [],
            )
            constraint_first_mode = _heybo_hard_constraint_mode(
                user_input,
                numeric_nutrient_mode or _chase_binding_mins,
                nutrient_filters=numeric_nutrient_filters
                if numeric_nutrient_mode
                else [],
            )
            if numeric_nutrient_mode:
                global_validations["filter_summary"].append(
                    "Only mode variety bowls: numeric nutrient filters apply to bowls 2–5"
                )
            if include_list and not global_validations.get("apriori_suggestions"):
                try:
                    global_validations["apriori_suggestions"] = _heybo_load_apriori_suggestions(
                        include_list, df_heybo
                    )
                except Exception as e:
                    print(f"Apriori enhancement skipped for variety bowls: {e}")
                    global_validations["apriori_suggestions"] = []
            _refresh_full_exclude_category_sets()
            _sync_full_exclude_global_state()
            _append_full_exclude_notices()
            global_validations["filter_summary"].append(
                "Only mode: switched to full filters for variety bowls (2–5) "
                "using normal category limits (spread Include/Extra + catalog variety)"
            )
            # Reset all relaxation counters so any failures during bowl 1 assembly
            # (which bypasses the filter gates) do not carry over into variety bowls.
            # Each variety bowl evaluates filters from scratch at relaxation level 0.
            nutrient_relaxation_levels = {}
            failed_nutrient_attempts = 0
            failed_light_hearty_attempts = 0
            light_hearty_relaxation_level = 0
            user_input["light_hearty_relaxation_level"] = 0
            failed_balanced_attempts = 0
            balanced_relaxation_level = 0
            user_input["balanced_relaxation_level"] = 0
            failed_co2_attempts = 0
            co2_relaxation_level = 0
            user_input["co2_relaxation_level"] = 0
            failed_price_attempts = 0
            failed_min_under_budget_attempts = 0
            price_relaxation_level = 0
            failed_prep_pool_attempts = 0
            failed_cuisine_pool_attempts = 0
            failed_flavor_pool_attempts = 0
            prep_relaxation_enabled = bool(user_input.get("_relax_preparation_method_filter"))
            cuisine_relaxation_enabled = bool(user_input.get("_relax_cuisine_filter"))
            flavor_relaxation_enabled = bool(user_input.get("_relax_flavor_filter"))
            variety_context_active = True

        if _rp or _sid:
            _pag_parts.append(
                f"prior meals for dedupe: {len(previous_bowls)} ingredient sets, "
                f"{len(used_protein_sauce_combos)} primary bowl keys from bowl_name (from DB)"
            )
            if heybo_cfg.get("_heybo_meal_dedupe_merged_session_orphans"):
                _pag_parts.append(
                    "merged same-session meals with blank recommend_page_id (page 1 before id assigned)"
                )
        else:
            _pag_parts.append("no recommend_page_id/session_id — pagination dedupe state empty")
        global_validations["filter_summary"].append("; ".join(_pag_parts))
        # Nutrient lookup cache — reused across all attempts (values are stable per ingredient).
        _ingredient_nutrient_cache: dict = {}
        # Feasible proteins under joint price+nutrient optimism (conditional variety).
        _price_headroom0 = _heybo_price_ingredient_headroom(
            user_input, heybo_cfg.get("price_config") or {}, price_relaxation_level=0
        )
        _wm0, _wmax0 = category_limits.get("Warm sides", (0, 0))
        _cm0, _cmax0 = category_limits.get("Cold sides", (0, 0))
        try:
            _wmax0 = int(_wmax0 or 0)
            _cmax0 = int(_cmax0 or 0)
        except (TypeError, ValueError):
            _wmax0, _cmax0 = 0, 0
        # Match joint-fill tight-price caps so "feasible" proteins are ones that can
        # actually hit Mins without Extra * overflow when Max ≈ BYB_Min_Price.
        _price_cfg0 = heybo_cfg.get("price_config") or {}
        _pricing_limits0 = _price_cfg0.get("pricing_limits") or {}
        _tier_sides0 = int(_pricing_limits0.get("Sides", (0, 3))[1] or 3)
        _feas_wmax, _feas_cmax = _wmax0, _cmax0
        if _price_headroom0 is not None and float(_price_headroom0) < 1.0:
            _feas_cmax = min(_cmax0, _tier_sides0)
            _feas_wmax = min(_wmax0, max(0, _tier_sides0 - _feas_cmax))
        feasible_proteins_for_constraints = []
        _constraint_variety_relaxed = False

        def _refresh_feasible_proteins():
            nonlocal feasible_proteins_for_constraints, _constraint_variety_relaxed
            feasible_proteins_for_constraints = (
                _heybo_estimate_feasible_proteins(
                    list(dict.fromkeys(list(proteins or []) + list(extra_proteins or []))),
                    df=df_heybo,
                    cache=_ingredient_nutrient_cache,
                    numeric_nutrient_filters=numeric_nutrient_filters if numeric_nutrient_mode else [],
                    remaining_headroom=_price_headroom0,
                    warm_pool=list(warm_sides),
                    cold_pool=list(cold_sides),
                    warm_max=_feas_wmax,
                    cold_max=_feas_cmax,
                    base_pool=list(bases),
                    dip_pool=list(dips),
                    garnish_pool=list(garnishes),
                    sauce_pool=list(sauces),
                    protein_max=int(_pricing_limits0.get("Proteins", (0, 1))[1] or 1),
                )
                if constraint_first_mode and numeric_nutrient_mode
                else list(dict.fromkeys(list(proteins or []) + list(extra_proteins or [])))
            )
            _constraint_variety_relaxed = (
                constraint_first_mode and len(feasible_proteins_for_constraints) < 2
            )

        def _rebuild_after_soft_filter_relax():
            nonlocal filtered_ingredients, fallback_categories, category_to_ingredients
            nonlocal bases, proteins, warm_sides, cold_sides, dips, garnishes, sauces
            nonlocal extra_proteins, extra_warm_sides, extra_cold_sides
            (
                filtered_ingredients,
                fallback_categories,
                filter_error,
                category_to_ingredients,
                bases,
                proteins,
                warm_sides,
                cold_sides,
                dips,
                garnishes,
                sauces,
                extra_proteins,
                extra_warm_sides,
                extra_cold_sides,
            ) = rebuild_filtered_context()
            if filter_error:
                return {
                    "error": filter_error,
                    "message_to_user": build_heybo_message_to_user(
                        [], None, user_input=user_input, error=filter_error
                    ),
                }
            _refresh_feasible_proteins()
            return None

        def _relax_prep_pool(reason: str):
            """Widen prep pool from allergen/diet-safe catalog. Never bypasses diet or allergens."""
            nonlocal prep_relaxation_enabled, failed_prep_attempts, failed_prep_pool_attempts
            if prep_relaxation_enabled or user_input.get("_relax_preparation_method_filter"):
                return None
            user_input["_relax_preparation_method_filter"] = True
            user_input["prep_relaxation_enabled"] = True
            prep_relaxation_enabled = True
            failed_prep_attempts = 0
            failed_prep_pool_attempts = 0
            global_validations.setdefault("preparation_method_matches", []).append(reason)
            global_validations.setdefault("filter_summary", []).append(reason)
            print(f"[CONSTRAINT-FIRST] {reason}")
            return _rebuild_after_soft_filter_relax()

        def _relax_cuisine_pool(reason: str):
            """Stop preferring cuisine-only picks. Never bypasses diet or allergens."""
            nonlocal cuisine_relaxation_enabled, failed_cuisine_attempts, failed_cuisine_pool_attempts
            if cuisine_relaxation_enabled or user_input.get("_relax_cuisine_filter"):
                return None
            user_input["_relax_cuisine_filter"] = True
            user_input["cuisine_relaxation_enabled"] = True
            cuisine_relaxation_enabled = True
            failed_cuisine_attempts = 0
            failed_cuisine_pool_attempts = 0
            global_validations.setdefault("cuisine_matches", []).append(reason)
            global_validations.setdefault("filter_summary", []).append(reason)
            print(f"[CONSTRAINT-FIRST] {reason}")
            return _rebuild_after_soft_filter_relax()

        def _relax_flavor_pool(reason: str):
            """Add pre-flavor (allergen/diet-safe) ingredients. Never bypasses diet or allergens."""
            nonlocal flavor_relaxation_enabled, failed_flavor_attempts, failed_flavor_pool_attempts
            if flavor_relaxation_enabled or user_input.get("_relax_flavor_filter"):
                return None
            user_input["_relax_flavor_filter"] = True
            user_input["flavor_relaxation_enabled"] = True
            flavor_relaxation_enabled = True
            failed_flavor_attempts = 0
            failed_flavor_pool_attempts = 0
            global_validations.setdefault("flavor_adjustments", []).append(reason)
            global_validations.setdefault("filter_summary", []).append(reason)
            print(f"[CONSTRAINT-FIRST] {reason}")
            return _rebuild_after_soft_filter_relax()

        _refresh_feasible_proteins()
        cpsat_state = CpsatFillState()

        def _heybo_stamp_cpsat_bowls():
            """Names/images/validations for solver bowls (random path already stamps)."""
            for bowl in heybo_bowls:
                if not bowl.get("_from_cpsat"):
                    continue
                if not bowl.get("Validations"):
                    bowl["Validations"] = {
                        "global": global_validations,
                        "bowl_specific": {
                            "preparation_method_matches": [],
                            "cuisine_matches": [],
                            "ingredient_explanations": [],
                            "diet_compatibility": [],
                            "allergen_safety": [],
                            "nutritional_targets": [],
                            "bowl_composition": ["Generated via CP-SAT"],
                            "compatibility_status": "all ingredients are compatible",
                            "price": [],
                            "flavor_adjustments": [],
                            "light_hearty_matches": [],
                            "balanced_diet_matches": [],
                            "apriori_matches": [],
                        },
                    }
                else:
                    bowl["Validations"]["global"] = global_validations
                if not bowl.get("image_details"):
                    bowl["image_details"] = build_heybo_dynamic_grid_image_details(
                        bowl, image_url_by_name
                    )
                pop_pricing_metadata(bowl)
                if not bowl.get("message"):
                    bowl["message"] = extract_heybo_user_warnings(bowl["Validations"])
                bowl.pop("_from_cpsat", None)

        def _heybo_try_cpsat_fill(reason: str = "initial") -> int:
            added = run_heybo_cpsat_fill(
                reason,
                state=cpsat_state,
                enabled=CPSAT_ENABLED,
                original_only_mode=only_mode,
                variety_context_active=variety_context_active,
                target_bowls=target_bowl_count,
                heybo_bowls=heybo_bowls,
                user_input=user_input,
                df=df_heybo,
                heybo_cfg=heybo_cfg,
                global_validations=global_validations,
                price_relaxation_level=price_relaxation_level,
                co2_relaxation_level=co2_relaxation_level,
                light_hearty_relaxation_level=light_hearty_relaxation_level,
                nutrient_relaxation_level=heybo_max_nutrient_relaxation_level(
                    nutrient_relaxation_levels
                ),
                nutrient_relaxation_levels=nutrient_relaxation_levels,
                nutrient_filters_all=nutrient_filters_all,
                numeric_nutrient_filters=numeric_nutrient_filters,
                cuisine_relaxation_enabled=cuisine_relaxation_enabled,
                flavor_relaxation_enabled=flavor_relaxation_enabled,
                prep_relaxation_enabled=prep_relaxation_enabled,
                numeric_expand_for_min=_numeric_expand_for_min,
                chase_binding_mins=_chase_binding_mins,
                active_nutrient_keys=_active_nutrient_keys,
                bases=bases,
                proteins=proteins,
                extra_proteins=extra_proteins,
                warm_sides=warm_sides,
                extra_warm_sides=extra_warm_sides,
                cold_sides=cold_sides,
                extra_cold_sides=extra_cold_sides,
                dips=dips,
                garnishes=garnishes,
                sauces=sauces,
                include_list=include_list,
                extra_list=extra_list,
                exclude_list=exclude_list,
                omit_categories=categories_ignored_full_exclude,
                normal_flow_category_limits=category_limits,
                customization_flow_extra_category_max=category_limits_customization,
                incompatible_pairs=incompatible_pairs,
                filtered_ingredients=filtered_ingredients,
                flavor_preferences=user_input.get("FlavorPreferences") or {},
                previous_bowls=previous_bowls,
                used_protein_sauce_combos=used_protein_sauce_combos,
                customization_category_limits=category_limits_customization,
            )
            if added:
                _heybo_stamp_cpsat_bowls()
            return added

        _heybo_try_cpsat_fill("initial")

        if constraint_first_mode and numeric_nutrient_mode:
            global_validations["filter_summary"].append(
                f"Constraint-first: {len(feasible_proteins_for_constraints)} feasible protein(s) "
                f"under joint nutrient"
                + ("+price" if _price_headroom0 is not None else "")
                + " optimism"
                + (
                    " — protein reuse allowed (variety deprioritized)"
                    if _constraint_variety_relaxed
                    else " — enforce protein variety when possible"
                )
            )
        failed_joint_constraint_attempts = 0
        while len(heybo_bowls) < target_bowl_count and attempts < max_attempts:
            if only_mode and len(heybo_bowls) >= 1:
                _was_variety = variety_context_active
                _activate_heybo_variety_context()
                if not _was_variety and variety_context_active:
                    _heybo_try_cpsat_fill("only_variety")
                    if len(heybo_bowls) >= target_bowl_count:
                        break
            if only_mode and user_input.get("_heybo_variety_bowls_unavailable"):
                target_bowl_count = max(1, len(heybo_bowls))
            strict_only_bowl = only_mode and not variety_context_active
            attempts += 1
            numeric_customization_stage = (
                numeric_nutrient_mode and attempts > NUTRIENT_NUMERIC_NORMAL_LIMIT_ATTEMPTS
            )
            _attempt_headroom = _heybo_price_ingredient_headroom(
                user_input,
                heybo_cfg.get("price_config") or {},
                price_relaxation_level=price_relaxation_level,
            )
            proteins_for_attempt, sauces_for_attempt = prefer_unused_protein_sauce_combos(
                list(dict.fromkeys(proteins)),
                list(dict.fromkeys(sauces)),
                used_protein_sauce_combos,
                include_list,
                extra_list,
            )
            if constraint_first_mode and feasible_proteins_for_constraints:
                _fp_attempt = [
                    p for p in feasible_proteins_for_constraints if p in set(proteins_for_attempt)
                ]
                if _fp_attempt:
                    if not _constraint_variety_relaxed:
                        _used_prots = {
                            str(k).split("_", 1)[0]
                            for k in used_protein_sauce_combos
                            if k and "_" in str(k)
                        }
                        _unused_fp = [p for p in _fp_attempt if p not in _used_prots]
                        proteins_for_attempt = (_unused_fp or _fp_attempt) + [
                            p for p in proteins_for_attempt if p not in _fp_attempt
                        ]
                    else:
                        proteins_for_attempt = _fp_attempt + [
                            p for p in proteins_for_attempt if p not in _fp_attempt
                        ]
            attempt_catalog = _heybo_build_generation_catalog_pools(
                bases=bases,
                proteins=proteins,
                extra_proteins=extra_proteins,
                warm_sides=warm_sides,
                cold_sides=cold_sides,
                extra_warm_sides=extra_warm_sides,
                extra_cold_sides=extra_cold_sides,
                dips=dips,
                garnishes=garnishes,
                sauces=sauces,
                sauces_for_attempt=sauces_for_attempt,
                exclude_list=exclude_list,
                minimum_fill_pools=minimum_fill_pools,
                category_limits=category_limits,
                only_mode=strict_only_bowl,
                exclude_ignore_categories=categories_full_exclude_conflict,
            )
            user_included_for_compat = set(include_list + extra_list)
            bowl = {
                'Shop Name': user_input.get('ShopName'),
                'Bowl Type': user_input.get('Bowl Type', 'bowl'),
                'Bases': [], 'Proteins': [], 'Extra Proteins': [],
                'Warm sides': [], 'Extra Warm sides': [],
                'Cold sides': [], 'Extra Cold sides': [],
                'Dips': [], 'Garnish': [], 'Sauces': [],
                'Total Cost': [], 'Total Nutrients': {}
            }
            pd_min, pd_max = category_limits["Proteins"]
            if strict_only_bowl:
                pd_min, pd_max = category_limits_customization.get("Proteins", (pd_min, pd_max))
            elif numeric_customization_stage:
                _cmin, _cmax = category_limits_customization.get("Proteins", (pd_min, pd_max))
                pd_min, pd_max = pd_min, _cmax
            elif "Proteins" in categories_with_customization:
                pd_min, pd_max = _heybo_ask_aware_category_limit(
                    "Proteins",
                    pd_min,
                    pd_max,
                    category_limits_customization,
                    requested_count_by_category,
                    use_high_min=True,
                )
            elif "Proteins" in categories_with_extras:
                pd_min, pd_max = _heybo_ask_aware_category_limit(
                    "Proteins",
                    pd_min,
                    pd_max,
                    category_limits_extras,
                    requested_count_by_category,
                    use_high_min=False,
                )
            bowl_incomplete = False
            bowl_validations = {
                "preparation_method_matches": [], "cuisine_matches": [], "ingredient_explanations": [],
                "diet_compatibility": [], "allergen_safety": [], "nutritional_targets": [],
                "bowl_composition": [], "compatibility_status": "all ingredients are compatible",
                "price": [], "flavor_adjustments": [], "light_hearty_matches": [],
                "balanced_diet_matches": [], "apriori_matches": [],
            }
            # Per-attempt nutrient budget — reset each attempt so budget tracks this bowl only.
            _current_nf = numeric_nutrient_filters if numeric_nutrient_mode else nutrient_filters_all
            if constraint_first_mode and _chase_binding_mins and not _current_nf:
                _current_nf = nutrient_filters_all
            _nutrient_budget: dict = (
                _heybo_init_nutrient_budget(_current_nf, nutrient_relaxation_levels, df_heybo)
                if _current_nf and not strict_only_bowl
                else {}
            )
            # Joint-fill only for ambitious / Min-only chase (not Max-first bands).
            # Price does not flip this — unreachable price widens via RELAXATION_ORDER.
            _use_joint_fill = (
                _chase_binding_mins
                and not strict_only_bowl
                and len(heybo_bowls) < target_bowl_count
            )
            _joint_fill_applied = False
            if attempts == 1:
                print(
                    f"[CONSTRAINT-FIRST] attempt=1 use_joint_fill={_use_joint_fill} "
                    f"chase_mins={_chase_binding_mins} "
                    f"mode={constraint_first_mode} "
                    f"primary_targets={_binding_min_targets}"
                )
            if _use_joint_fill:
                _joint_limits = {}
                # Same catalog shape as normal fill: preference pools + Extra * rows,
                # Exclude honored (unless full-exclude conflict category).
                _jf_exc_ignore = set(categories_full_exclude_conflict or set())
                _joint_pools = {
                    "Bases": [
                        t
                        for t in (bases or [])
                        if t
                        not in _heybo_category_exclude_set(
                            "Bases", exclude_list, _jf_exc_ignore
                        )
                    ],
                    "Proteins": list(
                        dict.fromkeys(
                            [
                                t
                                for t in (proteins or []) + (extra_proteins or [])
                                if t
                                not in _heybo_category_exclude_set(
                                    "Proteins", exclude_list, _jf_exc_ignore
                                )
                            ]
                        )
                    ),
                    "Warm sides": list(
                        dict.fromkeys(
                            [
                                t
                                for t in (warm_sides or []) + (extra_warm_sides or [])
                                if t
                                not in _heybo_category_exclude_set(
                                    "Warm sides", exclude_list, _jf_exc_ignore
                                )
                            ]
                        )
                    ),
                    "Cold sides": list(
                        dict.fromkeys(
                            [
                                t
                                for t in (cold_sides or []) + (extra_cold_sides or [])
                                if t
                                not in _heybo_category_exclude_set(
                                    "Cold sides", exclude_list, _jf_exc_ignore
                                )
                            ]
                        )
                    ),
                    "Dips": [
                        t
                        for t in (dips or [])
                        if t
                        not in _heybo_category_exclude_set(
                            "Dips", exclude_list, _jf_exc_ignore
                        )
                    ],
                    "Garnish": [
                        t
                        for t in (garnishes or [])
                        if t
                        not in _heybo_category_exclude_set(
                            "Garnish", exclude_list, _jf_exc_ignore
                        )
                    ],
                    "Sauces": [
                        t
                        for t in (sauces or [])
                        if t
                        not in _heybo_category_exclude_set(
                            "Sauces", exclude_list, _jf_exc_ignore
                        )
                    ],
                }
                for _cat in REQUIRED_BOWL_CATEGORIES:
                    if _cat in (categories_ignored_full_exclude or set()):
                        _joint_limits[_cat] = (0, 0)
                        _joint_pools[_cat] = []
                        continue
                    _lim = _effective_limit_for_cat(_cat, attempts) or category_limits.get(
                        _cat, (0, 0)
                    )
                    _joint_limits[_cat] = _lim
                # Tight price (Max ≈ BYB_Min_Price): a 2nd protein overflows to Extra Proteins
                # and often adds ai_price, blowing the $11.90 cap. Known-feasible bowls use
                # 1 protein + max high-protein sides (all free / in-tier). Cap protein slots
                # to the Pricing default tier when headroom is small.
                _price_cfg = heybo_cfg.get("price_config") or {}
                _pricing_limits = _price_cfg.get("pricing_limits") or {}
                _tier_prot = int(_pricing_limits.get("Proteins", (0, 1))[1] or 1)
                _tier_sides = int(_pricing_limits.get("Sides", (0, 3))[1] or 3)
                # Once price has been widened for unreachable Mins, allow Extra protein
                # slots even if remaining headroom is still small this attempt.
                _tight_price = (
                    _attempt_headroom is not None
                    and float(_attempt_headroom) < 1.0
                    and price_relaxation_level == 0
                )
                if _tight_price and "Proteins" in _joint_limits:
                    # Floor-price / free-tier: stay within Pricing default protein slots.
                    _pmin, _pmax = _joint_limits["Proteins"]
                    _joint_limits["Proteins"] = (_pmin, min(int(_pmax), max(1, _tier_prot)))
                elif (
                    not _tight_price
                    and _chase_binding_mins
                    and price_relaxation_level > 0
                    and "Proteins" in _joint_limits
                    and attempts <= NUTRIENT_NUMERIC_NORMAL_LIMIT_ATTEMPTS
                ):
                    # Price has been progressively widened but we are still in the
                    # normal-limits window — lift Proteins to customization max so
                    # Extra protein can help hit Mins (no artificial slot cap).
                    _pmin, _pmax = _joint_limits["Proteins"]
                    _cprot = category_limits_customization.get(
                        "Proteins", (_pmin, _pmax)
                    )
                    try:
                        _cprot_max = int(_cprot[1])
                    except (TypeError, ValueError, IndexError):
                        _cprot_max = int(_pmax)
                    _joint_limits["Proteins"] = (_pmin, max(int(_pmax), _cprot_max))
                # Keep warm+cold within Pricing side tier so nothing overflows to Extra * (paid).
                if _tight_price:
                    _wmin, _wmax = _joint_limits.get("Warm sides", (0, 0))
                    _cmin, _cmax = _joint_limits.get("Cold sides", (0, 0))
                    try:
                        _wmax_i, _cmax_i = int(_wmax), int(_cmax)
                    except (TypeError, ValueError):
                        _wmax_i, _cmax_i = 0, 0
                    # Prefer max cold (often high protein) then fill remaining tier with warm.
                    _c_keep = min(_cmax_i, _tier_sides)
                    _w_keep = min(_wmax_i, max(0, _tier_sides - _c_keep))
                    _joint_limits["Cold sides"] = (_cmin, _c_keep)
                    _joint_limits["Warm sides"] = (_wmin, _w_keep)
                # Cycle only proteins that can hit binding Mins under these same slot caps.
                _afford_prot = _heybo_pool_affordable(
                    _joint_pools["Proteins"],
                    df_heybo,
                    "Proteins",
                    _attempt_headroom,
                )
                _fp_cycle = _heybo_estimate_feasible_proteins(
                    _afford_prot or _joint_pools["Proteins"],
                    df=df_heybo,
                    cache=_ingredient_nutrient_cache,
                    numeric_nutrient_filters=_current_nf or [],
                    remaining_headroom=_attempt_headroom,
                    warm_pool=_joint_pools["Warm sides"],
                    cold_pool=_joint_pools["Cold sides"],
                    warm_max=int(_joint_limits.get("Warm sides", (0, 0))[1] or 0),
                    cold_max=int(_joint_limits.get("Cold sides", (0, 0))[1] or 0),
                    base_pool=_joint_pools["Bases"],
                    dip_pool=_joint_pools["Dips"],
                    garnish_pool=_joint_pools["Garnish"],
                    sauce_pool=_joint_pools["Sauces"],
                    protein_max=int(_joint_limits.get("Proteins", (0, 1))[1] or 1),
                )
                # Estimate empty under current slots — still probe top proteins for a
                # few attempts; progressive RELAXATION_ORDER (price first) widens later.
                if not _fp_cycle and _chase_binding_mins and len(heybo_bowls) == 0:
                    failed_min_under_budget_attempts += 1
                if not _fp_cycle:
                    _fp_cycle = _heybo_sort_pool_by_nutrient(
                        _afford_prot or _joint_pools["Proteins"],
                        _current_nf or [],
                        df_heybo,
                        _ingredient_nutrient_cache,
                    )[:3]
                # User Include/Extra proteins ALWAYS lead — do not drop them just
                # because feasibility optimism preferred other "capable" proteins.
                _user_prot = [
                    p
                    for p in (requested_by_category.get("Proteins") or [])
                    if p in (_joint_pools.get("Proteins") or [])
                    and p not in set(exclude_list or [])
                ]
                if _user_prot:
                    _fp_cycle = list(dict.fromkeys(_user_prot + list(_fp_cycle or [])))
                _sauce_list = list(_joint_pools.get("Sauces") or [])
                _prefer_sauce = None
                _user_sauces = [
                    s
                    for s in (requested_by_category.get("Sauces") or [])
                    if s in _sauce_list and s not in set(exclude_list or [])
                ]
                _primary = (
                    _fp_cycle[(attempts - 1) % len(_fp_cycle)] if _fp_cycle else None
                )
                # Prefer cycling user-requested proteins first across attempts.
                if _user_prot:
                    _primary = _user_prot[(attempts - 1) % len(_user_prot)]
                # User-requested sauce always wins (salad.py prefer_unused parity).
                if _user_sauces:
                    _prefer_sauce = _user_sauces[(attempts - 1) % len(_user_sauces)]
                # Prefer capable proteins that still have unused sauce keys — do not
                # burn attempts on weak proteins just because they are "unused".
                elif not _constraint_variety_relaxed and used_protein_sauce_combos is not None and _fp_cycle:
                    _cycle_for_sauce = _user_prot if _user_prot else _fp_cycle
                    _with_unused = []
                    for _p in _cycle_for_sauce:
                        _unused_s = [
                            s
                            for s in _sauce_list
                            if _heybo_primary_bowl_key(_p, s) not in used_protein_sauce_combos
                        ]
                        if _unused_s:
                            _with_unused.append((_p, _unused_s))
                    if _with_unused:
                        _p_idx = (attempts - 1) % len(_with_unused)
                        _primary, _unused_s = _with_unused[_p_idx]
                        _prefer_sauce = _unused_s[(attempts - 1) % len(_unused_s)]
                    elif _fp_cycle:
                        _primary = (
                            _user_prot[(attempts - 1) % len(_user_prot)]
                            if _user_prot
                            else _fp_cycle[(attempts - 1) % len(_fp_cycle)]
                        )
                elif _fp_cycle and _sauce_list:
                    # Even without combo tracking, rotate sauces for mild variety.
                    _prefer_sauce = _sauce_list[(attempts - 1) % len(_sauce_list)]
                _must_include = {
                    cat: [
                        x
                        for x in (requested_by_category.get(cat) or [])
                        if x not in set(exclude_list or [])
                    ]
                    for cat in REQUIRED_BOWL_CATEGORIES
                }
                print(
                    f"[CONSTRAINT-FIRST] attempt={attempts} joint-fill primary={_primary!r} "
                    f"sauce={_prefer_sauce!r} mins={_binding_min_targets} "
                    f"headroom={_attempt_headroom} "
                    f"protein_slots={_joint_limits.get('Proteins')} tight_price={_tight_price} "
                    f"capable={_fp_cycle[:5]} user_prot={_user_prot}"
                )
                _heybo_fill_bowl_joint_constraints(
                    bowl,
                    df=df_heybo,
                    cache=_ingredient_nutrient_cache,
                    pools_by_category=_joint_pools,
                    limits_by_category=_joint_limits,
                    numeric_nutrient_filters=_current_nf or [],
                    remaining_headroom=_attempt_headroom,
                    primary_protein=_primary,
                    prefer_sauce=_prefer_sauce,
                    exclude_set=set(exclude_list),
                    used_combos=used_protein_sauce_combos
                    if not _constraint_variety_relaxed
                    else None,
                    sauce_pool=list(sauces),
                    must_include_by_category=_must_include,
                    incompatible_pairs=incompatible_pairs,
                    user_included=user_included_for_compat,
                )
                for _ing in _user_prot:
                    if _ing in (bowl.get("Proteins") or []) + (bowl.get("Extra Proteins") or []):
                        bowl_validations["ingredient_explanations"].append(
                            f"{_ing}: User requested (Include)"
                            if _ing in set(include_list or [])
                            else f"{_ing}: User requested (Extra)"
                        )
                # Sync Max budget to joint-fill totals so structural side top-up
                # cannot re-pack high-protein sides past Max (e.g. protein 70–80).
                if _nutrient_budget:
                    _jf_tot = calculate_heybo_total_nutrients(bowl, df_heybo)
                    for _bk, _bud in _nutrient_budget.items():
                        _used = float((_jf_tot or {}).get(_bk) or 0.0)
                        _orig_max = None
                        for _nf in (_current_nf or []):
                            from .diet import _resolve_nutrient_filter_key as _rkey_b

                            _rk = _rkey_b(str((_nf or {}).get("Nutrient") or "").strip())
                            if _rk == _bk:
                                _rng = (_nf or {}).get("Range") or {}
                                if _rng.get("Max") is not None:
                                    try:
                                        _orig_max = float(_rng["Max"])
                                    except (TypeError, ValueError):
                                        _orig_max = None
                                break
                        if _orig_max is not None:
                            _bud["remaining_max"] = max(0.0, _orig_max - _used)
                bowl_validations["bowl_composition"].append(
                    f"Constraint-first joint fill (primary protein={_primary})"
                )
                for _cat in REQUIRED_BOWL_CATEGORIES:
                    bowl_validations["bowl_composition"].append(
                        f"{_cat}: {len(bowl.get(_cat) or [])} constructive pick(s) "
                        f"{bowl.get(_cat) or []}"
                    )
                _jf_sides = len(bowl.get("Warm sides") or []) + len(
                    bowl.get("Cold sides") or []
                )
                _jf_proteins = len(bowl.get("Proteins") or []) + len(
                    bowl.get("Extra Proteins") or []
                )
                # Keep joint-fill whenever it built a protein bowl. Falling back to
                # normal fill when sides < 3 wiped Max-safe protein bowls (e.g.
                # 60–70g) and re-packed Extra steak + high-protein sides past Max.
                # Only abandon when joint-fill could not place any sides under Max
                # (empty-side Max-ceiling packs) AND also got no proteins.
                if _jf_proteins > 0 and _jf_sides > 0:
                    _joint_fill_applied = True
                elif _jf_proteins > 0 and _jf_sides == 0:
                    # Zero sides under Max — keep proteins; later side top-up may
                    # still add light sides under remaining nutrient budget.
                    _joint_fill_applied = True
                    print(
                        f"[CONSTRAINT-FIRST] attempt={attempts}: joint-fill sides "
                        f"got=0 (Max-tight) — keeping joint-fill proteins for "
                        f"budgeted side top-up"
                    )
                else:
                    print(
                        f"[CONSTRAINT-FIRST] attempt={attempts}: joint-fill empty "
                        f"(proteins={_jf_proteins} sides={_jf_sides}) — "
                        f"fallback to normal fill"
                    )
                    for _cat in REQUIRED_BOWL_CATEGORIES:
                        bowl[_cat] = []
                    for _ek in (
                        "Extra Proteins",
                        "Extra Warm sides",
                        "Extra Cold sides",
                    ):
                        bowl[_ek] = []
                    bowl_validations["bowl_composition"] = [
                        c
                        for c in bowl_validations["bowl_composition"]
                        if "joint fill" not in c.lower()
                        and "constructive pick" not in c.lower()
                    ]
            if not _joint_fill_applied:
              for category in REQUIRED_BOWL_CATEGORIES:
                if category in categories_ignored_full_exclude:
                    bowl[category] = []
                    bowl_validations["bowl_composition"].append(
                        f"{category}: omitted — all available options excluded"
                    )
                    if attempts == 1:
                        print(f"Full exclude — omitting {category} (customization min 0)")
                    continue
                eff_lim = _effective_limit_for_cat(category, attempts)
                if not eff_lim or len(eff_lim) < 2:
                    continue
                min_count, max_count = int(eff_lim[0]), int(eff_lim[1])
                cat_exclude = _heybo_category_exclude_set(
                    category, exclude_list, categories_full_exclude_conflict
                )
                if attempts == 1:
                    if category in categories_full_exclude_conflict:
                        req = _heybo_full_exclude_conflict_requirement_phrase(
                            category,
                            _heybo_full_exclude_conflict_min(
                                category, category_limits, category_limits_customization
                            ),
                        )
                        print(
                            f"Full exclude conflict — filling {category} from catalog "
                            f"(each bowl still includes {req})"
                        )
                    if strict_only_bowl:
                        print(f"Only mode — limits for {category}: ({min_count}, {max_count})")
                    elif numeric_customization_stage and attempts == (
                        NUTRIENT_NUMERIC_NORMAL_LIMIT_ATTEMPTS + 1
                    ):
                        print(
                            "Numeric nutrient mode — switched to normal min + customization max "
                            f"for {category}: ({min_count}, {max_count})"
                        )
                    elif category in categories_with_customization:
                        print(
                            f"Using ask-aware customization ceiling for {category}: "
                            f"({min_count}, {max_count})"
                        )
                    elif category in categories_with_extras:
                        print(
                            f"Using ask-aware Extra ceiling for {category}: "
                            f"({min_count}, {max_count})"
                        )
                    else:
                        print(f"Using normal limits for {category}: ({min_count}, {max_count})")
                db_min, db_max = min_count, max_count
                bowl_validations["bowl_composition"].append(f"{category}: {min_count}-{max_count} ingredients")
                user_requested = list(requested_by_category.get(category, []))
                user_requested = [i for i in user_requested if i not in cat_exclude]
                # Deduct user-requested ingredients from the running nutrient budget before
                # random picks so the pool filter sees the correct remaining headroom.
                if _nutrient_budget and user_requested:
                    _heybo_budget_deduct(_nutrient_budget, user_requested, df_heybo, _ingredient_nutrient_cache)
                if category == "Bases":
                    full_pool = [t for t in bases if t not in user_requested and t not in cat_exclude]
                elif category == "Proteins":
                    # Use full filtered catalog for pool/eligibility; proteins_for_attempt is ordering only.
                    catalog_proteins = list(
                        dict.fromkeys(
                            [t for t in (proteins or []) + (extra_proteins or []) if t not in cat_exclude]
                        )
                    )
                    full_pool = [t for t in catalog_proteins if t not in user_requested]
                    if category in categories_with_extras and len(full_pool) < max_count:
                        extra_pool = [
                            t for t in extra_proteins if t not in user_requested and t not in cat_exclude
                        ]
                        if len(full_pool) < min_count:
                            full_pool.extend(extra_pool)
                elif category == "Warm sides":
                    full_pool = [t for t in warm_sides if t not in user_requested and t not in cat_exclude]
                    if category in categories_with_extras and len(full_pool) < max_count:
                        extra_pool = [
                            t for t in extra_warm_sides if t not in user_requested and t not in cat_exclude
                        ]
                        if len(full_pool) < min_count:
                            full_pool.extend(extra_pool)
                elif category == "Cold sides":
                    full_pool = [t for t in cold_sides if t not in user_requested and t not in cat_exclude]
                    if category in categories_with_extras and len(full_pool) < max_count:
                        extra_pool = [
                            t for t in extra_cold_sides if t not in user_requested and t not in cat_exclude
                        ]
                        if len(full_pool) < min_count:
                            full_pool.extend(extra_pool)
                elif category == "Dips":
                    full_pool = [t for t in dips if t not in user_requested and t not in cat_exclude]
                elif category == "Garnish":
                    full_pool = [t for t in garnishes if t not in user_requested and t not in cat_exclude]
                elif category == "Sauces":
                    full_pool = [
                        t for t in sauces_for_attempt if t not in user_requested and t not in cat_exclude
                    ]
                else:
                    full_pool = []

                if not strict_only_bowl and minimum_fill_pools and db_min > 0:
                    mf_names = _heybo_minimum_fill_names_for_category(category, minimum_fill_pools)
                    full_pool, widened_pool = _heybo_merge_preference_and_minimum_fill_pool(
                        full_pool,
                        mf_names,
                        expand_when_unique_below=db_min,
                        exclude_set=cat_exclude,
                    )
                    if widened_pool:
                        bowl_validations["bowl_composition"].append(
                            f"{category}: widened with allergen/diet-safe ingredients "
                            f"(preference pool below DB minimum {db_min})"
                        )

                available_count = len(set(full_pool) | set(user_requested))
                catalog_count = len(attempt_catalog.get(category) or [])
                if db_min > 0 and catalog_count < db_min:
                    bowl_validations["bowl_composition"].append(
                        f"{category}: filtered catalog ({catalog_count}) cannot satisfy DB minimum {db_min}"
                    )
                    bowl_incomplete = True
                    break
                if db_min > 0 and available_count < db_min:
                    bowl_validations["bowl_composition"].append(
                        f"{category}: only {available_count} selectable for DB minimum {db_min}"
                    )
                    bowl_incomplete = True
                    break
                if available_count == 0:
                    if db_min > 0:
                        bowl_incomplete = True
                        break
                    min_count, max_count = 0, 0
                else:
                    max_count = min(db_max, available_count)
                    min_count = db_min if db_min > 0 else 0
                    if min_count > max_count:
                        min_count = max_count

                pool = (
                    [t for t in full_pool if t in only_allowed_names]
                    if strict_only_bowl
                    else list(full_pool)
                )
                if category == "Proteins" and pool and proteins_for_attempt:
                    preferred = [t for t in proteins_for_attempt if t in pool]
                    pool = preferred + [t for t in pool if t not in preferred]
                if not strict_only_bowl and global_validations.get("apriori_suggestions"):
                    user_included_ingredients = set(include_list + extra_list)
                    pool = _apply_heybo_apriori_enhancement_to_pool(
                        category,
                        pool,
                        global_validations,
                        user_included_ingredients=user_included_ingredients,
                        incompatible_pairs=incompatible_pairs,
                    )
                # Nutrient-biased sort: when a numeric nutrient filter is active, sort the
                # pool so the most fitting ingredients appear first (ascending composite
                # score). Budget filtering below preserves this order on the safe subset,
                # and biased sampling picks from the top portion, reducing failed attempts.
                if numeric_nutrient_mode and not strict_only_bowl and pool:
                    pool = _heybo_sort_pool_by_nutrient(
                        pool, numeric_nutrient_filters, df_heybo, _ingredient_nutrient_cache
                    )
                # Constraint-first: for Proteins, prefer jointly feasible proteins and
                # unused ones when variety is enforceable (2+ feasible).
                if (
                    constraint_first_mode
                    and category == "Proteins"
                    and not strict_only_bowl
                    and pool
                    and feasible_proteins_for_constraints
                ):
                    _fp = [p for p in feasible_proteins_for_constraints if p in pool]
                    if _fp:
                        if not _constraint_variety_relaxed and used_protein_sauce_combos:
                            _used_prots = {
                                str(k).split("_", 1)[0]
                                for k in used_protein_sauce_combos
                                if k and "_" in str(k)
                            }
                            _unused_fp = [p for p in _fp if p not in _used_prots]
                            if _unused_fp:
                                _fp = _unused_fp
                        pool = _fp + [p for p in pool if p not in _fp]

                if strict_only_bowl:
                    final_list = list(user_requested)
                    if len(final_list) > max_count:
                        _orig_slots = len(final_list)
                        final_list = final_list[:max_count]
                        bowl_validations["bowl_composition"].append(
                            f"{category}: Only mode — truncated to customization max {max_count}"
                        )
                        _item_word = "portion" if _orig_slots == 1 else "portions"
                        _cap_msg = (
                            f"You chose {_orig_slots} {_item_word} for {category}, but this bowl style allows at most {max_count}. "
                            f"We included {max_count} in your bowl."
                        )
                        global_validations["category_limit_notices"].append(_cap_msg)
                        global_validations["filter_summary"].append(_cap_msg)
                    # Only mode: auto-fill Bases from catalog to satisfy DB/customization minimum.
                    if category == "Bases":
                        target_total = min(max(len(final_list), min_count), max_count)
                        needed = max(0, target_total - len(final_list))
                    else:
                        needed = 0
                else:
                    if variety_context_active and user_requested:
                        # Variety bowls (Only mode bowls 2-5): randomly sample a subset of the
                        # included ingredients so different bowls feature different combinations.
                        # The remaining slots are filled from the full filtered catalog, creating
                        # genuine ingredient variety across bowls.
                        n_keep = random.randint(0, min(len(user_requested), max_count))
                        final_list = random.sample(user_requested, n_keep)
                        if n_keep < len(user_requested):
                            bowl_validations["bowl_composition"].append(
                                f"{category}: variety bowl — using {n_keep} of "
                                f"{len(user_requested)} requested ingredient(s); "
                                f"remaining slots filled from catalog"
                            )
                    else:
                        final_list = list(user_requested)
                    # Cap user-requested list to effective max (ask-aware ceiling).
                    if (
                        category in categories_with_customization
                        or category in categories_with_extras
                    ) and len(final_list) > max_count:
                        _orig_slots = len(final_list)
                        final_list = random.sample(final_list, max_count)
                        bowl_validations["bowl_composition"].append(
                            f"{category}: sampled {max_count} of {_orig_slots} user-requested "
                            f"(ask-aware max {max_count})"
                        )
                    if not strict_only_bowl:
                        few_add = _heybo_few_true_match_additions(
                            user_input,
                            category,
                            max_count,
                            eligible=list(full_pool) + list(final_list),
                            already=final_list,
                            cuisine_active=cuisine_active
                            and not user_input.get("_relax_cuisine_filter"),
                            flavor_active=flavor_active
                            and not user_input.get("_relax_flavor_filter"),
                            prep_active=prep_active
                            and not user_input.get("_relax_preparation_method_filter"),
                        )
                        if few_add:
                            final_list.extend(few_add)
                            if _nutrient_budget:
                                _heybo_budget_deduct(
                                    _nutrient_budget,
                                    few_add,
                                    df_heybo,
                                    _ingredient_nutrient_cache,
                                )
                            bowl_validations["bowl_composition"].append(
                                f"{category}: FEW true cuisine/flavor/prep matches forced: {few_add}"
                            )
                    # Extra / overflow: ceiling only — pin slot count to user-requested
                    # size (do not pack random catalog proteins up to extras/customization max).
                    if (
                        (
                            category in categories_with_customization
                            or category in categories_with_extras
                        )
                        and not variety_context_active
                    ):
                        count_for_this_bowl = min(
                            max_count, max(min_count, len(final_list))
                        )
                    elif variety_context_active and _numeric_expand_for_min:
                        # When a Min-bound nutrient constraint is active (e.g. High Protein),
                        # user-requested Include items are pre-loaded and may fill all slots
                        # with nutritionally sub-optimal items (e.g. low-protein warm sides),
                        # leaving no room for high-nutrient catalog picks. Using max_count
                        # guarantees extra slots beyond the pre-loaded items so the nutrient
                        # sort can place the best-fit catalog ingredients into the bowl.
                        # This mirrors the Only=False behaviour where all slots are free.
                        count_for_this_bowl = max_count
                    elif (
                        constraint_first_mode
                        and (_numeric_expand_for_min or _chase_binding_mins)
                        and not strict_only_bowl
                    ):
                        # Always fill to max when chasing binding Mins (e.g. protein ≥ 60).
                        # random.randint(min, max) was picking Warm=0 / Proteins=1 and never
                        # reaching the Min even though a full max-slot bowl is feasible.
                        count_for_this_bowl = max_count
                    else:
                        count_for_this_bowl = random.randint(min_count, max_count)
                    needed = count_for_this_bowl - len(final_list)

                if needed > 0:
                    _cn_set = user_input.get("_cuisine_matched_ingredient_names") or set()
                    _use_cuisine_priority = (
                        cuisine_active
                        and not cuisine_relaxation_enabled
                        and bool(_cn_set)
                    )
                    if strict_only_bowl:
                        if category == "Bases":
                            added_ingredients = []
                            _only_include_names = set(include_list + extra_list)
                            for _ in range(needed):
                                choices = [
                                    t
                                    for t in full_pool
                                    if t not in final_list and t not in cat_exclude
                                ]
                                if not choices:
                                    break
                                if _only_include_names:
                                    compatible = [
                                        t
                                        for t in choices
                                        if _heybo_candidate_compatible(
                                            t,
                                            _only_include_names,
                                            incompatible_pairs,
                                            user_included_for_compat,
                                        )
                                    ]
                                    if compatible:
                                        choices = compatible
                                if _use_cuisine_priority:
                                    cfirst = [t for t in choices if t in _cn_set]
                                    pick = random.choice(cfirst if cfirst else choices)
                                else:
                                    pick = random.choice(choices)
                                added_ingredients.append(pick)
                                final_list.append(pick)
                                bowl_validations["ingredient_explanations"].append(
                                    f"{pick}: Only mode — base minimum (filtered catalog) for {category}"
                                )
                            if added_ingredients:
                                bowl_validations["bowl_composition"].append(
                                    f"{category}: Only mode — {len(user_requested)} user + "
                                    f"{len(added_ingredients)} base fill = {len(final_list)}"
                                )
                    elif pool:
                        _cn_set = user_input.get("_cuisine_matched_ingredient_names") or set()
                        _use_cuisine_priority = (
                            cuisine_active
                            and not cuisine_relaxation_enabled
                            and bool(_cn_set)
                        )
                        # Budget-aware pool filtering: divide remaining budget equally across
                        # the items about to be picked so the per-category total stays on track.
                        if _nutrient_budget:
                            _per_item_budget = {
                                k: {"remaining_max": bud["remaining_max"] / max(needed, 1), "min": bud.get("min")}
                                for k, bud in _nutrient_budget.items()
                            }
                            pool = _heybo_budget_filter_pool(
                                pool, _per_item_budget, df_heybo, _ingredient_nutrient_cache, needed
                            )
                        if category == "Sauces" and bowl.get("Proteins") and needed == 1:
                            prots = list(bowl["Proteins"])
                            added_ingredients = None
                            _user_sauces = [
                                s for s in pool if s in user_included_for_compat
                            ]
                            if _user_sauces:
                                added_ingredients = [_user_sauces[0]]
                            else:
                                for p in prots:
                                    unused_sauces = [
                                        s
                                        for s in pool
                                        if _heybo_primary_bowl_key(p, s)
                                        not in used_protein_sauce_combos
                                    ]
                                    if unused_sauces:
                                        if p != prots[0]:
                                            bowl["Proteins"] = [p] + [
                                                x for x in prots if x != p
                                            ]
                                        if _use_cuisine_priority:
                                            tier = _heybo_sample_pool_cuisine_first(
                                                unused_sauces, 1, _cn_set
                                            )
                                            added_ingredients = (
                                                tier
                                                if tier
                                                else [random.choice(unused_sauces)]
                                            )
                                        else:
                                            added_ingredients = [
                                                random.choice(unused_sauces)
                                            ]
                                        break
                            if added_ingredients is None:
                                n = min(needed, len(pool))
                                if _use_cuisine_priority:
                                    added_ingredients = _heybo_sample_pool_cuisine_first(pool, n, _cn_set)
                                else:
                                    added_ingredients = random.sample(pool, n)
                        else:
                            n = min(needed, len(pool))
                            if _use_cuisine_priority:
                                added_ingredients = _heybo_sample_pool_cuisine_first(pool, n, _cn_set)
                            elif _chase_binding_mins and not strict_only_bowl:
                                # Joint price+nutrient constructive pick (best-fit + affordable).
                                # Only when Min-chase is on — hard Min+Max bands must not
                                # filter by $0 headroom (drops light paid proteins).
                                _pref = None
                                if category == "Proteins" and feasible_proteins_for_constraints:
                                    _pref = [
                                        p for p in feasible_proteins_for_constraints if p in pool
                                    ]
                                added_ingredients = _heybo_constructive_sample(
                                    [x for x in pool if x not in final_list],
                                    n,
                                    df=df_heybo,
                                    category=category,
                                    numeric_nutrient_filters=(
                                        numeric_nutrient_filters if numeric_nutrient_mode else []
                                    ),
                                    cache=_ingredient_nutrient_cache,
                                    remaining_headroom=_attempt_headroom,
                                    prefer_names=_pref,
                                    exclude_names=set(final_list),
                                )
                                if len(added_ingredients) < n:
                                    _need_more = n - len(added_ingredients)
                                    _rest = [
                                        x
                                        for x in pool
                                        if x not in final_list and x not in added_ingredients
                                    ]
                                    if _rest:
                                        added_ingredients = added_ingredients + random.sample(
                                            _rest, min(_need_more, len(_rest))
                                        )
                                for _ing in added_ingredients:
                                    try:
                                        _c = float(
                                            _tier_default_price(_ing, df_heybo, category) or 0.0
                                        )
                                    except Exception:
                                        _c = 0.0
                                    if _attempt_headroom is not None:
                                        _attempt_headroom = max(0.0, _attempt_headroom - _c)
                                bowl_validations["ingredient_explanations"].append(
                                    f"{category}: constraint-first constructive pick "
                                    f"({len(added_ingredients)} item(s))"
                                )
                            elif numeric_nutrient_mode and len(pool) > n:
                                # Pool is already sorted by nutrient fitness (best-fit first).
                                # For narrow Max-only or tight low-nutrient windows (e.g. Max 300 kcal,
                                # Min:200 Max:300), tighten to the top 25% so picks cluster more
                                # aggressively toward low-calorie ingredients, reducing failed attempts
                                # and avoiding unnecessary relaxation.
                                # For Min-bound (High Protein / High Calorie) keep top 50% so
                                # high-nutrient ingredients are still reachable.
                                _is_tight_range = (
                                    not _numeric_expand_for_min
                                    and any(
                                        (
                                            (nf.get("Range") or {}).get("Max") is not None
                                            and (nf.get("Range") or {}).get("Min") is None
                                        )
                                        or (
                                            (nf.get("Range") or {}).get("Max") is not None
                                            and (nf.get("Range") or {}).get("Min") is not None
                                            and (
                                                float((nf.get("Range") or {}).get("Max", 999))
                                                - float((nf.get("Range") or {}).get("Min", 0))
                                            ) < 150
                                        )
                                        for nf in numeric_nutrient_filters
                                    )
                                )
                                _top_fraction = 0.25 if _is_tight_range else 0.5
                                _bias_n = min(len(pool), max(n, int(len(pool) * _top_fraction) + 1))
                                added_ingredients = random.sample(pool[:_bias_n], n)
                            else:
                                added_ingredients = random.sample(pool, n)
                        if _nutrient_budget and added_ingredients:
                            _heybo_budget_deduct(
                                _nutrient_budget, added_ingredients, df_heybo, _ingredient_nutrient_cache
                            )
                        final_list += added_ingredients
                        bowl_validations["bowl_composition"].append(
                            f"{category}: {len(user_requested)} requested + {len(added_ingredients)} random = {len(final_list)} total"
                        )
                        for ingredient in added_ingredients:
                            bowl_validations["ingredient_explanations"].append(
                                f"{ingredient}: Randomly selected for {category}"
                            )
                    else:
                        print(f"Warning: No pool available for {category}")
                        bowl_validations["bowl_composition"].append(f"{category}: No ingredients available in pool")
                else:
                    bowl_validations["bowl_composition"].append(
                        f"{category}: {len(user_requested)} requested (no additional needed)"
                    )
                for ingredient in user_requested:
                    if ingredient in include_list:
                        bowl_validations["ingredient_explanations"].append(f"{ingredient}: User requested (Include)")
                    elif ingredient in extra_list:
                        bowl_validations["ingredient_explanations"].append(f"{ingredient}: User requested (Extra)")
                if strict_only_bowl and category != "Bases":
                    if len(final_list) < min_count:
                        if len(final_list) == 0:
                            bowl_validations["bowl_composition"].append(
                                f"{category}: Only mode — no auto-fill (not in Include/Extra)"
                            )
                        else:
                            bowl_validations["bowl_composition"].append(
                                f"{category}: Only mode — {len(final_list)} user-requested "
                                f"(below catalog min {min_count}; not auto-filled)"
                            )
                else:
                    fallback_candidates = full_pool if strict_only_bowl else pool
                    if len(final_list) < min_count:
                        # Reach min_count only; do not use a large max_count (e.g. customization) as
                        # how many fallback picks to add in one batch for any category.
                        _slots_to_min = max(0, min_count - len(final_list))
                        possible_to_add = min(max_count - len(final_list), _slots_to_min)
                        eligible = [
                            t for t in fallback_candidates if t not in final_list and t not in cat_exclude
                        ]
                        if (
                            strict_only_bowl
                            and category == "Bases"
                            and (include_list or extra_list)
                        ):
                            _only_inc = set(include_list + extra_list)
                            compatible = [
                                t
                                for t in eligible
                                if _heybo_candidate_compatible(
                                    t,
                                    _only_inc,
                                    incompatible_pairs,
                                    user_included_for_compat,
                                )
                            ]
                            if compatible:
                                eligible = compatible
                        if possible_to_add > 0 and eligible:
                            add_count = min(possible_to_add, len(eligible))
                            _cn_set = user_input.get("_cuisine_matched_ingredient_names") or set()
                            _use_cuisine_priority = (
                                cuisine_active
                                and not cuisine_relaxation_enabled
                                and bool(_cn_set)
                            )
                            # Budget-aware fallback: divide remaining budget equally across
                            # the fallback items about to be picked.
                            if _nutrient_budget and not strict_only_bowl:
                                _fb_per_item = {
                                    k: {"remaining_max": bud["remaining_max"] / max(add_count, 1), "min": bud.get("min")}
                                    for k, bud in _nutrient_budget.items()
                                }
                                eligible = _heybo_budget_filter_pool(
                                    eligible, _fb_per_item, df_heybo, _ingredient_nutrient_cache, add_count
                                )
                                add_count = min(add_count, len(eligible))
                            if _use_cuisine_priority:
                                fallback_ingredients = _heybo_sample_pool_cuisine_first(
                                    eligible, add_count, _cn_set
                                )
                            else:
                                fallback_ingredients = random.sample(eligible, add_count)
                            if _nutrient_budget and fallback_ingredients:
                                _heybo_budget_deduct(
                                    _nutrient_budget, fallback_ingredients, df_heybo, _ingredient_nutrient_cache
                                )
                            final_list += fallback_ingredients
                            if strict_only_bowl and category == "Bases":
                                fb_label = "Only mode base minimum"
                            else:
                                fb_label = "Fallback (structural min)" if strict_only_bowl else "Fallback"
                            bowl_validations["bowl_composition"].append(
                                f"{category}: {fb_label} added {len(fallback_ingredients)} ingredients"
                            )
                            for ingredient in fallback_ingredients:
                                bowl_validations["ingredient_explanations"].append(
                                    f"{ingredient}: Fallback selection for {category} (minimum requirement)"
                                )
                        if len(final_list) < min_count:
                            print(
                                f"Warning: Could only fill {category} with {len(final_list)} items "
                                f"(minimum {min_count})"
                            )
                            bowl_validations["bowl_composition"].append(
                                f"{category}: Insufficient ingredients ({len(final_list)}/{min_count} minimum)"
                            )
                            bowl_incomplete = True
                            break
                bowl[category] = final_list
            if bowl_incomplete:
                if _use_joint_fill:
                    print(f"[CONSTRAINT-FIRST] skip attempt={attempts}: bowl_incomplete {bowl_validations.get('bowl_composition', [])[-3:]}")
                continue

            wm, warm_max = category_limits["Warm sides"]
            cm, cold_max = category_limits["Cold sides"]
            if strict_only_bowl:
                wm, warm_max = category_limits_customization.get("Warm sides", (wm, warm_max))
                cm, cold_max = category_limits_customization.get("Cold sides", (cm, cold_max))
            else:
                if numeric_customization_stage:
                    # Expand sides max only when nutrient filters are Min-bound (need more)
                    # OR the user explicitly requested extra sides.  For Max-only constraints
                    # (e.g. Low Calorie), adding extra sides increases calories and price.
                    _w_expand = _numeric_expand_for_min or _chase_binding_mins or (
                        "Warm sides" in categories_with_customization
                        or "Warm sides" in categories_with_extras
                    )
                    _c_expand = _numeric_expand_for_min or _chase_binding_mins or (
                        "Cold sides" in categories_with_customization
                        or "Cold sides" in categories_with_extras
                    )
                    if _w_expand:
                        _cmin, _cmax = category_limits_customization.get("Warm sides", (wm, warm_max))
                        warm_max = _cmax
                    if _c_expand:
                        _cmin, _cmax = category_limits_customization.get("Cold sides", (cm, cold_max))
                        cold_max = _cmax
                elif "Warm sides" in categories_with_customization:
                    wm, warm_max = category_limits_customization.get("Warm sides", (wm, warm_max))
                elif "Warm sides" in categories_with_extras:
                    wm, warm_max = category_limits_extras.get("Warm sides", (wm, warm_max))
                if "Cold sides" in categories_with_customization:
                    cm, cold_max = category_limits_customization.get("Cold sides", (cm, cold_max))
                elif "Cold sides" in categories_with_extras:
                    cm, cold_max = category_limits_extras.get("Cold sides", (cm, cold_max))
            # Structural MIN_TOTAL_SIDES top-up: do not use customization max as headroom here
            # when the user already overflowed normal limits for that side category.
            if not strict_only_bowl:
                if "Warm sides" in categories_with_customization:
                    wm, warm_max = category_limits["Warm sides"]
                if "Cold sides" in categories_with_customization:
                    cm, cold_max = category_limits["Cold sides"]
            wlist = list(bowl.get("Warm sides", []))
            clist = list(bowl.get("Cold sides", []))
            total_sides = len(wlist) + len(clist)
            warm_fill, cold_fill, sides_pool_widened = _heybo_side_fill_pools(
                warm_sides,
                cold_sides,
                extra_warm_sides,
                extra_cold_sides,
                minimum_fill_pools if not strict_only_bowl else {},
                exclude_list,
                MIN_TOTAL_SIDES_NORMAL if not strict_only_bowl else 0,
            )
            if sides_pool_widened:
                bowl_validations["bowl_composition"].append(
                    f"Warm/cold: widened with allergen/diet-safe sides "
                    f"(preference pool had fewer than {MIN_TOTAL_SIDES_NORMAL} sides)"
                )
            possible_sides_capacity = len(set(warm_fill + cold_fill))
            required_sides = min(MIN_TOTAL_SIDES_NORMAL, possible_sides_capacity)
            if not strict_only_bowl and required_sides > 0 and total_sides < required_sides:
                need = required_sides - total_sides
                exc_set = set(exclude_list)
                while need > 0:
                    warm_pool = [t for t in warm_fill if t not in wlist and t not in exc_set]
                    cold_pool = [t for t in cold_fill if t not in clist and t not in exc_set]
                    # Budget-aware top-up: budget each remaining side equally so the
                    # top-up doesn't undo the calorie-aware picks from the category loop.
                    if _nutrient_budget:
                        _topup_item_budget = {
                            k: {"remaining_max": bud["remaining_max"] / max(need, 1), "min": bud.get("min")}
                            for k, bud in _nutrient_budget.items()
                        }
                        if warm_pool:
                            _wf = _heybo_budget_filter_pool(
                                warm_pool, _topup_item_budget, df_heybo, _ingredient_nutrient_cache, 1
                            )
                            # After joint-fill Max sync, do not fall back to the full
                            # pool when nothing fits — that re-blows protein Max.
                            if _use_joint_fill and _wf is warm_pool:
                                _fit_w = [
                                    t for t in warm_pool
                                    if all(
                                        _heybo_ingredient_nutrient_val(
                                            t, k, df_heybo, _ingredient_nutrient_cache
                                        )
                                        <= bud["remaining_max"]
                                        for k, bud in _topup_item_budget.items()
                                    )
                                ]
                                warm_pool = _fit_w
                            else:
                                warm_pool = _wf
                        if cold_pool:
                            _cf = _heybo_budget_filter_pool(
                                cold_pool, _topup_item_budget, df_heybo, _ingredient_nutrient_cache, 1
                            )
                            if _use_joint_fill and _cf is cold_pool:
                                _fit_c = [
                                    t for t in cold_pool
                                    if all(
                                        _heybo_ingredient_nutrient_val(
                                            t, k, df_heybo, _ingredient_nutrient_cache
                                        )
                                        <= bud["remaining_max"]
                                        for k, bud in _topup_item_budget.items()
                                    )
                                ]
                                cold_pool = _fit_c
                            else:
                                cold_pool = _cf
                    can_w = len(wlist) < warm_max and bool(warm_pool)
                    can_c = len(clist) < cold_max and bool(cold_pool)
                    if not can_w and not can_c:
                        break
                    pick = None
                    if can_w and (not can_c or random.random() < 0.5):
                        pick = _heybo_pick_compatible_from_pool(
                            warm_pool,
                            bowl,
                            incompatible_pairs,
                            user_included_for_compat,
                            exc_set,
                            set(wlist),
                        )
                        if pick:
                            wlist.append(pick)
                            if _nutrient_budget:
                                _heybo_budget_deduct(_nutrient_budget, [pick], df_heybo, _ingredient_nutrient_cache)
                    if pick is None and can_c:
                        pick = _heybo_pick_compatible_from_pool(
                            cold_pool,
                            bowl,
                            incompatible_pairs,
                            user_included_for_compat,
                            exc_set,
                            set(clist),
                        )
                        if pick:
                            clist.append(pick)
                            if _nutrient_budget:
                                _heybo_budget_deduct(_nutrient_budget, [pick], df_heybo, _ingredient_nutrient_cache)
                    if pick is None:
                        break
                    need = required_sides - len(wlist) - len(clist)
                bowl["Warm sides"] = wlist
                bowl["Cold sides"] = clist
                if len(wlist) + len(clist) < required_sides:
                    bowl_validations["bowl_composition"].append(
                        f"Requires at least {required_sides} warm+cold sides; "
                        f"got {len(wlist) + len(clist)} (max warm {warm_max}, max cold {cold_max}) - skipped"
                    )
                    if _use_joint_fill:
                        print(
                            f"[CONSTRAINT-FIRST] skip attempt={attempts}: sides "
                            f"got={len(wlist)+len(clist)} need={required_sides} "
                            f"W={wlist} C={clist}"
                        )
                    continue

            bowl_weight = calculate_heybo_bowl_weight(bowl, df_heybo)
            if bowl_weight > max_bowl_weight:
                bowl_validations["bowl_composition"].append(f"Weight {bowl_weight}g exceeds {max_bowl_weight}g limit - skipped")
                if _use_joint_fill:
                    print(
                        f"[CONSTRAINT-FIRST] skip attempt={attempts}: weight "
                        f"{bowl_weight}g > {max_bowl_weight}g "
                        f"P={bowl.get('Proteins')} W={bowl.get('Warm sides')} "
                        f"C={bowl.get('Cold sides')}"
                    )
                continue
            else:
                bowl_validations["bowl_composition"].append(f"Weight: {bowl_weight}g (within {max_bowl_weight}g limit)")
            exclude_set = set(exclude_list)
            for key in ['Bases', 'Proteins', 'Warm sides', 'Cold sides', 'Dips', 'Garnish', 'Sauces']:
                if key in categories_full_exclude_conflict:
                    continue
                bowl[key] = [ing for ing in bowl.get(key, []) if ing not in exclude_set]
            if not _heybo_bowl_meets_db_limits(
                bowl,
                lambda c: _effective_limit_for_cat(c, attempts),
                attempt_catalog,
                exclude_set,
                bowl_validations,
                try_refill=True,
                only_allowed_names=only_allowed_names if strict_only_bowl else None,
                incompatible_pairs=incompatible_pairs,
                user_included=user_included_for_compat,
                full_exclude_conflict_categories=categories_full_exclude_conflict,
                nutrient_budget=_nutrient_budget if _use_joint_fill else None,
                df=df_heybo if _use_joint_fill else None,
                cache=_ingredient_nutrient_cache if _use_joint_fill else None,
            ):
                if _use_joint_fill:
                    print(
                        f"[CONSTRAINT-FIRST] skip attempt={attempts}: db_limits "
                        f"{bowl_validations.get('bowl_composition', [])[-5:]}"
                    )
                continue
            _redistribute_user_extra_slots_to_extra_categories(
                bowl,
                df_heybo,
                per_category_include_extra,
                user_prefix_lengths,
                required_categories,
            )
            _apply_combined_category_caps(bowl)
            if not _heybo_bowl_meets_db_limits(
                bowl,
                lambda c: _effective_limit_for_cat(c, attempts),
                attempt_catalog,
                exclude_set,
                bowl_validations,
                try_refill=True,
                only_allowed_names=only_allowed_names if strict_only_bowl else None,
                incompatible_pairs=incompatible_pairs,
                user_included=user_included_for_compat,
                full_exclude_conflict_categories=categories_full_exclude_conflict,
                nutrient_budget=_nutrient_budget if _use_joint_fill else None,
                df=df_heybo if _use_joint_fill else None,
                cache=_ingredient_nutrient_cache if _use_joint_fill else None,
            ):
                if _use_joint_fill:
                    print(
                        f"[CONSTRAINT-FIRST] skip attempt={attempts}: db_limits_post_cap "
                        f"{bowl_validations.get('bowl_composition', [])[-5:]}"
                    )
                continue
            bowl_weight = calculate_heybo_bowl_weight(bowl, df_heybo)
            bowl_ingredients = []
            for key in HEYBO_BOWL_COMPONENT_KEYS:
                bowl_ingredients += bowl.get(key, [])
            active_prep_methods = [
                m for m, enabled in user_input.get("PreparationMethod", {}).items() if enabled is True
            ]
            active_cuisines = user_input.get("CuisineFilters", [])
            if isinstance(active_cuisines, dict):
                active_cuisines = [k for k, v in active_cuisines.items() if v is True]
            active_flavors = {}
            for flavor, intensity in user_input.get("FlavorPreferences", {}).items():
                if not intensity:
                    continue
                if isinstance(intensity, str):
                    intensity = intensity.strip()
                    if not intensity:
                        continue
                active_flavors[flavor] = intensity

            has_prep_match = True
            if prep_active and not prep_relaxation_enabled and active_prep_methods:
                has_prep_match = False
                for ing in bowl_ingredients:
                    prep_val = str((ingredient_records.get(ing) or {}).get("preparation_method", ""))
                    if any(method in prep_val for method in active_prep_methods):
                        has_prep_match = True
                        break
                if has_prep_match:
                    if failed_prep_attempts > 0:
                        failed_prep_attempts = 0
                else:
                    failed_prep_attempts += 1
            else:
                failed_prep_attempts = 0

            has_cuisine_match = True
            if cuisine_active and not cuisine_relaxation_enabled and active_cuisines:
                has_cuisine_match = False
                for ing in bowl_ingredients:
                    cuisine_val = str((ingredient_records.get(ing) or {}).get("cuisine", ""))
                    if any(cuisine in cuisine_val for cuisine in active_cuisines):
                        has_cuisine_match = True
                        break
                if has_cuisine_match:
                    if failed_cuisine_attempts > 0:
                        failed_cuisine_attempts = 0
                else:
                    failed_cuisine_attempts += 1
            else:
                failed_cuisine_attempts = 0

            has_flavor_match = True
            if flavor_active and not flavor_relaxation_enabled and active_flavors and flavor_thresholds:
                has_flavor_match = False
                for ing in bowl_ingredients:
                    row = ingredient_records.get(ing) or {}
                    for flavor, intensity in active_flavors.items():
                        flavor_col = str(flavor).lower()
                        bounds = (flavor_thresholds.get(flavor_col) or {}).get(intensity)
                        if not bounds:
                            continue
                        val = row.get(flavor_col)
                        if val is None:
                            continue
                        mn, mx = bounds
                        try:
                            flavor_score = float(val)
                        except (TypeError, ValueError):
                            continue
                        if mn <= flavor_score <= mx:
                            has_flavor_match = True
                            break
                    if has_flavor_match:
                        break
                if has_flavor_match:
                    if failed_flavor_attempts > 0:
                        failed_flavor_attempts = 0
                else:
                    failed_flavor_attempts += 1
            else:
                failed_flavor_attempts = 0

            # Progressive relaxation in configurable priority order.
            relaxable_state = {
                "prep": {
                    "active": prep_active,
                    "enabled": prep_relaxation_enabled,
                    "failed": failed_prep_attempts,
                    "input_flag": "_relax_preparation_method_filter",
                    "validation_key": "preparation_method_matches",
                    "message": "Progressive relaxation: relaxed PreparationMethod after consecutive failures",
                },
                "cuisine": {
                    "active": cuisine_active,
                    "enabled": cuisine_relaxation_enabled,
                    "failed": failed_cuisine_attempts,
                    "input_flag": "_relax_cuisine_filter",
                    "validation_key": "cuisine_matches",
                    "message": "Progressive relaxation: relaxed CuisineFilters after consecutive failures",
                },
                "flavor": {
                    "active": flavor_active,
                    "enabled": flavor_relaxation_enabled,
                    "failed": failed_flavor_attempts,
                    "input_flag": "_relax_flavor_filter",
                    "validation_key": "flavor_adjustments",
                    "message": "Progressive relaxation: relaxed FlavorPreferences after consecutive failures",
                },
            }
            relaxed_non_nutrient = False
            for filter_name in RELAXATION_ORDER:
                if filter_name in ("nutrient", "light_hearty", "balanced_diet", "co2"):
                    continue
                if filter_name == "price":
                    _pf = user_input.get("Price", {})
                    _pmin, _pmax = _pf.get("Min"), _pf.get("Max")
                    # Progressive price widen after repeated over-budget OR repeated
                    # Min misses under floor price (price is first in RELAXATION_ORDER).
                    # Do not unlock only at 75% — give the requested budget several
                    # attempts first (threshold 40), then step slack one level at a time.
                    _price_fail_ready = (
                        failed_price_attempts >= 40
                        or failed_min_under_budget_attempts >= 40
                    )
                    if (
                        (_pmin is not None or _pmax is not None)
                        and _price_fail_ready
                        and price_relaxation_level < len(PRICE_RELAXATION_SLACKS) - 1
                    ):
                        failed_price_attempts = 0
                        failed_min_under_budget_attempts = 0
                        price_relaxation_level += 1
                        global_validations["filter_summary"].append(
                            "Progressive relaxation: widened price filter band after repeated misses "
                            f"(step {price_relaxation_level}, ±${PRICE_RELAXATION_SLACKS[price_relaxation_level]:.2f})"
                        )
                        print(
                            f"[CONSTRAINT-FIRST] progressive price relax "
                            f"level={price_relaxation_level}"
                        )
                        relaxed_non_nutrient = True
                        break
                    continue
                state = relaxable_state.get(filter_name)
                if not state:
                    continue
                if state["active"] and (not state["enabled"]) and state["failed"] >= 40:
                    user_input[state["input_flag"]] = True
                    global_validations[state["validation_key"]].append(state["message"])
                    if filter_name == "prep":
                        prep_relaxation_enabled = True
                        user_input["prep_relaxation_enabled"] = True
                        failed_prep_attempts = 0
                        failed_prep_pool_attempts = 0
                    elif filter_name == "cuisine":
                        cuisine_relaxation_enabled = True
                        user_input["cuisine_relaxation_enabled"] = True
                        failed_cuisine_attempts = 0
                        failed_cuisine_pool_attempts = 0
                    elif filter_name == "flavor":
                        flavor_relaxation_enabled = True
                        user_input["flavor_relaxation_enabled"] = True
                        failed_flavor_attempts = 0
                        failed_flavor_pool_attempts = 0
                    (
                        filtered_ingredients,
                        fallback_categories,
                        filter_error,
                        category_to_ingredients,
                        bases,
                        proteins,
                        warm_sides,
                        cold_sides,
                        dips,
                        garnishes,
                        sauces,
                        extra_proteins,
                        extra_warm_sides,
                        extra_cold_sides,
                    ) = rebuild_filtered_context()
                    if filter_error:
                        return {
                            "error": filter_error,
                            "message_to_user": build_heybo_message_to_user(
                                [], None, user_input=user_input, error=filter_error
                            ),
                        }
                    if filter_name in ("prep", "cuisine", "flavor"):
                        _refresh_feasible_proteins()
                    relaxed_non_nutrient = True
                    break
            if relaxed_non_nutrient:
                _heybo_try_cpsat_fill(
                    f"companion_relax=price{price_relaxation_level}"
                    f"/prep{int(prep_relaxation_enabled)}"
                    f"/cuisine{int(cuisine_relaxation_enabled)}"
                    f"/flavor{int(flavor_relaxation_enabled)}"
                )
                if len(heybo_bowls) >= target_bowl_count:
                    break
                continue

            if prep_active and not prep_relaxation_enabled and not has_prep_match:
                if _use_joint_fill:
                    print(f"[CONSTRAINT-FIRST] skip attempt={attempts}: prep_match")
                continue
            if cuisine_active and not cuisine_relaxation_enabled and not has_cuisine_match:
                if _use_joint_fill:
                    print(f"[CONSTRAINT-FIRST] skip attempt={attempts}: cuisine_match")
                continue
            if flavor_active and not flavor_relaxation_enabled and not has_flavor_match:
                if _use_joint_fill:
                    print(f"[CONSTRAINT-FIRST] skip attempt={attempts}: flavor_match")
                continue
            conflict_override_names = set()
            for _conf_cat in categories_full_exclude_conflict:
                conflict_override_names.update(category_to_ingredients.get(_conf_cat) or [])
            excluded_in_bowl = [
                ing
                for ing in bowl_ingredients
                if ing in exclude_set and ing not in conflict_override_names
            ]
            if excluded_in_bowl:
                bowl_validations["bowl_composition"].append(f"Excluded ingredients found: {', '.join(excluded_in_bowl)} - skipped")
                if _use_joint_fill:
                    print(
                        f"[CONSTRAINT-FIRST] skip attempt={attempts}: excluded {excluded_in_bowl}"
                    )
                continue
            else:
                bowl_validations["bowl_composition"].append("No excluded ingredients in bowl")

            if not strict_only_bowl:
                # Prefer unique protein+sauce combinations across generated bowls for this request.
                # This uses already-filtered pools, so it naturally applies to cuisine and all other active filters.
                # Uniqueness applies only when both primary protein and sauce were system-added —
                # never swap or reject a bowl because a user-requested Include/Extra repeats.
                _user_requested_ingredients = _heybo_user_requested_ingredient_set(
                    include_list, extra_list
                )
                available_proteins_for_combo = list(
                    dict.fromkeys(
                        [p for p in proteins if p not in exclude_set]
                        + [p for p in extra_proteins if p not in exclude_set]
                    )
                )
                available_sauces_for_combo = list(dict.fromkeys([s for s in sauces if s not in exclude_set]))
                primary_protein = _primary_protein_name(bowl)
                primary_sauce = (bowl.get("Sauces") or [None])[0]
                # When Light/Hearty is active and there are fewer proteins than
                # bowls to generate, the same protein must repeat across bowls —
                # enforcing unique protein+sauce combos is meaningless.
                # Relax immediately so all bowls can be filled with varied sides/sauces.
                # When proteins >= bowls needed, keep the normal uniqueness logic.
                _lh_dedup_relaxed = (
                    _lh_active and len(available_proteins_for_combo) < HEYBO_BOWLS_PER_PAGE
                )
                # Constraint-first: if fewer than 2 jointly feasible proteins, allow protein reuse
                # (variety is secondary to satisfying price+nutrient).
                _dedup_relaxed = _lh_dedup_relaxed or _constraint_variety_relaxed
                if _constraint_variety_relaxed and not lh_small_pool_dedup_logged:
                    # reuse flag slot only for one-time log; constraint-first has its own summary
                    pass
                if _lh_dedup_relaxed and not lh_small_pool_dedup_logged:
                    global_validations["filter_summary"].append(
                        f"Light/Hearty: only {len(available_proteins_for_combo)} protein(s) available "
                        f"(need {HEYBO_BOWLS_PER_PAGE} for unique combos) — "
                        "same protein can repeat across bowls, variety comes from sides/sauces"
                    )
                    lh_small_pool_dedup_logged = True
                if primary_protein and primary_sauce:
                    combo = _heybo_primary_bowl_key(primary_protein, primary_sauce)
                    _enforce_combo_diversity = _heybo_enforce_primary_combo_diversity(
                        primary_protein, primary_sauce, _user_requested_ingredients
                    )
                    if not _enforce_combo_diversity and combo in used_protein_sauce_combos:
                        bowl_validations["bowl_composition"].append(
                            f"Repeated primary pair {primary_protein}+{primary_sauce} kept "
                            "(user-requested Include/Extra — diversity not applied)"
                        )
                    unused_pairs_remain = (
                        False
                        if _dedup_relaxed or not _enforce_combo_diversity
                        else _unused_protein_sauce_pairs_exist(
                            available_proteins_for_combo, available_sauces_for_combo, used_protein_sauce_combos
                        )
                    )
                    if combo in used_protein_sauce_combos and unused_pairs_remain:
                        # Same overflow rule as category fill: do not add proteins for diversity
                        # up to customization max when customization was triggered by request overflow.
                        _div_protein_cap = (
                            len(_ordered_proteins_for_bowl(bowl))
                            if "Proteins" in categories_with_customization
                            else pd_max
                        )
                        if _try_fix_duplicate_primary_combo(
                            bowl,
                            available_proteins_for_combo,
                            available_sauces_for_combo,
                            used_protein_sauce_combos,
                            bowl_validations,
                            max_proteins_for_bowl=_div_protein_cap,
                            user_requested=_user_requested_ingredients,
                        ):
                            _redistribute_proteins_user_extra_slots(
                                bowl, df_heybo, per_category_include_extra
                            )
                            _apply_combined_category_caps(bowl)
                            bowl_ingredients = []
                            for key in HEYBO_BOWL_COMPONENT_KEYS:
                                bowl_ingredients += bowl.get(key, [])
                            primary_protein = _primary_protein_name(bowl)
                            primary_sauce = (bowl.get("Sauces") or [None])[0]
                            combo = _heybo_primary_bowl_key(primary_protein, primary_sauce)
                            bowl_weight = calculate_heybo_bowl_weight(bowl, df_heybo)
                            if bowl_weight > max_bowl_weight:
                                bowl_validations["bowl_composition"].append(
                                    f"Diversity fix increases weight to {bowl_weight}g over {max_bowl_weight}g limit - skipped"
                                )
                                continue
                        unused_pairs_remain = (
                            False
                            if not _enforce_combo_diversity
                            else _unused_protein_sauce_pairs_exist(
                                available_proteins_for_combo,
                                available_sauces_for_combo,
                                used_protein_sauce_combos,
                            )
                        )
                        if combo in used_protein_sauce_combos and unused_pairs_remain:
                            bowl_validations["bowl_composition"].append(
                                f"Repeated protein+sauce combo {primary_protein}+{primary_sauce} while unique combos remain - skipped"
                            )
                            global_validations["filter_summary"].append(
                                "Skipped bowl to maintain unique protein+sauce combinations under active filters"
                            )
                            continue
                    # Prefer unique (Proteins[0], Sauces[0]) while unused pairs exist; if the filtered pool
                    # cannot supply enough distinct pairs, allow repeats to still return a full page.
                    # Also skip this check when dedup is already relaxed (small Light/Hearty protein pool
                    # or constraint-first with <2 jointly feasible proteins).
                    if (
                        combo in used_protein_sauce_combos
                        and not _dedup_relaxed
                        and _enforce_combo_diversity
                    ):
                        still_unused = _unused_protein_sauce_pairs_exist(
                            available_proteins_for_combo, available_sauces_for_combo, used_protein_sauce_combos
                        )
                        if still_unused:
                            bowl_validations["bowl_composition"].append(
                                f"Repeated primary pair {primary_protein}+{primary_sauce} - skipped "
                                "(unique protein+sauce combinations still available)"
                            )
                            global_validations["filter_summary"].append(
                                "Skipped bowl because strict unique protein+sauce pairing is enabled"
                            )
                            continue
                        if not primary_combo_fallback_logged:
                            global_validations["filter_summary"].append(
                                "Unique primary protein+sauce pairs exhausted for the filtered pool — "
                                f"allowing repeated pairs to reach {HEYBO_BOWLS_PER_PAGE} bowls."
                            )
                            primary_combo_fallback_logged = True
                        bowl_validations["bowl_composition"].append(
                            f"Repeated primary pair {primary_protein}+{primary_sauce} accepted "
                            "(fallback: not enough distinct pairs under current filters)"
                        )
            bowl_set = frozenset(bowl_ingredients)
            if bowl_set in previous_bowls:
                bowl_validations["bowl_composition"].append("Duplicate bowl combination - skipped")
                if _use_joint_fill:
                    print(f"[CONSTRAINT-FIRST] skip attempt={attempts}: duplicate ingredient set")
                continue
            else:
                bowl_validations["bowl_composition"].append("Unique bowl combination")
            user_included = set(user_input.get('Ingredients', {}).get('Include', []))
            if not is_heybo_bowl_compatible(bowl, incompatible_pairs, user_included):
                bowl_validations["compatibility_status"] = "incompatible ingredients detected"
                if _use_joint_fill:
                    print(
                        f"[CONSTRAINT-FIRST] skip attempt={attempts}: incompatible "
                        f"P={bowl.get('Proteins')} S={bowl.get('Sauces')} "
                        f"W={bowl.get('Warm sides')} C={bowl.get('Cold sides')}"
                    )
                continue
            else:
                bowl_validations["compatibility_status"] = "all ingredients are compatible"
            if not (1 <= bowl_weight <= max_bowl_weight):
                bowl_validations["bowl_composition"].append(f"Weight {bowl_weight}g outside valid range (1-{max_bowl_weight}g) - skipped")
                continue
            else:
                bowl_validations["bowl_composition"].append(f"Weight {bowl_weight}g within valid range")
            apply_pricing_tier_split_to_bowl(bowl, df_heybo, heybo_cfg["price_config"])
            if not strict_only_bowl:
                _side_cap = len(set(warm_fill + cold_fill))
                _required_sides_final = min(MIN_TOTAL_SIDES_NORMAL, _side_cap)
                if (
                    _required_sides_final > 0
                    and _heybo_combined_warm_cold_side_count(bowl) < _required_sides_final
                ):
                    bowl_validations["bowl_composition"].append(
                        f"After pricing split: {_heybo_combined_warm_cold_side_count(bowl)} warm+cold "
                        f"side(s) (main + Extra *), need {_required_sides_final} — skipped"
                    )
                    continue
            bowl_cost, price_breakdown = calculate_heybo_bowl_cost_with_breakdown(
                bowl, df_heybo, price_config=heybo_cfg["price_config"]
            )
            price_filter = user_input.get("Price", {})
            min_price = price_filter.get("Min")
            max_price = price_filter.get("Max")
            price_validation = []
            eff_min, eff_max = _effective_price_bounds(
                min_price, max_price, price_relaxation_level
            )
            # If we capped a costlier request to MaxBowlPrice, don't let relaxation widen beyond that cap.
            max_bowl_price_cap = (heybo_cfg.get("price_config") or {}).get("max_bowl_price")
            if (
                user_input.get("_costlier_bowls_capped") is True
                and max_bowl_price_cap is not None
                and eff_max is not None
            ):
                try:
                    eff_max = min(float(eff_max), float(max_bowl_price_cap))
                except (TypeError, ValueError):
                    pass
            # Bowl 1 in Only mode is the user's exact ingredient selection — skip all
            # preference-based post-assembly gates (price, nutrient, Light/Hearty,
            # balanced, sustainable).  Only the ingredient compatibility, weight, and
            # dedup checks above apply.  Bowls 2-5 (variety) enforce all filters normally.
            if strict_only_bowl:
                price_validation.append("Only mode bowl 1: price filter not applied to exact-ingredient bowl")
                bowl_validations["price"] = price_validation
                total_nutrients = calculate_heybo_total_nutrients(bowl, df_heybo)
                bowl_validations["nutritional_targets"].append(
                    "Only mode bowl 1: nutrient filters not applied to exact-ingredient bowl"
                )
                bowl_validations["light_hearty_matches"].append(
                    "Only mode bowl 1: Light/Hearty filter not applied to exact-ingredient bowl"
                )
                bowl_validations["balanced_diet_matches"].append(
                    "Only mode bowl 1: balanced filter not applied to exact-ingredient bowl"
                )
            else:
                price_active = eff_min is not None or eff_max is not None
                price_ok = True
                if price_active:
                    if eff_min is not None and bowl_cost < eff_min:
                        price_ok = False
                        price_validation.append(
                            f"Cost ${bowl_cost:.2f} below allowed ${eff_min:.2f} - skipped"
                        )
                    elif eff_max is not None and bowl_cost > eff_max:
                        price_ok = False
                        price_validation.append(
                            f"Cost ${bowl_cost:.2f} above allowed ${eff_max:.2f} - skipped"
                        )
                    else:
                        # Only clear price-fail streak after we already have accepted bowls.
                        # At 0 bowls, in-budget (often free-tier) bowls that later fail a
                        # hard nutrient Max must not wipe pressure needed to widen price
                        # for paid light proteins (e.g. Baked Prawn at +$2.90).
                        if failed_price_attempts > 0 and len(heybo_bowls) > 0:
                            failed_price_attempts = 0
                        lo = f"{eff_min:.2f}" if eff_min is not None else "—"
                        hi = f"{eff_max:.2f}" if eff_max is not None else "—"
                        price_validation.append(
                            f"Cost ${bowl_cost:.2f} within allowed ${lo}–${hi}"
                        )
                if not price_ok:
                    failed_price_attempts += 1
                    if constraint_first_mode:
                        failed_joint_constraint_attempts += 1
                    if _use_joint_fill:
                        print(
                            f"[CONSTRAINT-FIRST] skip attempt={attempts}: price "
                            f"cost={bowl_cost} bounds=({eff_min},{eff_max}) "
                            f"extras={bowl.get('Extra Proteins')} "
                            f"breakdown_total={(price_breakdown or {}).get('total')}"
                        )
                    continue
                if not price_validation:
                    price_validation.append(f"Cost: ${bowl_cost:.2f} (no price constraints)")
                bowl_validations["price"] = price_validation
                total_nutrients = calculate_heybo_total_nutrients(bowl, df_heybo)
                nutrient_filters = numeric_nutrient_filters if numeric_nutrient_mode else nutrient_filters_all
                if nutrient_filters:
                    meets_nutrients, _ = meets_heybo_nutrient_filters_with_relaxation(
                        total_nutrients,
                        nutrient_filters,
                        relaxation_levels=nutrient_relaxation_levels,
                    )
                    if not meets_nutrients:
                        failed_nutrient_attempts += 1
                        if constraint_first_mode:
                            failed_joint_constraint_attempts += 1
                        # Mins missed while price Max still set → count toward progressive
                        # price widen (RELAXATION_ORDER: price before nutrient).
                        _pf_now = user_input.get("Price") or {}
                        _max_tg_early = _heybo_binding_max_nutrient_targets(
                            list(numeric_nutrient_filters or [])
                            or list(nutrient_filters or [])
                        )
                        _under_early = any(
                            float(total_nutrients.get(k) or 0.0) + 1e-9 < float(mn)
                            for k, mn in (_binding_min_targets or {}).items()
                        )
                        _over_max_early = (not _under_early) and any(
                            float(total_nutrients.get(k) or 0.0) > float(mx) + 1e-9
                            for k, mx in _max_tg_early.items()
                        )
                        if (
                            _chase_binding_mins
                            and len(heybo_bowls) == 0
                            and _pf_now.get("Max") is not None
                            and price_relaxation_level < len(PRICE_RELAXATION_SLACKS) - 1
                        ):
                            failed_min_under_budget_attempts += 1
                        # Hard Max band (e.g. calories ≤300) + floor price: free-tier bowls
                        # pass price then overshoot Max; paid light proteins need a wider
                        # band. Count Max overshoots toward price relax while still at 0 bowls.
                        elif (
                            _over_max_early
                            and len(heybo_bowls) == 0
                            and _pf_now.get("Max") is not None
                            and price_relaxation_level < len(PRICE_RELAXATION_SLACKS) - 1
                        ):
                            failed_price_attempts += 1
                        # When LH is active and the failing nutrient is one that is
                        # "covered" by the nutrient filter (skipped inside the LH check),
                        # the same failure would have incremented failed_light_hearty_attempts
                        # in a "Light only" run (where there is no nutrient filter and the
                        # calorie check lives inside the LH logic).  Count it here too so
                        # LH relaxation progresses at the same rate in both cases.
                        if _lh_active and _active_nutrient_keys:
                            failed_light_hearty_attempts += 1
                        bowl_validations["nutritional_targets"].append("Bowl does not meet nutrient filter requirements - skipped")
                        if _use_joint_fill:
                            print(
                                f"[CONSTRAINT-FIRST] skip attempt={attempts}: nutrients "
                                f"totals={ {k: total_nutrients.get(k) for k in (set(_binding_min_targets or {}) | set(_heybo_binding_max_nutrient_targets(_current_nf or [])))} } "
                                f"targets={_binding_min_targets} "
                                f"P={bowl.get('Proteins')} XP={bowl.get('Extra Proteins')} "
                                f"W={bowl.get('Warm sides')} "
                                f"C={bowl.get('Cold sides')} D={bowl.get('Dips')} "
                                f"G={bowl.get('Garnish')} S={bowl.get('Sauces')}"
                            )
                        # Progressive price widen (RELAXATION_ORDER: price first) after
                        # enough Min misses under the requested budget — not on first miss.
                        if (
                            len(heybo_bowls) == 0
                            and _chase_binding_mins
                            and _pf_now.get("Max") is not None
                            and price_relaxation_level < len(PRICE_RELAXATION_SLACKS) - 1
                            and failed_min_under_budget_attempts >= 40
                        ):
                            price_relaxation_level += 1
                            failed_min_under_budget_attempts = 0
                            failed_price_attempts = 0
                            global_validations["filter_summary"].append(
                                "Progressive relaxation: widened price after repeated Min misses "
                                f"under requested budget (step {price_relaxation_level}, "
                                f"±${PRICE_RELAXATION_SLACKS[price_relaxation_level]:.2f})"
                            )
                            print(
                                f"[CONSTRAINT-FIRST] progressive price relax after Min misses "
                                f"level={price_relaxation_level}"
                            )
                            _heybo_try_cpsat_fill(f"price_relaxation_level={price_relaxation_level}")
                            if len(heybo_bowls) >= target_bowl_count:
                                break
                            continue
                        # RELAXATION_ORDER: prep → cuisine → flavor → nutrient.
                        # Preferred-only bowls still "match" those filters, so bowl-check
                        # counters stay at 0 — count Min misses here and add extras instead
                        # of widening protein/calorie bands.
                        if (
                            prep_active
                            and not prep_relaxation_enabled
                            and _under_early
                            and len(heybo_bowls) == 0
                        ):
                            failed_prep_pool_attempts += 1
                            if failed_prep_pool_attempts >= 40:
                                _prep_err = _relax_prep_pool(
                                    "Progressive relaxation: added extra allergen/diet-safe "
                                    "ingredients after repeated nutrient misses under PreparationMethod "
                                    "(diet and allergens stay strict)"
                                )
                                if _prep_err:
                                    return _prep_err
                            continue
                        if (
                            cuisine_active
                            and not cuisine_relaxation_enabled
                            and _under_early
                            and len(heybo_bowls) == 0
                        ):
                            failed_cuisine_pool_attempts += 1
                            if failed_cuisine_pool_attempts >= 40:
                                _cuisine_err = _relax_cuisine_pool(
                                    "Progressive relaxation: added extra allergen/diet-safe "
                                    "ingredients after repeated nutrient misses under CuisineFilters "
                                    "(diet and allergens stay strict)"
                                )
                                if _cuisine_err:
                                    return _cuisine_err
                            continue
                        if (
                            flavor_active
                            and not flavor_relaxation_enabled
                            and _under_early
                            and len(heybo_bowls) == 0
                        ):
                            failed_flavor_pool_attempts += 1
                            if failed_flavor_pool_attempts >= 40:
                                _flavor_err = _relax_flavor_pool(
                                    "Progressive relaxation: added extra allergen/diet-safe "
                                    "ingredients after repeated nutrient misses under FlavorPreferences "
                                    "(diet and allergens stay strict)"
                                )
                                if _flavor_err:
                                    return _flavor_err
                            continue
                        # Do not widen Mins once we already have bowls that pass the
                        # current check — that only produces a false "relaxed" message.
                        # Also wait until price is fully widened (or absent) so price
                        # stays lower priority / sacrificed first per RELAXATION_ORDER.
                        # Prep, cuisine, and flavor must already be relaxed (or unset)
                        # before nutrient bands move.
                        _joint_relax_unlocked = (
                            (not constraint_first_mode)
                            or attempts
                            >= int(max_attempts * JOINT_CONSTRAINT_RELAX_AFTER_ATTEMPT_FRACTION)
                        )
                        _price_done = (
                            _pf_now.get("Max") is None
                            or price_relaxation_level >= len(PRICE_RELAXATION_SLACKS) - 1
                        )
                        _prep_done = (not prep_active) or prep_relaxation_enabled
                        _cuisine_done = (not cuisine_active) or cuisine_relaxation_enabled
                        _flavor_done = (not flavor_active) or flavor_relaxation_enabled
                        _co2_done = (not sustainable_on) or co2_relaxation_level >= CO2_MAX_RELAXATION_LEVEL
                        # Only widen nutrient bands when bowls are still UNDER Min.
                        # Max-only overshoots (e.g. 99g vs Max 80) must not trigger a
                        # 15% Min drop to ~60 — that returned false "relaxed" bowls.
                        _max_tg_now = _max_tg_early
                        _under_min_now = _under_early
                        _over_max_only = _over_max_early
                        if (
                            _joint_relax_unlocked
                            and _price_done
                            and _co2_done
                            and _prep_done
                            and _cuisine_done
                            and _flavor_done
                            and failed_nutrient_attempts >= 40
                            and len(heybo_bowls) == 0
                            and not _over_max_only
                        ):
                            active_keys = heybo_active_nutrient_filter_keys(nutrient_filters)
                            nutrient_relaxation_levels, relaxed_key = (
                                heybo_advance_priority_nutrient_relaxation(
                                    active_keys, nutrient_relaxation_levels
                                )
                            )
                            if relaxed_key:
                                failed_nutrient_attempts = 0
                                relaxed_lvl = nutrient_relaxation_levels[relaxed_key]
                                relaxed_pct = RELAXATION_PERCENTAGES[relaxed_lvl]
                                global_validations["diet_fallbacks"].append(
                                    f"Applied nutrient relaxation for {relaxed_key} "
                                    f"(priority order, level {relaxed_lvl}, {relaxed_pct}%)"
                                )
                                global_validations["filter_summary"].append(
                                    f"Nutrient priority relaxation: {relaxed_key} widened to "
                                    f"{relaxed_pct}% (others unchanged)"
                                )
                                _heybo_try_cpsat_fill(
                                    f"nutrient_relaxation={relaxed_key}={relaxed_lvl}"
                                )
                                if len(heybo_bowls) >= target_bowl_count:
                                    break
                        continue
                    else:
                        if failed_nutrient_attempts > 0:
                            failed_nutrient_attempts = 0
                        nutrient_relaxation_level = heybo_max_nutrient_relaxation_level(
                            nutrient_relaxation_levels
                        )
                        if nutrient_relaxation_level > 0:
                            relaxed_parts = []
                            for key in heybo_active_nutrient_filter_keys(nutrient_filters):
                                lvl = int(nutrient_relaxation_levels.get(key, 0) or 0)
                                if lvl > 0:
                                    relaxed_parts.append(
                                        f"{key} {RELAXATION_PERCENTAGES[lvl]}%"
                                    )
                            if relaxed_parts:
                                bowl_validations["nutritional_targets"].append(
                                    "Nutrient filters matched with priority-order relaxation: "
                                    + ", ".join(relaxed_parts)
                                )
                            else:
                                bowl_validations["nutritional_targets"].append(
                                    f"Nutrient filters matched with relaxation level "
                                    f"{nutrient_relaxation_level} "
                                    f"({RELAXATION_PERCENTAGES[nutrient_relaxation_level]}%)"
                                )
                        nut_names = [str(nf.get('Nutrient', nf)) for nf in nutrient_filters]
                        bowl_validations["nutritional_targets"].append(f"Applied nutrient filters: {', '.join(nut_names)}")
                else:
                    bowl_validations["nutritional_targets"].append("No nutrient filters applied")

                light_on = _is_truthy_flag(user_input.get("Light"))
                hearty_on = _is_truthy_flag(user_input.get("Hearty"))
                _lh_suppressed_now = bool(
                    user_input.get("_light_hearty_suppressed_by_nutrients")
                ) or light_hearty_suppressed_by_nutrient_filters(
                    list(nutrient_filters or []) or list(nutrient_filters_all or [])
                )
                if _lh_suppressed_now and (light_on or hearty_on):
                    # Explicit nutrient targets for calories/protein/fiber own the band —
                    # do not apply Light/Hearty bowl validation at all.
                    if not any(
                        isinstance(x, str) and "Light/Hearty ignored" in x
                        for x in bowl_validations.get("light_hearty_matches") or []
                    ):
                        _sk = user_input.get("_light_hearty_suppressed_nutrient_keys") or (
                            light_hearty_criteria_nutrient_overlap(
                                list(nutrient_filters or [])
                                or list(nutrient_filters_all or [])
                            )
                        )
                        bowl_validations["light_hearty_matches"].append(
                            "Light/Hearty ignored — NutrientFilters already set "
                            + ", ".join(_sk)
                            + " (same nutrients as Light/Hearty criteria)"
                        )
                elif not light_on and not hearty_on:
                    bowl_validations["light_hearty_matches"].append("No Light/Hearty preference specified")
                elif light_on and hearty_on:
                    bowl_validations["light_hearty_matches"].append(
                        "Light and Hearty both selected — combined ingredient scoring; bowl not scored to a single calorie band"
                    )
                else:
                    meets_lh, lh_details = evaluate_light_hearty_bowl(
                        total_nutrients,
                        bowl_weight,
                        light_on,
                        hearty_on,
                        light_hearty_relaxation_level,
                    )
                    if not meets_lh and lh_details:
                        failed_light_hearty_attempts += 1
                        if failed_light_hearty_attempts >= 40 and light_hearty_relaxation_level < 5:
                            failed_light_hearty_attempts = 0
                            light_hearty_relaxation_level += 1
                            user_input["light_hearty_relaxation_level"] = light_hearty_relaxation_level
                            _lh_pct = [0, 15, 30, 50, 70, 85][light_hearty_relaxation_level]
                            global_validations["filter_summary"].append(
                                f"Progressive relaxation: widened Light/Hearty bowl targets "
                                f"(level {light_hearty_relaxation_level}, {_lh_pct}% wider ranges)"
                            )
                            if light_hearty_relaxation_level == 1 and not _lh_dedup_relaxed:
                                global_validations["filter_summary"].append(
                                    "Light/Hearty dedup relaxed (level 1): duplicate protein+sauce combos now allowed to fill page"
                                )
                            # Legacy sync: if other nutrients share the run with Light,
                            # keep their relaxation in step. Overlapping Light criteria
                            # nutrients (cal/protein/fiber) no longer reach this path —
                            # Light is fully suppressed when those filters are present.
                            _max_relax_lvl = len(RELAXATION_PERCENTAGES) - 1
                            for _sync_key in (_active_nutrient_keys or []):
                                if _sync_key in (
                                    "calories_kCal",
                                    "protein_g",
                                    "fiber_g",
                                ):
                                    continue
                                _cur_sync_lvl = int(
                                    nutrient_relaxation_levels.get(_sync_key, 0) or 0
                                )
                                if _cur_sync_lvl < _max_relax_lvl:
                                    nutrient_relaxation_levels[_sync_key] = _cur_sync_lvl + 1
                                    _sync_pct = RELAXATION_PERCENTAGES[
                                        nutrient_relaxation_levels[_sync_key]
                                    ]
                                    global_validations["filter_summary"].append(
                                        f"Light/Hearty sync: nutrient constraint {_sync_key} "
                                        f"widened to {_sync_pct}% to match LH relaxation level "
                                        f"{light_hearty_relaxation_level}"
                                    )
                                    failed_nutrient_attempts = 0
                            _heybo_try_cpsat_fill(
                                f"light_hearty_relaxation_level={light_hearty_relaxation_level}"
                            )
                            if len(heybo_bowls) >= target_bowl_count:
                                break
                            continue
                        if light_hearty_relaxation_level >= 5:
                            bowl_validations["light_hearty_matches"].append(
                                "Maximum Light/Hearty relaxation reached — bowl accepted with wider nutritional tolerance"
                            )
                        else:
                            bowl_validations["light_hearty_matches"].append(
                                f"Bowl skipped — {lh_details['type']}: {format_light_hearty_failure(lh_details)}"
                            )
                            continue
                    else:
                        if failed_light_hearty_attempts > 0:
                            failed_light_hearty_attempts = 0
                        if lh_details:
                            msg = f"Bowl meets {lh_details['type']} guidelines (calories, protein, fiber, weight)"
                            if light_hearty_relaxation_level > 0:
                                msg += (
                                    f" (relaxation level {light_hearty_relaxation_level}, "
                                    f"{[0, 15, 30, 50, 70, 85][light_hearty_relaxation_level]}% wider ranges)"
                                )
                            bowl_validations["light_hearty_matches"].append(msg)

                if is_balanced_requested and (
                    user_input.get("_balanced_suppressed_by_nutrients")
                    or balanced_suppressed_by_nutrient_filters(
                        list(nutrient_filters or [])
                        or list(nutrient_filters_all or [])
                    )
                ):
                    if not user_input.get("_balanced_suppressed_by_nutrients"):
                        _sk = balanced_criteria_nutrient_overlap(
                            list(nutrient_filters or [])
                            or list(nutrient_filters_all or [])
                        )
                        user_input["_balanced_suppressed_by_nutrients"] = True
                        user_input["_balanced_suppressed_nutrient_keys"] = list(_sk)
                        _msg = (
                            "Balanced ignored — NutrientFilters already set "
                            + ", ".join(_sk)
                            + " (same nutrients as Balanced criteria)"
                        )
                        if not any(
                            isinstance(x, str) and "Balanced ignored" in x
                            for x in bowl_validations.get("balanced_diet_matches") or []
                        ):
                            bowl_validations["balanced_diet_matches"].append(_msg)
                    is_balanced_requested = False

                if is_balanced_requested:
                    cat_violations = _balanced_category_limit_violations(bowl)
                    if cat_violations:
                        bowl_validations["balanced_diet_matches"].append(
                            "Balanced diet: skipped — category limits exceeded: " + ", ".join(cat_violations)
                        )
                        continue
                    if not heybo_meets_balanced_diet(total_nutrients, balanced_relaxation_level):
                        failed_balanced_attempts += 1
                        if (
                            failed_balanced_attempts >= BALANCED_DIET_FAILURES_BEFORE_RELAX
                            and balanced_relaxation_level < 5
                        ):
                            failed_balanced_attempts = 0
                            balanced_relaxation_level += 1
                            user_input["balanced_relaxation_level"] = balanced_relaxation_level
                            _b_pct = RELAXATION_PERCENTAGES[balanced_relaxation_level]
                            global_validations["filter_summary"].append(
                                f"Progressive relaxation: widened balanced diet targets "
                                f"(level {balanced_relaxation_level}, {_b_pct}% wider ranges)"
                            )
                            _heybo_try_cpsat_fill(
                                f"balanced_relaxation_level={balanced_relaxation_level}"
                            )
                            if len(heybo_bowls) >= target_bowl_count:
                                break
                            continue
                        bowl_validations["balanced_diet_matches"].append(
                            "Bowl skipped — does not meet balanced diet criteria (calories, carbs, protein, fat)"
                        )
                        continue
                    if failed_balanced_attempts > 0:
                        failed_balanced_attempts = 0
                    bowl["Balanced_Nutrients"] = {k: total_nutrients.get(k, 0) for k in BALANCED_DIET_NUTRIENTS}
                    bowl["Balanced_Diet_Status"] = (
                        "Meets balanced diet criteria"
                        if balanced_relaxation_level == 0
                        else (
                            f"Meets balanced diet criteria (relaxed by "
                            f"{RELAXATION_PERCENTAGES[balanced_relaxation_level]}%)"
                        )
                    )
                    bowl_validations["balanced_diet_matches"].append(bowl["Balanced_Diet_Status"])
                elif global_validations.get("balanced_override"):
                    bowl_validations["balanced_diet_matches"].append(
                        global_validations.get(
                            "balanced_override_message",
                            "Balanced preference ignored — NutrientFilters already set overlapping criteria",
                        )
                    )
                else:
                    bowl_validations["balanced_diet_matches"].append("No balanced diet preference specified")

            bowl['Total Weight'] = round(bowl_weight)
            bowl['Total Cost'] = f"{bowl_cost:.2f}"
            bowl['PriceBreakdown'] = price_breakdown
            bowl['Total Nutrients'] = {
                nutrient: round(total_nutrients.get(nutrient, 0), 2)
                for nutrient in NUTRIENT_COLUMNS
            }
            bowl['Total_CO2e_g'] = calculate_heybo_total_co2e(bowl, df_heybo)
            if not strict_only_bowl and sustainable_on:
                _co2_lo, _co2_hi = _effective_sustainable_co2_bounds(
                    heybo_cfg, co2_relaxation_level
                )
                if _co2_lo is not None and _co2_hi is not None:
                    total_co2 = float(pd.to_numeric(bowl.get("Total_CO2e_g"), errors="coerce") or 0)
                    if total_co2 < _co2_lo or total_co2 > _co2_hi:
                        failed_co2_attempts += 1
                        if (
                            failed_co2_attempts >= 40
                            and co2_relaxation_level < CO2_MAX_RELAXATION_LEVEL
                        ):
                            failed_co2_attempts = 0
                            co2_relaxation_level += 1
                            user_input["co2_relaxation_level"] = co2_relaxation_level
                            if co2_relaxation_level == 1:
                                _co2_msg = (
                                    "Progressive relaxation: widened Sustainable CO2e band "
                                    f"(level {co2_relaxation_level}, max {CO2_RELAXATION_LEVEL_1_MAX:g})"
                                )
                            else:
                                _co2_msg = (
                                    "Progressive relaxation: widened Sustainable CO2e band "
                                    f"(level {co2_relaxation_level}, no upper limit)"
                                )
                            global_validations["filter_summary"].append(_co2_msg)
                            print(f"[CONSTRAINT-FIRST] {_co2_msg}")
                            _heybo_try_cpsat_fill(f"co2_relaxation_level={co2_relaxation_level}")
                            if len(heybo_bowls) >= target_bowl_count:
                                break
                            continue
                        if co2_relaxation_level < CO2_MAX_RELAXATION_LEVEL:
                            continue
                    elif failed_co2_attempts > 0:
                        failed_co2_attempts = 0
            bowl_validations["preparation_method_matches"] = validate_heybo_preparation_method_matches(
                bowl_ingredients, [], user_input, df_heybo
            )
            bowl_validations["cuisine_matches"] = validate_heybo_cuisine_matches(bowl_ingredients, [], user_input)
            bowl_validations["ingredient_explanations"] = validate_heybo_ingredient_selections(
                bowl, set(include_list), user_input, global_validations=global_validations
            )
            bowl_validations["diet_compatibility"] = validate_heybo_diet_compatibility(
                bowl, filtered_ingredients, user_input, fallback_categories
            )
            bowl_validations["allergen_safety"] = validate_heybo_allergen_safety(
                bowl, filtered_ingredients, user_input, catalog_df=df_heybo
            )
            bowl_validations["nutritional_targets"] = validate_heybo_nutritional_targets(
                bowl, total_nutrients, user_input
            )
            bowl_validations["bowl_composition"] = validate_heybo_bowl_composition(
                bowl, bowl.get("Bowl Type", "bowl"), bowl_weight, max_bowl_weight=max_bowl_weight
            )
            bowl_validations["price"] = validate_heybo_price(
                bowl_cost,
                user_input,
                effective_min=eff_min,
                effective_max=eff_max,
                price_relaxation_level=price_relaxation_level,
            )
            bowl['Validations'] = {"global": global_validations, "bowl_specific": bowl_validations}
            apriori_used = []
            for suggestion in (global_validations.get("apriori_suggestions") or []):
                ing_name = suggestion.get("ingredient_name")
                if ing_name and ing_name in bowl_ingredients:
                    apriori_used.append(
                        {
                            "ingredient": ing_name,
                            "category": suggestion.get("category"),
                            "confidence": suggestion.get("confidence", 0.0),
                            "support": suggestion.get("frequency", 0.0),
                        }
                    )
            if apriori_used:
                apriori_used.sort(key=lambda x: x.get("confidence", 0), reverse=True)
                bowl_validations["apriori_matches"].append(
                    f"Apriori-enhanced ingredients used: {len(apriori_used)}"
                )
                bowl_validations["apriori_matches"].append(
                    "Apriori details: "
                    + ", ".join(
                        [
                            f"{x['ingredient']} ({x['category']}, conf={float(x['confidence']):.3f})"
                            for x in apriori_used[:5]
                        ]
                    )
                )
            elif global_validations.get("apriori_suggestions"):
                bowl_validations["apriori_matches"].append(
                    "Apriori suggestions were available but none were selected in this bowl"
                )
            else:
                bowl_validations["apriori_matches"].append("No Apriori suggestions applied")
            if not _heybo_bowl_meets_db_limits(
                bowl,
                lambda c: _effective_limit_for_cat(c, attempts),
                attempt_catalog,
                exclude_set,
                bowl_validations,
                try_refill=False,
                only_allowed_names=only_allowed_names if strict_only_bowl else None,
                min_total_sides=(
                    None
                    if strict_only_bowl
                    else min(MIN_TOTAL_SIDES_NORMAL, len(set(warm_fill + cold_fill)))
                ),
                incompatible_pairs=incompatible_pairs,
                user_included=user_included_for_compat,
                full_exclude_conflict_categories=categories_full_exclude_conflict,
            ):
                continue
            bowl['message'] = extract_heybo_user_warnings(bowl['Validations'])
            pop_pricing_metadata(bowl)
            bowl["image_details"] = build_heybo_dynamic_grid_image_details(bowl, image_url_by_name)
            heybo_bowls.append(bowl)
            if constraint_first_mode:
                failed_joint_constraint_attempts = 0
            final_primary_protein = _primary_protein_name(bowl)
            final_primary_sauce = (bowl.get("Sauces") or [None])[0]
            if final_primary_protein and final_primary_sauce:
                used_protein_sauce_combos.add(
                    _heybo_primary_bowl_key(final_primary_protein, final_primary_sauce)
                )
            previous_bowls.add(bowl_set)
        _heybo_stamp_cpsat_bowls()
        if only_mode and heybo_bowls:
            base = "Lulu-BYB"
            prev_max = int(heybo_cfg.get("lulu_byb_only_mode_max_index") or 0)
            strict_bowl = heybo_bowls[0]
            count = prev_max + 1
            strict_bowl["Bowl Name"] = base if count == 1 else f"{base} - {count}"
            bowl_name_counts = dict(heybo_cfg.get("previous_bowl_name_index_by_base") or {})
            for bowl in heybo_bowls[1:]:
                final_protein = _primary_protein_name(bowl) or ""
                final_sauce = bowl["Sauces"][0] if bowl["Sauces"] else ""
                if final_protein and final_sauce:
                    base_bowl_name = f"{final_protein}_{final_sauce}"
                elif final_sauce:
                    base_bowl_name = f"{final_sauce}"
                elif final_protein:
                    base_bowl_name = f"{final_protein}"
                else:
                    base_bowl_name = "Lulu-BYB"
                bowl_name_counts[base_bowl_name] = bowl_name_counts.get(base_bowl_name, 0) + 1
                count = bowl_name_counts[base_bowl_name]
                bowl["Bowl Name"] = base_bowl_name if count == 1 else f"{base_bowl_name} - {count}"
        elif not only_mode:
            # ``previous_bowl_name_index_by_base``: from DB ``bowl_name`` for this session — largest
            # ``- N`` already used per base. We bump the counter here so new bowls get ``…`` or ``… - 2``, etc.
            bowl_name_counts = dict(heybo_cfg.get("previous_bowl_name_index_by_base") or {})
            for bowl in heybo_bowls:
                final_protein = _primary_protein_name(bowl) or ""
                final_sauce = bowl["Sauces"][0] if bowl["Sauces"] else ""
                if final_protein and final_sauce:
                    base_bowl_name = f"{final_protein}_{final_sauce}"
                elif final_sauce:
                    base_bowl_name = f"{final_sauce}"
                elif final_protein:
                    base_bowl_name = f"{final_protein}"
                else:
                    base_bowl_name = "Lulu-BYB"
                bowl_name_counts[base_bowl_name] = bowl_name_counts.get(base_bowl_name, 0) + 1
                count = bowl_name_counts[base_bowl_name]
                bowl["Bowl Name"] = base_bowl_name if count == 1 else f"{base_bowl_name} - {count}"
        if not heybo_bowls:
            err = "No bowls could be generated with current filters"
            nutrient_relaxation_level = heybo_max_nutrient_relaxation_level(
                nutrient_relaxation_levels
            )
            user_input["prep_relaxation_enabled"] = bool(
                prep_relaxation_enabled or user_input.get("prep_relaxation_enabled")
            )
            user_input["cuisine_relaxation_enabled"] = bool(
                cuisine_relaxation_enabled or user_input.get("cuisine_relaxation_enabled")
            )
            user_input["flavor_relaxation_enabled"] = bool(
                flavor_relaxation_enabled or user_input.get("flavor_relaxation_enabled")
            )
            user_input["co2_relaxation_level"] = int(co2_relaxation_level or 0)
            user_input["nutrient_relaxation_levels"] = dict(nutrient_relaxation_levels)
            user_input["nutrient_relaxation_level"] = nutrient_relaxation_level
            user_input["nutrient_relaxation_active_keys"] = heybo_active_nutrient_filter_keys(
                user_input.get("NutrientFilters") or []
            )
            global_validations["nutrient_relaxation_levels"] = dict(nutrient_relaxation_levels)
            global_validations["nutrient_relaxation_active_keys"] = list(
                user_input["nutrient_relaxation_active_keys"]
            )
            return {
                "error": err,
                "message_to_user": build_heybo_message_to_user(
                    [],
                    global_validations,
                    user_input=user_input,
                    fallback_categories=fallback_categories or [],
                    nutrient_relaxation_level=nutrient_relaxation_level,
                    nutrient_relaxation_levels=nutrient_relaxation_levels,
                    failed_nutrient_attempts=failed_nutrient_attempts,
                    price_relaxation_level=price_relaxation_level,
                ),
            }
        _cpsat_n = int(global_validations.get("_cpsat_bowls_generated") or 0)
        if _cpsat_n and attempts == 0:
            global_validations["filter_summary"].append(
                f"Successfully generated {len(heybo_bowls)} bowls via CP-SAT "
                f"({_cpsat_n} from solver, 0 random attempts)"
            )
        elif _cpsat_n:
            global_validations["filter_summary"].append(
                f"Successfully generated {len(heybo_bowls)} bowls "
                f"({_cpsat_n} via CP-SAT, rest after {attempts} random attempts)"
            )
        else:
            global_validations["filter_summary"].append(
                f"Successfully generated {len(heybo_bowls)} bowls after {attempts} attempts"
            )
        if (
            only_mode
            or categories_with_customization
            or (
                numeric_nutrient_mode
                and attempts > NUTRIENT_NUMERIC_NORMAL_LIMIT_ATTEMPTS
            )
        ):
            global_validations["category_limits_used"] = "customization"
        else:
            global_validations["category_limits_used"] = "normal"
        _heybo_flag_most_bowls_outside_price_range(
            heybo_bowls, user_input, price_relaxation_level
        )
        # Don't tell the user nutrients were relaxed when every returned bowl already
        # meets the original (unrelaxed) Min/Max — common when relaxation ran while
        # hunting for bowls 3–5 after 1–2 strict successes.
        if heybo_bowls and nutrient_relaxation_levels and any(
            int(v or 0) > 0 for v in nutrient_relaxation_levels.values()
        ):
            _orig_nf = user_input.get("NutrientFilters") or nutrient_filters or []
            if _orig_nf and all(
                meets_heybo_nutrient_filters(b.get("Total Nutrients") or {}, _orig_nf)
                for b in heybo_bowls
            ):
                nutrient_relaxation_levels = {}
                global_validations["diet_fallbacks"] = [
                    x
                    for x in (global_validations.get("diet_fallbacks") or [])
                    if "nutrient relaxation" not in str(x).lower()
                ]
                global_validations["filter_summary"] = [
                    x
                    for x in (global_validations.get("filter_summary") or [])
                    if "Nutrient priority relaxation" not in str(x)
                ]
                global_validations["nutrient_relaxation_active_keys"] = []
        nutrient_relaxation_level = heybo_max_nutrient_relaxation_level(
            nutrient_relaxation_levels
        )
        user_input["prep_relaxation_enabled"] = bool(
            prep_relaxation_enabled or user_input.get("prep_relaxation_enabled")
        )
        user_input["cuisine_relaxation_enabled"] = bool(
            cuisine_relaxation_enabled or user_input.get("cuisine_relaxation_enabled")
        )
        user_input["flavor_relaxation_enabled"] = bool(
            flavor_relaxation_enabled or user_input.get("flavor_relaxation_enabled")
        )
        user_input["co2_relaxation_level"] = int(co2_relaxation_level or 0)
        user_input["nutrient_relaxation_levels"] = dict(nutrient_relaxation_levels)
        user_input["nutrient_relaxation_level"] = nutrient_relaxation_level
        user_input["nutrient_relaxation_active_keys"] = heybo_active_nutrient_filter_keys(
            user_input.get("NutrientFilters") or []
        )
        global_validations["nutrient_relaxation_levels"] = dict(nutrient_relaxation_levels)
        global_validations["nutrient_relaxation_active_keys"] = list(
            user_input["nutrient_relaxation_active_keys"]
        )
        msg = build_heybo_message_to_user(
            heybo_bowls,
            global_validations,
            user_input=user_input,
            fallback_categories=fallback_categories or [],
            nutrient_relaxation_level=nutrient_relaxation_level,
            nutrient_relaxation_levels=nutrient_relaxation_levels,
            failed_nutrient_attempts=failed_nutrient_attempts,
            price_relaxation_level=price_relaxation_level,
        )
        return {
            "recommended_meal": heybo_bowls,
            "message_to_user": msg,
        }
    except Exception as e:
        print(f"Error generating Heybo bowls: {str(e)}")
        err = str(e)
        return {
            "error": err,
            "message_to_user": build_heybo_message_to_user(
                [], None, user_input=user_input, error=err
            ),
        }

