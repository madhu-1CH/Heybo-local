"""
Light / Hearty filtering and bowl validation (aligned with salad.py preference_filters + validate_light_hearty_criteria).
"""
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import DB_CONFIG
from .db import get_db_connection, return_db_connection

_FILTER_RANGES_CACHE: Optional[Dict[str, Any]] = None

_RELAX_PCT = [0, 0.15, 0.30, 0.50, 0.70, 0.85]

# Ingredient light_hearty score bands when preference_filters has no Light_Min/Max rows.
_DEFAULT_LIGHT_HEARTY_SCORES = {
    "Light": {"preferred": (1, 1), "fallback": (2, 2)},
    "Hearty": {"preferred": (2, 2), "fallback": (1, 1)},
}

# Bowl calorie/protein/fiber/weight bands only if DB load fails or preference_filters is missing
# any of Light_* / Hearty_* bowl keys (same fallbacks as salad.validate_light_hearty_criteria).
DEFAULT_LIGHT_BOWL_VALIDATION = {
    "calories_range": (300, 450),
    "protein_range": (20, 30),
    "fiber_range": (5, 8),
    "weight_range": (280, 400),
}
DEFAULT_HEARTY_BOWL_VALIDATION = {
    "calories_range": (650, 900),
    "protein_range": (35, 50),
    "fiber_range": (10, 18),
    "weight_range": (500, 700),
}

_LIGHT_BOWL_KEYS = (
    "Light_Calories_Min",
    "Light_Calories_Max",
    "Light_Protein_Min",
    "Light_Protein_Max",
    "Light_Fiber_Min",
    "Light_Fiber_Max",
    "Light_Weight_Min",
    "Light_Weight_Max",
)
_HEARTY_BOWL_KEYS = (
    "Hearty_Calories_Min",
    "Hearty_Calories_Max",
    "Hearty_Protein_Min",
    "Hearty_Protein_Max",
    "Hearty_Fiber_Min",
    "Hearty_Fiber_Max",
    "Hearty_Weight_Min",
    "Hearty_Weight_Max",
)


def _bowl_validation_light_from_ranges(ranges: Dict[str, float]) -> Dict[str, Any]:
    if all(k in ranges for k in _LIGHT_BOWL_KEYS):
        return {
            "calories_range": (int(ranges["Light_Calories_Min"]), int(ranges["Light_Calories_Max"])),
            "protein_range": (int(ranges["Light_Protein_Min"]), int(ranges["Light_Protein_Max"])),
            "fiber_range": (int(ranges["Light_Fiber_Min"]), int(ranges["Light_Fiber_Max"])),
            "weight_range": (int(ranges["Light_Weight_Min"]), int(ranges["Light_Weight_Max"])),
        }
    return dict(DEFAULT_LIGHT_BOWL_VALIDATION)


def _bowl_validation_hearty_from_ranges(ranges: Dict[str, float]) -> Dict[str, Any]:
    if all(k in ranges for k in _HEARTY_BOWL_KEYS):
        return {
            "calories_range": (int(ranges["Hearty_Calories_Min"]), int(ranges["Hearty_Calories_Max"])),
            "protein_range": (int(ranges["Hearty_Protein_Min"]), int(ranges["Hearty_Protein_Max"])),
            "fiber_range": (int(ranges["Hearty_Fiber_Min"]), int(ranges["Hearty_Fiber_Max"])),
            "weight_range": (int(ranges["Hearty_Weight_Min"]), int(ranges["Hearty_Weight_Max"])),
        }
    return dict(DEFAULT_HEARTY_BOWL_VALIDATION)


def _default_light_hearty_ranges() -> Dict[str, Any]:
    """Used only when the preference_filters query fails entirely (connection/DDL)."""
    return {
        "light_hearty": dict(_DEFAULT_LIGHT_HEARTY_SCORES),
        "light_hearty_bowl_validation": {
            "Light": dict(DEFAULT_LIGHT_BOWL_VALIDATION),
            "Hearty": dict(DEFAULT_HEARTY_BOWL_VALIDATION),
        },
    }


def load_light_hearty_filter_ranges() -> Dict[str, Any]:
    """
    Load Light/Hearty from heybo.preference_filters (authoritative when rows exist).

    Bowl bands use DB values for all Light_* / Hearty_* min/max keys; only if some rows are
    missing do we fill gaps with DEFAULT_*_BOWL_VALIDATION.
    """
    conn = None
    try:
        conn = get_db_connection(DB_CONFIG)
        query = """
            SELECT filter_name, filter_value
            FROM heybo.preference_filters
            WHERE filter_name IN (
                'Light_Min', 'Light_Max',
                'Hearty_Min', 'Hearty_Max',
                'Light_Calories_Min', 'Light_Calories_Max',
                'Light_Protein_Min', 'Light_Protein_Max',
                'Light_Fiber_Min', 'Light_Fiber_Max',
                'Light_Weight_Min', 'Light_Weight_Max',
                'Hearty_Calories_Min', 'Hearty_Calories_Max',
                'Hearty_Protein_Min', 'Hearty_Protein_Max',
                'Hearty_Fiber_Min', 'Hearty_Fiber_Max',
                'Hearty_Weight_Min', 'Hearty_Weight_Max'
            )
        """
        pdf = pd.read_sql(query, conn)
        ranges = {}
        for _, row in pdf.iterrows():
            ranges[row["filter_name"]] = float(row["filter_value"])

        light_hearty_config: Dict[str, Any] = {}
        if all(k in ranges for k in ("Light_Min", "Light_Max", "Hearty_Min", "Hearty_Max")):
            light_hearty_config["Light"] = {
                "preferred": (int(ranges["Light_Min"]), int(ranges["Light_Max"])),
                "fallback": (int(ranges["Hearty_Min"]), int(ranges["Hearty_Max"])),
            }
            light_hearty_config["Hearty"] = {
                "preferred": (int(ranges["Hearty_Min"]), int(ranges["Hearty_Max"])),
                "fallback": (int(ranges["Light_Min"]), int(ranges["Light_Max"])),
            }

        bowl_validation = {
            "Light": _bowl_validation_light_from_ranges(ranges),
            "Hearty": _bowl_validation_hearty_from_ranges(ranges),
        }

        return {
            "light_hearty": light_hearty_config,
            "light_hearty_bowl_validation": bowl_validation,
        }
    except Exception as exc:
        print(f"Warning: Light/Hearty preference_filters load failed ({exc}); using defaults.")
        return _default_light_hearty_ranges()
    finally:
        if conn:
            return_db_connection(conn, DB_CONFIG)


def get_light_hearty_ranges_cached() -> Dict[str, Any]:
    global _FILTER_RANGES_CACHE
    if _FILTER_RANGES_CACHE is None:
        _FILTER_RANGES_CACHE = load_light_hearty_filter_ranges()
    return _FILTER_RANGES_CACHE


def prime_light_hearty_cache_from_cfg(heybo_cfg: Dict[str, Any]) -> None:
    """
    Pre-warm the Light/Hearty cache from already-loaded heybo_cfg data so that
    evaluate_light_hearty_bowl never triggers a DB call during the generation loop.

    Called once per request right after get_heybo_config() returns.
    Falls back to the lazy DB load if heybo_cfg has no light_hearty_config.
    """
    global _FILTER_RANGES_CACHE
    lh_cfg = (heybo_cfg or {}).get("light_hearty_config")
    if not lh_cfg:
        # Config pre-loading not available — ensure cache is warm via DB fallback.
        get_light_hearty_ranges_cached()
        return
    # Only overwrite if the cache is currently cold to avoid clobbering a warm cache.
    if _FILTER_RANGES_CACHE is None:
        _FILTER_RANGES_CACHE = lh_cfg


def _light_hearty_flags(user_input: dict) -> Tuple[bool, bool]:
    """Salad-style: Light xor Hearty drive scoring filter; both True uses union range 1–2."""

    def on(key):
        v = user_input.get(key, False)
        if v is True:
            return True
        return str(v).strip().lower() in ("true", "1", "yes", "on")

    return on("Light"), on("Hearty")


def apply_light_hearty_filter(
    df: pd.DataFrame,
    user_input: dict,
    global_validations: dict,
    category_limits: Dict[str, Any],
) -> pd.DataFrame:
    """
    Restrict ingredient pool by heybo.ingredients_details.light_hearty score (Salad-style preferred/fallback per category).
    """
    key = "light_hearty_matches"
    if key not in global_validations:
        global_validations[key] = []

    if user_input.get("_relax_light_hearty_filter"):
        global_validations[key].append(
            "Light/Hearty ingredient filter relaxed — using full filtered pool"
        )
        return df

    light, hearty = _light_hearty_flags(user_input)
    if not light and not hearty:
        global_validations[key].append("No Light/Hearty filter applied")
        return df

    # NutrientFilters that overlap Light criteria (calories/protein/fiber) own those
    # targets — ignore Light/Hearty ingredient scoring entirely.
    _nf = (user_input or {}).get("NutrientFilters") or []
    _overlap = light_hearty_criteria_nutrient_overlap(_nf)
    if _overlap:
        user_input["_light_hearty_suppressed_by_nutrients"] = True
        user_input["_light_hearty_suppressed_nutrient_keys"] = list(_overlap)
        global_validations[key].append(
            "Light/Hearty ignored — NutrientFilters already set "
            + ", ".join(_overlap)
            + " (same nutrients as Light/Hearty criteria)"
        )
        return df

    if "light_hearty" not in df.columns:
        global_validations[key].append(
            "light_hearty column missing on ingredients — skipping Light/Hearty ingredient filter"
        )
        return df

    df = df.copy()
    df["light_hearty"] = pd.to_numeric(df["light_hearty"], errors="coerce")

    ranges = get_light_hearty_ranges_cached()
    light_hearty_config = ranges.get("light_hearty") or {}

    if light and hearty:
        both = df[df["light_hearty"].between(1, 2, inclusive="both")]
        if len(both) > 0:
            global_validations[key].append(
                f"Light and Hearty both on — using ingredients with light_hearty in 1–2 ({len(both)} rows)"
            )
            return both
        global_validations[key].append(
            "Light and Hearty both on — no rows in 1–2; keeping current pool"
        )
        return df

    if light and not hearty:
        tag = "Light"
        if "Light" in light_hearty_config:
            preferred_range = light_hearty_config["Light"]["preferred"]
            fallback_range = light_hearty_config["Light"]["fallback"]
        else:
            preferred_range = _DEFAULT_LIGHT_HEARTY_SCORES["Light"]["preferred"]
            fallback_range = _DEFAULT_LIGHT_HEARTY_SCORES["Light"]["fallback"]
    elif hearty and not light:
        tag = "Hearty"
        if "Hearty" in light_hearty_config:
            preferred_range = light_hearty_config["Hearty"]["preferred"]
            fallback_range = light_hearty_config["Hearty"]["fallback"]
        else:
            preferred_range = _DEFAULT_LIGHT_HEARTY_SCORES["Hearty"]["preferred"]
            fallback_range = _DEFAULT_LIGHT_HEARTY_SCORES["Hearty"]["fallback"]
    else:
        return df

    dedupe_col = "sku_code" if "sku_code" in df.columns else "ingredient_name"
    filtered_parts: List[pd.DataFrame] = []
    categories_fallback: List[str] = []

    for category in df["category"].dropna().unique():
        category_df = df[df["category"] == category].copy()
        min_required = 1
        if category in category_limits:
            lim = category_limits[category]
            if isinstance(lim, tuple) and len(lim) >= 1:
                min_required = int(lim[0])

        preferred_ingredients = category_df[
            category_df["light_hearty"].between(preferred_range[0], preferred_range[1], inclusive="both")
        ]
        preferred_count = len(preferred_ingredients)

        if preferred_count >= min_required:
            filtered_parts.append(preferred_ingredients)
        elif preferred_count > 0:
            fallback_ingredients = category_df[
                category_df["light_hearty"].between(fallback_range[0], fallback_range[1], inclusive="both")
            ]
            needed = min_required - preferred_count
            if len(fallback_ingredients) > 0:
                take = fallback_ingredients.head(min(needed, len(fallback_ingredients)))
                combined = pd.concat([preferred_ingredients, take], ignore_index=True).drop_duplicates(
                    subset=[dedupe_col]
                )
                filtered_parts.append(combined)
                categories_fallback.append(str(category))
            else:
                filtered_parts.append(preferred_ingredients)
                categories_fallback.append(str(category))
        else:
            fallback_ingredients = category_df[
                category_df["light_hearty"].between(fallback_range[0], fallback_range[1], inclusive="both")
            ]
            if len(fallback_ingredients) > 0:
                filtered_parts.append(fallback_ingredients)
                categories_fallback.append(str(category))
            else:
                filtered_parts.append(category_df)

    if not filtered_parts:
        global_validations[key].append(f"{tag} filter produced no rows — keeping original pool")
        return df

    out = pd.concat(filtered_parts, ignore_index=True).drop_duplicates(subset=[dedupe_col])
    global_validations[key].append(
        f"Applied {tag} light_hearty filter — {len(out)} ingredients in pool"
    )
    if categories_fallback:
        global_validations[key].append(
            f"{tag}: used fallback light_hearty band in categories: {', '.join(sorted(set(categories_fallback)))}"
        )
    return out


# Nutrients checked by Light/Hearty bowl validation in preference_filters
# (Light_Calories_*, Light_Protein_*, Light_Fiber_* — weight is not a NutrientFilters column).
LIGHT_HEARTY_CRITERIA_NUTRIENT_KEYS = frozenset(
    {"calories_kCal", "protein_g", "fiber_g"}
)

# Maps nutrient-filter column names → the corresponding Light/Hearty check dimension.
# Kept for evaluate_light_hearty_bowl(skip_nutrient_keys=...) callers; generation now
# suppresses the whole Light/Hearty constraint on any overlap instead.
_LH_NUTRIENT_KEY_MAP: Dict[str, str] = {
    "calories_kCal": "calories",
    "protein_g": "protein",
    "fiber_g": "fiber",
}


def light_hearty_criteria_nutrient_overlap(nutrient_filters) -> List[str]:
    """
    Return Light/Hearty criteria nutrient keys that also appear in NutrientFilters
    (calories_kCal, protein_g, fiber_g). Empty ⇒ no overlap.
    """
    from .diet import heybo_active_nutrient_filter_keys

    active = set(heybo_active_nutrient_filter_keys(nutrient_filters or []))
    return sorted(active & LIGHT_HEARTY_CRITERIA_NUTRIENT_KEYS)


def light_hearty_suppressed_by_nutrient_filters(nutrient_filters) -> bool:
    """
    True when any NutrientFilters nutrient is part of Light/Hearty bowl criteria.

    In that case the Light/Hearty constraint is ignored entirely (ingredient pool
    filter + bowl validation) so the explicit nutrient filter owns those targets.
    """
    return bool(light_hearty_criteria_nutrient_overlap(nutrient_filters))


def light_hearty_bowl_criteria_for_cpsat(
    light,
    hearty,
    relaxation_level: int = 0,
    skip_nutrient_keys=None,
) -> Tuple[List[dict], Optional[Tuple[float, float]]]:
    """
    Build CP-SAT inputs from Light/Hearty DB bowl-validation ranges.

    Returns:
        (nutrient_filters, weight_range_or_none)
        nutrient_filters use NutrientFilters shape for calories/protein/fiber.
        weight_range is (min_g, max_g) or None when Light/Hearty criteria do not apply.
    """
    if not (light or hearty) or (light and hearty):
        return [], None

    _skip = set(skip_nutrient_keys or [])
    ranges = get_light_hearty_ranges_cached()
    bowl_validation = ranges.get("light_hearty_bowl_validation") or {}
    if light and not hearty:
        config = bowl_validation.get("Light") or {}
        _fb = DEFAULT_LIGHT_BOWL_VALIDATION
    else:
        config = bowl_validation.get("Hearty") or {}
        _fb = DEFAULT_HEARTY_BOWL_VALIDATION

    calories_range = config.get("calories_range", _fb["calories_range"])
    protein_range = config.get("protein_range", _fb["protein_range"])
    fiber_range = config.get("fiber_range", _fb["fiber_range"])
    weight_range = config.get("weight_range", _fb["weight_range"])

    relax_pct = _RELAX_PCT[min(int(relaxation_level or 0), 5)]

    def _widen(r):
        lo, hi = float(r[0]), float(r[1])
        if relax_pct <= 0:
            return lo, hi
        delta = (hi - lo) * relax_pct
        return max(0.0, lo - delta), hi + delta

    calories_range = _widen(calories_range)
    protein_range = _widen(protein_range)
    fiber_range = _widen(fiber_range)
    weight_range = _widen(weight_range)

    nutrient_map = [
        ("calories_kCal", calories_range),
        ("protein_g", protein_range),
        ("fiber_g", fiber_range),
    ]
    filters: List[dict] = []
    for key, (lo, hi) in nutrient_map:
        if key in _skip:
            continue
        filters.append({
            "Nutrient": key,
            "Range": {"Min": lo, "Max": hi},
            "_cpsat_source": "light" if light and not hearty else "hearty",
        })
    return filters, (float(weight_range[0]), float(weight_range[1]))


def evaluate_light_hearty_bowl(
    total_nutrients: dict,
    bowl_weight: float,
    light: bool,
    hearty: bool,
    relaxation_level: int,
    skip_nutrient_keys: Optional[List[str]] = None,
) -> Tuple[bool, Optional[dict]]:
    """
    Final bowl check (calories, protein, fiber, weight). Matches salad validate_light_hearty_criteria.
    Returns (meets_all, details_or_none).

    skip_nutrient_keys: nutrient column names (e.g. ["calories_kCal"]) already enforced by
    Diet/Nutrient filters. The corresponding Light/Hearty dimension is auto-passed so the
    same nutrient is not double-checked with a potentially conflicting range.
    """
    if not (light or hearty):
        return True, None
    if light and hearty:
        return True, None

    _skip: set = set(skip_nutrient_keys) if skip_nutrient_keys else set()
    # Dimensions to skip (e.g. {"calories", "protein"})
    _skip_dims: set = {dim for key, dim in _LH_NUTRIENT_KEY_MAP.items() if key in _skip}

    ranges = get_light_hearty_ranges_cached()
    bowl_validation = ranges.get("light_hearty_bowl_validation") or {}
    if light and not hearty:
        config = bowl_validation.get("Light") or {}
        _fb = DEFAULT_LIGHT_BOWL_VALIDATION
    else:
        config = bowl_validation.get("Hearty") or {}
        _fb = DEFAULT_HEARTY_BOWL_VALIDATION

    calories_range = config.get("calories_range", _fb["calories_range"])
    protein_range = config.get("protein_range", _fb["protein_range"])
    fiber_range = config.get("fiber_range", _fb["fiber_range"])
    weight_range = config.get("weight_range", _fb["weight_range"])

    relax_pct = _RELAX_PCT[min(relaxation_level, 5)]
    if relax_pct > 0:

        def widen(r):
            lo, hi = r
            width = hi - lo
            delta = width * relax_pct
            return (max(0, lo - delta), hi + delta)

        calories_range = widen(calories_range)
        protein_range = widen(protein_range)
        fiber_range = widen(fiber_range)
        weight_range = widen(weight_range)

    details = {
        "type": "Light" if light else "Hearty",
        "calories_valid": True,
        "protein_valid": True,
        "fiber_valid": True,
        "weight_valid": True,
        "calories_actual": None,
        "calories_range": None,
        "protein_actual": None,
        "protein_range": None,
        "fiber_actual": None,
        "fiber_range": None,
        "weight_actual": None,
        "weight_range": None,
        "skipped_dims": sorted(_skip_dims) if _skip_dims else [],
    }

    calories_actual = float(total_nutrients.get("calories_kCal", 0) or 0)
    protein_actual = float(total_nutrients.get("protein_g", 0) or 0)
    fiber_actual = float(total_nutrients.get("fiber_g", 0) or 0)
    weight_actual = float(bowl_weight)

    cmin, cmax = calories_range
    details["calories_actual"] = round(calories_actual, 1)
    details["calories_range"] = f"{int(cmin)}-{int(cmax)}"
    if "calories" not in _skip_dims and not (cmin <= calories_actual <= cmax):
        details["calories_valid"] = False

    pmin, pmax = protein_range
    details["protein_actual"] = round(protein_actual, 1)
    details["protein_range"] = f"{int(pmin)}-{int(pmax)}"
    if "protein" not in _skip_dims and not (pmin <= protein_actual <= pmax):
        details["protein_valid"] = False

    fmin, fmax = fiber_range
    details["fiber_actual"] = round(fiber_actual, 1)
    details["fiber_range"] = f"{int(fmin)}-{int(fmax)}"
    if "fiber" not in _skip_dims and not (fmin <= fiber_actual <= fmax):
        details["fiber_valid"] = False

    wmin, wmax = weight_range
    details["weight_actual"] = round(weight_actual)
    details["weight_range"] = f"{int(wmin)}-{int(wmax)}"
    if not (wmin <= weight_actual <= wmax):
        details["weight_valid"] = False

    meets = (
        details["calories_valid"]
        and details["protein_valid"]
        and details["fiber_valid"]
        and details["weight_valid"]
    )
    return meets, details


def format_light_hearty_failure(details: dict) -> str:
    parts = []
    if not details.get("calories_valid"):
        parts.append(f"Calories {details['calories_actual']} (target {details['calories_range']} kcal)")
    if not details.get("protein_valid"):
        parts.append(f"Protein {details['protein_actual']}g (target {details['protein_range']} g)")
    if not details.get("fiber_valid"):
        parts.append(f"Fiber {details['fiber_actual']}g (target {details['fiber_range']} g)")
    if not details.get("weight_valid"):
        parts.append(f"Weight {details['weight_actual']}g (target {details['weight_range']} g)")
    skipped = details.get("skipped_dims") or []
    if skipped:
        parts.append(f"(skipped by nutrient filter: {', '.join(skipped)})")
    return "; ".join(parts) if parts else "Light/Hearty criteria not met"
