"""
Load heybo.nutrition_constraint_information and map diet names to nutrient filters (DB-driven).

Keep imports limited to .config and .db only — do not import .filters or .nutrition_constraints
(here or via copy-paste), or you will get a circular import.
"""
from functools import lru_cache
from typing import Any, Dict, List

import pandas as pd

from .config import DB_CONFIG
from .db import get_db_connection, return_db_connection

# Diet label → ingredient / bowl nutrient column (same semantics as SaladStop)
DIET_TO_NUTRIENT_MAP = {
    "High Calorie": "calories_kCal",
    "Low Calorie": "calories_kCal",
    "High Fat": "total_fat_g",
    "Low Fat": "total_fat_g",
    "Fat Free": "total_fat_g",
    "High Saturated Fat": "saturated_fat_g",
    "Low Saturated Fat": "saturated_fat_g",
    "Saturated Fat Free": "saturated_fat_g",
    "Lean Meat": "saturated_fat_g",
    "Extra Lean Meat": "saturated_fat_g",
    "High Cholesterol": "cholesterol_mg",
    "Low Cholesterol": "cholesterol_mg",
    "Cholesterol Free": "cholesterol_mg",
    "High Sodium": "sodium_mg",
    "Low Sodium": "sodium_mg",
    "Sodium Free": "sodium_mg",
    "Very Low in Sodium": "sodium_mg",
    "High Carbohydrates": "carbs_g",
    "Low Carbohydrates": "carbs_g",
    "High Fiber": "fiber_g",
    "Low Fiber": "fiber_g",
    "High Sugar": "sugar_g",
    "Low Sugar": "sugar_g",
    "Sugar Free": "sugar_g",
    "High Protein": "protein_g",
    "Low Protein": "protein_g",
    "High Calcium": "calcium_mg",
    "Low Calcium": "calcium_mg",
    "High Iron": "iron_mg",
    "Low Iron": "iron_mg",
    "High Potassium": "potassium_mg",
    "Low Potassium": "potassium_mg",
    "High Vitamin D": "vitamin_d_mcg",
    "Low Vitamin D": "vitamin_d_mcg",
    "High Phosphorus": "phosphorus_mg",
    "Low Phosphorus": "phosphorus_mg",
}

PROTEIN_SPECIFIC_DIETS = frozenset(["Lean Meat", "Extra Lean Meat"])


def _heybo_low_nutrient_range(constraint: Dict[str, Any]) -> Dict[str, Any]:
    """DB low-nutrient band for Low * / Very Low / lean-meat style diets."""
    rng: Dict[str, Any] = {}
    low_min = constraint.get("low_min")
    low_max = constraint.get("low_max")
    if low_min is not None and pd.notna(low_min):
        rng["Min"] = low_min
    if low_max is not None and pd.notna(low_max):
        rng["Max"] = low_max
    return rng


@lru_cache(maxsize=1)
def load_heybo_nutrition_constraints_from_db() -> Dict[str, Dict[str, Any]]:
    """Load per-nutrient bands from heybo.nutrition_constraint_information."""
    conn = None
    try:
        conn = get_db_connection(DB_CONFIG)
        # Try progressively simpler queries as older DB schemas may lack newer columns.
        query = """
        SELECT nutrient_key, nutrient_unit, low_min, low_max, high_min, high_max,
               balanced_min, balanced_max, customization_min, customization_max,
               priority_order
        FROM heybo.nutrition_constraint_information
        WHERE nutrient_key IS NOT NULL
        """
        try:
            df = pd.read_sql(query, conn)
        except Exception:
            query = """
            SELECT nutrient_key, nutrient_unit, low_min, low_max, high_min, high_max,
                   balanced_min, balanced_max, customization_min, customization_max
            FROM heybo.nutrition_constraint_information
            WHERE nutrient_key IS NOT NULL
            """
            try:
                df = pd.read_sql(query, conn)
            except Exception:
                query = """
                SELECT nutrient_key, nutrient_unit, low_min, low_max, high_min, high_max,
                       balanced_min, balanced_max, customization_max
                FROM heybo.nutrition_constraint_information
                WHERE nutrient_key IS NOT NULL
                """
                try:
                    df = pd.read_sql(query, conn)
                except Exception:
                    query = """
                    SELECT nutrient_key, nutrient_unit, low_min, low_max, high_min, high_max,
                           balanced_min, balanced_max
                    FROM heybo.nutrition_constraint_information
                    WHERE nutrient_key IS NOT NULL
                    """
                    try:
                        df = pd.read_sql(query, conn)
                    except Exception:
                        query = """
                        SELECT nutrient_key, nutrient_unit, low_min, low_max, high_min, high_max
                        FROM heybo.nutrition_constraint_information
                        WHERE nutrient_key IS NOT NULL
                        """
                        df = pd.read_sql(query, conn)
        if df is None or df.empty:
            raise ValueError("heybo.nutrition_constraint_information is empty or unreachable")
        constraints: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            key = str(row.get("nutrient_key"))
            if not key or key == "None":
                continue
            raw_priority = row.get("priority_order") if "priority_order" in df.columns else None
            constraints[key] = {
                "unit": row.get("nutrient_unit"),
                "low_min": row.get("low_min"),
                "low_max": row.get("low_max"),
                "high_min": row.get("high_min"),
                "high_max": row.get("high_max"),
                "balanced_min": row.get("balanced_min") if "balanced_min" in df.columns else None,
                "balanced_max": row.get("balanced_max") if "balanced_max" in df.columns else None,
                "customization_min": (
                    row.get("customization_min") if "customization_min" in df.columns else None
                ),
                "customization_max": (
                    row.get("customization_max") if "customization_max" in df.columns else None
                ),
                "priority_order": (
                    int(raw_priority)
                    if raw_priority is not None and pd.notna(raw_priority)
                    else None
                ),
            }
        return constraints
    finally:
        if conn:
            return_db_connection(conn, DB_CONFIG)


def load_heybo_nutrient_priority_order_from_db() -> tuple:
    """
    Return nutrient keys ordered by ``priority_order`` ascending (1 = highest priority,
    i.e. kept strict longest during relaxation).  Falls back to an empty tuple when the
    DB column is absent or all values are NULL, allowing callers to use the hardcoded
    ``HEYBO_NUTRIENT_RELAXATION_PRIORITY_ORDER`` constant instead.
    """
    try:
        constraints = load_heybo_nutrition_constraints_from_db()
    except Exception:
        return ()
    ranked = [
        (key, info["priority_order"])
        for key, info in constraints.items()
        if info.get("priority_order") is not None
    ]
    if not ranked:
        return ()
    ranked.sort(key=lambda x: x[1])
    return tuple(k for k, _ in ranked)


def convert_heybo_diet_to_nutrient_filters(
    diet_filters: List[str],
    nutrition_constraints: Dict[str, Dict[str, Any]],
    silent: bool = False,
) -> List[Dict[str, Any]]:
    """Turn nutrient-style diet labels into NutrientFilters using DB constraint rows."""
    out: List[Dict[str, Any]] = []
    for diet in diet_filters:
        nutrient_key = DIET_TO_NUTRIENT_MAP.get(diet)
        if not nutrient_key:
            if not silent:
                print(f"WARNING: Unknown diet '{diet}' for nutrient mapping — skipped")
            continue
        if nutrient_key not in nutrition_constraints:
            if not silent:
                print(f"WARNING: No nutrition constraints for nutrient_key '{nutrient_key}' — skipped")
            continue
        constraint = nutrition_constraints[nutrient_key]
        base_meta = {"_diet_converted": True, "_original_diet": diet}
        if diet.startswith("Low "):
            rng = _heybo_low_nutrient_range(constraint)
            if rng:
                out.append({"Nutrient": nutrient_key, "Range": rng, **base_meta})
        elif diet.startswith("High "):
            rng: Dict[str, Any] = {}
            hm = constraint.get("high_min")
            hx = constraint.get("high_max")
            if hm is not None and pd.notna(hm):
                rng["Min"] = hm
            if hx is not None and pd.notna(hx):
                rng["Max"] = hx
            if rng:
                out.append({"Nutrient": nutrient_key, "Range": rng, **base_meta})
        elif diet.endswith(" Free"):
            out.append({"Nutrient": nutrient_key, "Range": {"Max": 0}, **base_meta})
        elif "Very Low" in diet:
            rng = _heybo_low_nutrient_range(constraint)
            if rng:
                out.append({"Nutrient": nutrient_key, "Range": rng, **base_meta})
        elif diet in PROTEIN_SPECIFIC_DIETS:
            rng = _heybo_low_nutrient_range(constraint)
            if rng:
                out.append({"Nutrient": nutrient_key, "Range": rng, **base_meta})
    return out


def infer_heybo_diets_from_nutrient_filters(
    nutrient_filters: List[Dict[str, Any]],
    nutrition_constraints: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Salad `infer_diets_from_nutrients_dynamic` parity — infer High/Low * labels from explicit ranges."""
    inferred: List[str] = []
    for nf in nutrient_filters or []:
        nutrient_name = str(nf.get("Nutrient") or "").strip()
        rng = nf.get("Range") or {}
        min_val = rng.get("Min")
        max_val = rng.get("Max")
        if not nutrient_name or nutrient_name not in nutrition_constraints:
            continue
        constraint = nutrition_constraints[nutrient_name]
        low_max = constraint.get("low_max")
        if max_val is not None and pd.notna(max_val) and low_max is not None and pd.notna(low_max):
            try:
                if float(max_val) <= float(low_max):
                    base = (
                        nutrient_name.replace("_g", "")
                        .replace("_mg", "")
                        .replace("_mcg", "")
                        .replace("_kCal", "")
                    )
                    if float(max_val) == 0:
                        inferred.append(f"{base.title()} Free")
                    else:
                        inferred.append(f"Low {base.title()}")
            except (TypeError, ValueError):
                pass
        high_min = constraint.get("high_min")
        if min_val is not None and pd.notna(min_val) and high_min is not None and pd.notna(high_min):
            try:
                if float(min_val) >= float(high_min):
                    base = (
                        nutrient_name.replace("_g", "")
                        .replace("_mg", "")
                        .replace("_mcg", "")
                        .replace("_kCal", "")
                    )
                    inferred.append(f"High {base.title()}")
            except (TypeError, ValueError):
                pass
    return sorted(set(inferred))
