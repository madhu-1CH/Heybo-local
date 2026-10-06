"""
Heybo bowl utilities: weight, total nutrients, CO2e, incompatible pairs, compatibility check.
"""
import pandas as pd

from .diet_constants import HEYBO_BOWL_COMPONENT_KEYS, NUTRIENT_COLUMNS


def calculate_heybo_bowl_weight(bowl, df):
    """Calculate total weight of the Heybo bowl."""
    total_weight = 0.0

    def safe_weight(item, df):
        if not item:
            return 0.0
        weight = df.loc[df['ingredient_name'] == item, 'serving_amount_per_portion_in_grams']
        return pd.to_numeric(weight.iloc[0], errors='coerce') if not weight.empty else 0.0

    for component_name in HEYBO_BOWL_COMPONENT_KEYS:
        for ingredient in bowl.get(component_name, []):
            total_weight += safe_weight(ingredient, df)
    return total_weight


def calculate_heybo_total_nutrients(bowl, df):
    """Calculate total nutrients for a Heybo bowl."""
    total_nutrients = {}

    def safe_nutrient_value(item, nutrient, df):
        try:
            value = df.loc[df['ingredient_name'] == item, nutrient]
            if not value.empty:
                val = value.iloc[0]
                numeric_val = pd.to_numeric(val, errors='coerce')
                return numeric_val if pd.notna(numeric_val) else 0.0
            return 0.0
        except Exception as e:
            print(f"Error processing {item} for {nutrient}: {e}")
            return 0.0

    all_nutrients = [col for col in df.columns if col in NUTRIENT_COLUMNS]
    for nutrient in all_nutrients:
        total_nutrients[nutrient] = 0.0
    components = [(k, bowl.get(k, [])) for k in HEYBO_BOWL_COMPONENT_KEYS]
    for component_name, component_items in components:
        if isinstance(component_items, str):
            items = [item.strip() for item in component_items.split(',')]
        else:
            items = component_items if isinstance(component_items, list) else []
        for item in items:
            for nutrient in all_nutrients:
                total_nutrients[nutrient] += safe_nutrient_value(item, nutrient, df)
    return {nutrient: round(value, 2) for nutrient, value in total_nutrients.items()}


def calculate_heybo_total_co2e(bowl, df):
    """Calculate total CO₂ emissions for a Heybo bowl."""
    total = 0.0

    def safe_co2e(ingredient):
        if not ingredient:
            return 0.0
        val = df.loc[df['ingredient_name'] == ingredient, 'co2e_values_per_serving']
        return float(val.iloc[0]) if not val.empty else 0.0

    components = [bowl.get(k, []) for k in HEYBO_BOWL_COMPONENT_KEYS]
    for component in components:
        for ingredient in component:
            if isinstance(ingredient, list):
                for ing in ingredient:
                    total += safe_co2e(ing)
            else:
                total += safe_co2e(ingredient)
    return round(total, 2)


def get_heybo_incompatible_pairs(df):
    """Create a dictionary of incompatible ingredient pairs from the Heybo database."""
    incompatible_pairs = {}
    for _, row in df.iterrows():
        ingredient = row['ingredient_name']
        incompatible_str = row.get('does_not_go_well_with_ingredients', '')
        if pd.isna(incompatible_str) or not incompatible_str.strip():
            continue
        incompatible_list = {i.strip() for i in incompatible_str.split(',') if i.strip()}
        incompatible_pairs[ingredient] = incompatible_list
    return incompatible_pairs


def is_heybo_bowl_compatible(bowl, incompatible_pairs, user_included=None):
    """Check if all ingredients in the Heybo bowl are compatible with each other."""
    print("\n=== Checking Heybo bowl compatibility ===")
    if user_included is None:
        user_included = set()
    all_ingredients = set()
    for component_name in HEYBO_BOWL_COMPONENT_KEYS:
        all_ingredients.update(bowl.get(component_name, []))
    for ingredient in all_ingredients:
        if ingredient not in incompatible_pairs:
            continue
        incompatible_with = incompatible_pairs[ingredient]
        conflicts = all_ingredients & incompatible_with
        conflicts = {c for c in conflicts if not (ingredient in user_included and c in user_included)}
        if conflicts:
            print(f"Conflict found: {ingredient} does not go well with {', '.join(conflicts)}")
            return False
        else:
            print(f"{ingredient} has no conflicts with other ingredients")
    return True
