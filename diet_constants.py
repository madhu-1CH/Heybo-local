"""
Structural Heybo constants: category wiring and dataframe column names.
Business rules (limits, pricing, flavor bands, diet→nutrient ranges) come from the database.
"""

# Maps extra-ingredient categories to their base category for pooling
HEYBO_EXTRA_CATEGORIES = {
    "Extra Proteins": "Proteins",
    "Extra Warm sides": "Warm sides",
    "Extra Cold sides": "Cold sides",
}

# All ingredient-list keys on a bowl (for totals, compatibility, JSON); extras sit next to their base category.
HEYBO_BOWL_COMPONENT_KEYS = (
    "Bases",
    "Proteins",
    "Extra Proteins",
    "Warm sides",
    "Extra Warm sides",
    "Cold sides",
    "Extra Cold sides",
    "Dips",
    "Garnish",
    "Sauces",
)

# Columns summed when reporting bowl nutrients
NUTRIENT_COLUMNS = {
    "calories_kCal",
    "carbs_g",
    "protein_g",
    "total_fat_g",
    "saturated_fat_g",
    "trans_fat_g",
    "cholesterol_mg",
    "sodium_mg",
    "fiber_g",
    "sugar_g",
    "added_sugar_g",
    "calcium_mg",
    "iron_mg",
    "potassium_mg",
    "vitamin_d_mcg",
    "phosphorus_mg",
}

# User-facing nutrient labels → column names in ingredients_details
NUTRIENT_NAME_MAP = {
    "Calories": "calories_kCal",
    "Protein": "protein_g",
    "Total Fat": "total_fat_g",
    "Cholesterol": "cholesterol_mg",
    "Carbohydrates": "carbs_g",
    "Fiber": "fiber_g",
    "Sugar": "sugar_g",
    "Sodium": "sodium_mg",
    "Iron": "iron_mg",
    "Saturated Fat": "saturated_fat_g",
    "Trans Fat": "trans_fat_g",
    "Calcium": "calcium_mg",
    "Vitamin D": "vitamin_d_mcg",
    "Potassium": "potassium_mg",
    "Added Sugar": "added_sugar_g",
}

# Nutrient relaxation priority (index 0 = highest priority, relaxed last).
# Rearrange this tuple when product finalizes priority; unknown active filters relax first.
HEYBO_NUTRIENT_RELAXATION_PRIORITY_ORDER = (
    "calories_kCal",
    "protein_g",
    "carbs_g",
    "total_fat_g",
    "fiber_g",
    "sugar_g",
    "sodium_mg",
    "saturated_fat_g",
    "cholesterol_mg",
    "calcium_mg",
    "potassium_mg",
    "iron_mg",
    "vitamin_d_mcg",
    "phosphorus_mg",
)

# Order used when resolving diet_parameters fallback per category
DIET_HIERARCHY = [
    "Vegan",
    "Vegetarian",
    "Pescatarian",
    "Paleo",
    "Anti-inflammatory",
    "No Diet",
]
