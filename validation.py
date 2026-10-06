"""
Heybo validation: preparation method, cuisine, ingredients, diet, allergen, nutrition, composition, price.
"""
import pandas as pd

from .diet import heybo_compatible_diet_tags
from .diet_constants import HEYBO_BOWL_COMPONENT_KEYS, NUTRIENT_NAME_MAP
from .filters import (
    _get_filter_list,
    ingredient_lists_excluded_allergen,
    ingredient_names_excluded_by_allergen,
)


def validate_heybo_preparation_method_matches(selected_ingredients, prep_matched_ingredients, user_input, df=None):
    """Validate preparation method matches and provide explanations."""
    explanations = []
    prep_methods_dict = user_input.get('PreparationMethod', {})
    include_methods = [method for method, include in prep_methods_dict.items() if include]
    exclude_methods = [method for method, include in prep_methods_dict.items() if include is False]
    if not prep_methods_dict:
        return ["No preparation method preferences specified"]
    requested_methods = include_methods
    if not requested_methods:
        explanations.append("No preparation methods selected for inclusion")
    else:
        explanations.append(f"Included ingredients with preparation methods: {', '.join(requested_methods)}")
    if exclude_methods:
        explanations.append(f"Explicitly excluded ingredients with preparation methods: {', '.join(exclude_methods)}")
    matched_ingredients = []
    unmatched_ingredients = []
    if df is not None and requested_methods:
        for ing in selected_ingredients:
            prep_method = df.loc[df['ingredient_name'] == ing, 'preparation_method']
            if not prep_method.empty and prep_method.iloc[0] in requested_methods:
                matched_ingredients.append(ing)
            else:
                unmatched_ingredients.append(ing)
    else:
        prep_matched_set = set(prep_matched_ingredients)
        matched_ingredients = [ing for ing in selected_ingredients if ing in prep_matched_set]
        unmatched_ingredients = [ing for ing in selected_ingredients if ing not in prep_matched_set]
    if matched_ingredients:
        explanations.append(f"Successfully matched {len(matched_ingredients)} ingredients with requested preparation methods ({', '.join(requested_methods)})")
        explanations.append(f"Preparation-matched ingredients: {', '.join(matched_ingredients)}")
    if unmatched_ingredients:
        explanations.append(f"Added {len(unmatched_ingredients)} additional ingredients to meet variety requirements: {', '.join(unmatched_ingredients)}")
    if selected_ingredients:
        match_percentage = (len(matched_ingredients) / len(selected_ingredients)) * 100
        explanations.append(f"Preparation method alignment: {match_percentage:.1f}% of ingredients match requested methods")
    return explanations


def validate_heybo_cuisine_matches(selected_ingredients, cuisine_matched_ingredients, user_input):
    """Validate cuisine matches and provide explanations."""
    explanations = []
    cuisine_filters = _get_filter_list(user_input.get('CuisineFilters'))
    if not cuisine_filters:
        return ["No cuisine preferences specified - selected from all available ingredients"]
    if selected_ingredients:
        explanations.append(f"Selected ingredient(s) match requested cuisine ({', '.join(cuisine_filters)}): {', '.join(selected_ingredients)}")
    else:
        explanations.append(f"No ingredients selected for requested cuisine ({', '.join(cuisine_filters)})")
    return explanations


def validate_heybo_ingredient_selections(
    bowl, included_ingredients, user_input, *, global_validations=None
):
    """Validate ingredient inclusion, exclusion, and extra requests."""
    explanations = []
    all_bowl_ingredients = []
    for key in HEYBO_BOWL_COMPONENT_KEYS:
        items = bowl.get(key, [])
        if isinstance(items, str):
            all_bowl_ingredients.append(items)
        else:
            all_bowl_ingredients.extend(items)
    ingredient_filters = user_input.get('Ingredients', {})
    include_list = ingredient_filters.get('Include', [])
    exclude_list = ingredient_filters.get('Exclude', [])
    extra_list = ingredient_filters.get('Extra', [])
    if include_list:
        successfully_included = [req_ing for req_ing in include_list if req_ing in all_bowl_ingredients]
        missing_ingredients = [ing for ing in include_list if ing not in all_bowl_ingredients]
        if successfully_included:
            explanations.append(
                f"Successfully included requested ingredients: {', '.join(successfully_included)}"
            )
        if missing_ingredients:
            explanations.append(f"Requested ingredients not available or not included: {', '.join(missing_ingredients)}")
    if exclude_list:
        gv = (
            global_validations
            if global_validations is not None
            else (user_input.get("global_validations") or {})
        )
        conflict_override = {
            str(n).strip()
            for n in (gv.get("full_exclude_conflict_ingredient_names") or [])
            if str(n).strip()
        }
        accidentally_included = [
            exc_ing
            for exc_ing in exclude_list
            if exc_ing in all_bowl_ingredients and exc_ing not in conflict_override
        ]
        honored_excludes = [
            exc_ing for exc_ing in exclude_list if exc_ing not in all_bowl_ingredients
        ]
        if accidentally_included:
            explanations.append(f"WARNING: Excluded ingredients found in bowl: {', '.join(accidentally_included)}")
        elif honored_excludes and not conflict_override:
            explanations.append(f"Successfully excluded unwanted ingredients: {', '.join(honored_excludes)}")
    if extra_list:
        repeat_count = {}
        for ing in include_list:
            repeat_count[ing] = repeat_count.get(ing, 0) + 1
        for ing in extra_list:
            repeat_count[ing] = repeat_count.get(ing, 0) + 1
        for ing in set(extra_list):
            expected = repeat_count[ing]
            actual = all_bowl_ingredients.count(ing)
            if actual == expected:
                explanations.append(f"Extra ingredient '{ing}' correctly included {actual} times.")
            else:
                explanations.append(f"WARNING: Extra ingredient '{ing}' expected {expected} times, but found {actual} times in bowl.")
    priority_ingredients = [ing for ing in all_bowl_ingredients if ing in included_ingredients]
    if priority_ingredients:
        explanations.append(f"High-priority ingredients successfully included: {', '.join(priority_ingredients)}")
    if not include_list and not exclude_list and not extra_list:
        explanations.append("No specific ingredient preferences - selected variety from available options")
    return explanations


def validate_heybo_diet_compatibility(bowl, filtered_ingredients, user_input, fallback_categories):
    """Validate diet compatibility and provide explanations."""
    explanations = []
    diet_filters = _get_filter_list(user_input.get('DietFilters'))
    if not diet_filters:
        return ["No dietary restrictions specified"]
    all_bowl_ingredients = set()
    for key in HEYBO_BOWL_COMPONENT_KEYS:
        all_bowl_ingredients.update(bowl.get(key, []))
    compliant_ingredients = []
    non_compliant_ingredients = []
    for ingredient in all_bowl_ingredients:
        ingredient_data = filtered_ingredients[filtered_ingredients['ingredient_name'] == ingredient]
        if not ingredient_data.empty:
            diet_params = ingredient_data.iloc[0].get('diet_parameters', '')
            if pd.notna(diet_params):
                tokens = [
                    t.strip().lower()
                    for t in str(diet_params).replace("/", "|").replace(";", "|").replace(",", "|").split("|")
                    if t.strip()
                ]
                allowed = {
                    tag.strip().lower()
                    for diet in diet_filters
                    for tag in heybo_compatible_diet_tags(diet)
                }
                if any(tok in allowed for tok in tokens):
                    compliant_ingredients.append(ingredient)
                else:
                    non_compliant_ingredients.append(ingredient)
            else:
                non_compliant_ingredients.append(ingredient)
        else:
            non_compliant_ingredients.append(ingredient)
    if compliant_ingredients:
        explanations.append(f"Diet-compliant ingredients ({', '.join(diet_filters)}): {', '.join(compliant_ingredients)}")
    if non_compliant_ingredients:
        explanations.append(f"Non-compliant ingredients included for variety: {', '.join(non_compliant_ingredients)}")
    if fallback_categories:
        explanations.append(f"Fallback applied for categories: {', '.join(fallback_categories)} - limited compliant options available")
    if all_bowl_ingredients:
        compliance_rate = (len(compliant_ingredients) / len(all_bowl_ingredients)) * 100
        explanations.append(f"Overall diet compliance rate: {compliance_rate:.1f}%")
    return explanations


def validate_heybo_allergen_safety(bowl, filtered_ingredients, user_input, catalog_df=None):
    """Validate allergen safety against full catalog (all SKUs per ingredient name)."""
    explanations = []
    allergen_filters = _get_filter_list(user_input.get('AllergenFilters'))
    if not allergen_filters:
        return ["No allergen restrictions specified"]
    catalog = catalog_df if catalog_df is not None else filtered_ingredients
    excluded_names = ingredient_names_excluded_by_allergen(catalog, allergen_filters)
    all_bowl_ingredients = set()
    for key in HEYBO_BOWL_COMPONENT_KEYS:
        all_bowl_ingredients.update(bowl.get(key, []))
    safe_ingredients = []
    potentially_unsafe = []
    for ingredient in all_bowl_ingredients:
        if ingredient in excluded_names:
            potentially_unsafe.append(ingredient)
            continue
        ingredient_data = filtered_ingredients[filtered_ingredients['ingredient_name'] == ingredient]
        if not ingredient_data.empty:
            rows_unsafe = any(
                ingredient_lists_excluded_allergen(raw, allergen_filters)
                for raw in ingredient_data["allergens"]
            )
            if rows_unsafe:
                potentially_unsafe.append(ingredient)
            else:
                safe_ingredients.append(ingredient)
        elif ingredient not in excluded_names:
            safe_ingredients.append(ingredient)
    if safe_ingredients:
        explanations.append(f"Allergen-safe ingredients: {', '.join(safe_ingredients)}")
    if potentially_unsafe:
        explanations.append(f"WARNING: Potentially unsafe ingredients found: {', '.join(potentially_unsafe)}")
        explanations.append(f"These ingredients may contain: {', '.join(allergen_filters)}")
    else:
        explanations.append(f"All ingredients verified safe from excluded allergens: {', '.join(allergen_filters)}")
    return explanations


def validate_heybo_nutritional_targets(bowl, total_nutrients, user_input):
    """Validate nutritional targets and provide explanations."""
    explanations = []
    nutrient_filters = user_input.get('NutrientFilters', [])
    if not nutrient_filters:
        return ["No specific nutritional targets set"]
    for nutrient_filter in nutrient_filters:
        input_name = nutrient_filter.get('Nutrient', '')
        range_filter = nutrient_filter.get('Range', {})
        min_val = range_filter.get('Min')
        max_val = range_filter.get('Max')
        stored_name = NUTRIENT_NAME_MAP.get(input_name)
        if not stored_name and input_name in total_nutrients:
            stored_name = input_name
        if not stored_name:
            explanations.append(f"Warning: Unrecognized nutrient '{input_name}' in filters")
            continue
        actual_value = total_nutrients.get(stored_name, 0)
        if min_val is not None and max_val is not None:
            if min_val <= actual_value <= max_val:
                explanations.append(f"{input_name}: {actual_value} (within target range {min_val}-{max_val})")
            else:
                explanations.append(f"{input_name}: {actual_value} (outside target range {min_val}-{max_val})")
        elif min_val is not None:
            if actual_value >= min_val:
                explanations.append(f"{input_name}: {actual_value} (meets minimum {min_val})")
            else:
                explanations.append(f"{input_name}: {actual_value} (below minimum {min_val})")
        elif max_val is not None:
            if actual_value <= max_val:
                explanations.append(f"{input_name}: {actual_value} (within maximum {max_val})")
            else:
                explanations.append(f"{input_name}: {actual_value} (exceeds maximum {max_val})")
    return explanations


def validate_heybo_bowl_composition(bowl, bowl_type, bowl_weight, max_bowl_weight=900):
    """Validate overall bowl composition and provide explanations."""
    explanations = []
    bases = bowl.get('Bases', [])
    explanations.append(f"Base composition: {', '.join(bases)} ({len(bases)} components)")
    proteins_count = len(bowl.get('Proteins', [])) + len(bowl.get('Extra Proteins', []))
    warm_sides_count = len(bowl.get('Warm sides', [])) + len(bowl.get('Extra Warm sides', []))
    cold_sides_count = len(bowl.get('Cold sides', [])) + len(bowl.get('Extra Cold sides', []))
    dips_count = len(bowl.get('Dips', []))
    garnish_count = len(bowl.get('Garnish', []))
    sauces_count = len(bowl.get('Sauces', []))
    explanations.append(f"Ingredient distribution: {proteins_count} proteins, {warm_sides_count} warm sides, {cold_sides_count} cold sides")
    explanations.append(f"Finishing touches: {dips_count} dip(s), {garnish_count} garnish(es), {sauces_count} sauce(s)")
    cap = int(max_bowl_weight) if max_bowl_weight is not None else 900
    if 1 <= bowl_weight <= cap:
        explanations.append(f"Bowl weight ({bowl_weight}g) within optimal range ({cap}g)")
    else:
        explanations.append(f"Bowl weight ({bowl_weight}g) outside optimal range ({cap}g)")
    return explanations


def validate_heybo_price(
    bowl_cost,
    user_input,
    *,
    effective_min=None,
    effective_max=None,
    price_relaxation_level=0,
):
    """Validate price against user-supplied min/max and provide explanations."""
    explanations = []
    orig = user_input.get("_price_original_before_snap")
    if isinstance(orig, dict) and (orig.get("Min") is not None or orig.get("Max") is not None):
        min_price = orig.get("Min")
        max_price = orig.get("Max")
    else:
        price_filter = user_input.get("Price", {})
        min_price = price_filter.get("Min")
        max_price = price_filter.get("Max")
    use_effective = (
        (effective_min is not None or effective_max is not None)
        and int(price_relaxation_level or 0) > 0
    )
    check_min = effective_min if use_effective else min_price
    check_max = effective_max if use_effective else max_price
    if check_min is not None and check_max is not None:
        if check_min <= bowl_cost <= check_max:
            if use_effective:
                explanations.append(
                    f"Bowl price {bowl_cost:.2f} is within the effective relaxed range "
                    f"({check_min:.2f} to {check_max:.2f}; requested {min_price} to {max_price})"
                )
            else:
                explanations.append(
                    f"Bowl price {bowl_cost:.2f} is within the requested range ({min_price} to {max_price})"
                )
        else:
            if use_effective:
                explanations.append(
                    f"Bowl price {bowl_cost:.2f} is outside the effective relaxed range "
                    f"({check_min:.2f} to {check_max:.2f}; requested {min_price} to {max_price})"
                )
            else:
                explanations.append(
                    f"Bowl price {bowl_cost:.2f} is outside the requested range ({min_price} to {max_price})"
                )
    elif check_min is not None:
        if bowl_cost >= check_min:
            if use_effective:
                explanations.append(
                    f"Bowl price {bowl_cost:.2f} meets the effective relaxed minimum "
                    f"({check_min:.2f}; requested minimum {min_price})"
                )
            else:
                explanations.append(f"Bowl price {bowl_cost:.2f} meets the minimum price ({min_price})")
        else:
            if use_effective:
                explanations.append(
                    f"Bowl price {bowl_cost:.2f} is below the effective relaxed minimum "
                    f"({check_min:.2f}; requested minimum {min_price})"
                )
            else:
                explanations.append(f"Bowl price {bowl_cost:.2f} is below the minimum price ({min_price})")
    elif check_max is not None:
        if bowl_cost <= check_max:
            if use_effective:
                explanations.append(
                    f"Bowl price {bowl_cost:.2f} is within the effective relaxed maximum "
                    f"({check_max:.2f}; requested maximum {max_price})"
                )
            else:
                explanations.append(f"Bowl price {bowl_cost:.2f} is within the maximum price ({max_price})")
        else:
            if use_effective:
                explanations.append(
                    f"Bowl price {bowl_cost:.2f} exceeds the effective relaxed maximum "
                    f"({check_max:.2f}; requested maximum {max_price})"
                )
            else:
                explanations.append(f"Bowl price {bowl_cost:.2f} exceeds the maximum price ({max_price})")
    else:
        explanations.append("No price constraints specified")
    return explanations


def extract_heybo_user_warnings(validations):
    """Extract user warnings from validations."""
    warnings = []
    for section in ['global', 'bowl_specific']:
        if section in validations:
            for key, value in validations[section].items():
                if isinstance(value, list):
                    for explanation in value:
                        if isinstance(explanation, str) and (
                            explanation.startswith("WARNING:") or "not available or not included" in explanation
                        ):
                            warnings.append(explanation.replace("WARNING: ", ""))
    return warnings
