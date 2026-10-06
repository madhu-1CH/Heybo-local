"""
Heybo pricing: default/extra price lookup, bowl cost calculation.
Always pass price_config from get_heybo_config() (loaded from heybo.preference_filters).

JSON lists align with ingredient_wise_count flow_type=Pricing: tier overflow moves to Extra * keys;
user-requested extras (Ingredients.Extra) use Extra * SKU rows. Call apply_pricing_tier_split_to_bowl
before calculate_heybo_bowl_cost so output and totals match.
"""
from typing import Any, Dict, List, Tuple

import pandas as pd


def _get_heybo_default_price(ingredient, df, category):
    """Get default ai_price for an ingredient in the given category (no Extra variant)."""
    if not ingredient:
        return 0.0
    row = df[(df['ingredient_name'] == ingredient) & (df['category'] == category)]
    if row.empty:
        return 0.0
    p = pd.to_numeric(row.iloc[0].get('ai_price'), errors='coerce')
    return 0.0 if pd.isna(p) else float(p)


def _get_ingredient_price_for_category(ingredient, df, category):
    if not ingredient:
        return 0.0
    rows = df[(df['ingredient_name'] == ingredient) & (df['category'] == category)]
    if rows.empty:
        return 0.0
    p = pd.to_numeric(rows.iloc[0].get('ai_price'), errors='coerce')
    return 0.0 if pd.isna(p) else float(p)


def _tier_default_price(ingredient, df, default_category: str) -> float:
    """
    ai_price for the default-category row only (Proteins, Warm sides, Cold sides, …).
    Used for tier ordering and for default-tier slots — never substitute Extra-category price.
    """
    return _get_ingredient_price_for_category(ingredient, df, default_category)


def _tier_overflow_price(ingredient, df, default_category: str, extra_category: str) -> float:
    """
    Price when the ingredient is past the pricing default-slot limit (tier overflow).
    Uses Extra-category ai_price; if zero, falls back to default-category ai_price.
    """
    default_p = _get_ingredient_price_for_category(ingredient, df, default_category)
    extra_p = _get_ingredient_price_for_category(ingredient, df, extra_category)
    if extra_p == 0.0:
        extra_p = default_p
    return extra_p


def apply_pricing_tier_split_to_bowl(bowl, df, price_config):
    """
    Split Proteins / Warm / Cold vs Extra * using pricing_limits (flow_type=Pricing from DB).

    - Tier competition uses only main-line picks (Proteins, Warm sides, Cold sides) before split.
    - Cheapest default proteins (up to limit) stay in Proteins; overflow appends to Extra Proteins.
    - Highest default-price sides (warm+cold combined, up to limit) stay in main lists; overflow
      appends to Extra Warm sides / Extra Cold sides by side type.
    - Ingredients already placed in Extra * from user Include/Extra are preserved and listed first
      in each Extra list; tier overflow is appended after.

    Sets bowl["_pricing_user_extra_proteins"], ["_pricing_user_extra_warm"], ["_pricing_user_extra_cold"]
    to lists of names that must be charged at Extra * SKU (not tier pair overflow).
    """
    if price_config is None or not isinstance(price_config, dict):
        raise ValueError("apply_pricing_tier_split_to_bowl requires price_config from get_heybo_config()")
    pricing_limits = price_config.get("pricing_limits") or {}
    protein_default_limit = int(pricing_limits.get("Proteins", (0, 0))[1] or 0)
    sides_default_limit = int(pricing_limits.get("Sides", (0, 0))[1] or 0)

    user_extra_p = list(bowl.get("Extra Proteins") or [])
    user_extra_w = list(bowl.get("Extra Warm sides") or [])
    user_extra_c = list(bowl.get("Extra Cold sides") or [])
    bowl["_pricing_user_extra_proteins"] = list(user_extra_p)
    bowl["_pricing_user_extra_warm"] = list(user_extra_w)
    bowl["_pricing_user_extra_cold"] = list(user_extra_c)

    main_p = list(bowl.get("Proteins") or [])
    if main_p:
        pairs = [(ing, _tier_default_price(ing, df, "Proteins")) for ing in main_p]
        pairs.sort(key=lambda x: x[1])
        kept_pairs = pairs[:protein_default_limit]
        keep_set = {p[0] for p in kept_pairs}
        # Match tier slot order (cheapest default-slot first), not pre-split list order — so
        # ``Bowl Name`` / ``_primary_protein_name`` align with ``PriceBreakdown`` slot 1 protein.
        bowl["Proteins"] = [p[0] for p in kept_pairs]
        overflow_p = [ing for ing in main_p if ing not in keep_set]
        bowl["Extra Proteins"] = user_extra_p + overflow_p
    else:
        bowl["Extra Proteins"] = list(user_extra_p)

    warm_m = list(bowl.get("Warm sides") or [])
    cold_m = list(bowl.get("Cold sides") or [])
    tagged = []
    for ing in warm_m:
        tagged.append(("warm", ing, _tier_default_price(ing, df, "Warm sides")))
    for ing in cold_m:
        tagged.append(("cold", ing, _tier_default_price(ing, df, "Cold sides")))
    if tagged:
        tagged.sort(key=lambda x: x[2], reverse=True)
        kept = tagged[:sides_default_limit]
        overflow = tagged[sides_default_limit:]
        kept_w = {ing for kind, ing, _ in kept if kind == "warm"}
        kept_c = {ing for kind, ing, _ in kept if kind == "cold"}
        bowl["Warm sides"] = [ing for ing in warm_m if ing in kept_w]
        bowl["Cold sides"] = [ing for ing in cold_m if ing in kept_c]
        ow = [ing for kind, ing, _ in overflow if kind == "warm"]
        oc = [ing for kind, ing, _ in overflow if kind == "cold"]
        bowl["Extra Warm sides"] = user_extra_w + ow
        bowl["Extra Cold sides"] = user_extra_c + oc
    else:
        bowl["Extra Warm sides"] = list(user_extra_w)
        bowl["Extra Cold sides"] = list(user_extra_c)


def _append_line(lines: List[Dict[str, Any]], *, group: str, ingredient: str, amount: float, note: str = "") -> None:
    lines.append(
        {
            "category": group,
            "ingredient": ingredient,
            "amount": round(amount, 2),
            "note": note,
        }
    )


def calculate_heybo_bowl_cost_with_breakdown(
    bowl,
    df,
    user_input=None,
    price_config=None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Returns (total, breakdown) where breakdown has base_price, lines[], pricing_limits, total.
    Same pricing rules as calculate_heybo_bowl_cost.
    """
    if price_config is None or not isinstance(price_config, dict):
        raise ValueError("price_config from get_heybo_config() is required")
    base_price = price_config.get("base_price")
    if base_price is None:
        raise ValueError(
            "price_config must include base_price (from heybo.preference_filters BYB_Min_Price)"
        )
    lines: List[Dict[str, Any]] = []
    total_cost = float(base_price)
    _append_line(
        lines,
        group="Platform minimum (BYB_Min_Price)",
        ingredient="—",
        amount=float(base_price),
        note="preference_filters",
    )

    pricing_limits = price_config.get("pricing_limits", {})
    protein_default_limit = int(pricing_limits.get("Proteins", (0, 0))[1] or 0)
    sides_default_limit = int(pricing_limits.get("Sides", (0, 0))[1] or 0)

    for ing in list(bowl.get('Bases', [])):
        p = _get_heybo_default_price(ing, df, 'Bases')
        total_cost += p
        _append_line(lines, group="Bases", ingredient=ing, amount=p, note="ai_price")
    for ing in list(bowl.get('Dips', [])):
        p = _get_heybo_default_price(ing, df, 'Dips')
        total_cost += p
        _append_line(lines, group="Dips", ingredient=ing, amount=p, note="ai_price")
    for ing in list(bowl.get('Garnish', [])):
        p = _get_heybo_default_price(ing, df, 'Garnish')
        total_cost += p
        _append_line(lines, group="Garnish", ingredient=ing, amount=p, note="ai_price")
    for ing in list(bowl.get('Sauces', [])):
        p = _get_heybo_default_price(ing, df, 'Sauces')
        total_cost += p
        _append_line(lines, group="Sauces", ingredient=ing, amount=p, note="ai_price")

    user_extra_p = set(bowl.get("_pricing_user_extra_proteins") or [])
    user_extra_w = set(bowl.get("_pricing_user_extra_warm") or [])
    user_extra_c = set(bowl.get("_pricing_user_extra_cold") or [])
    split_applied = "_pricing_user_extra_proteins" in bowl

    if split_applied:
        main_proteins = list(bowl.get("Proteins") or [])
        extra_proteins = list(bowl.get("Extra Proteins") or [])
        tier_overflow_p = [ing for ing in extra_proteins if ing not in user_extra_p]
        tier_pool = main_proteins + tier_overflow_p
        tier_rows = []
        for ing in tier_pool:
            d_slot = _tier_default_price(ing, df, "Proteins")
            o_slot = _tier_overflow_price(ing, df, "Proteins", "Extra Proteins")
            tier_rows.append((ing, d_slot, o_slot))
        tier_rows.sort(key=lambda x: x[1])
        for idx, (ing, default_strict, overflow_price) in enumerate(tier_rows):
            amt = default_strict if idx < protein_default_limit else overflow_price
            total_cost += amt
            if idx < protein_default_limit:
                note = (
                    f"protein tier default slot {idx + 1} of {protein_default_limit} "
                    "(lowest default-category ai_price first; Proteins row only)"
                )
            else:
                note = "protein tier overflow (Extra Proteins ai_price, else Proteins)"
            _append_line(lines, group="Proteins (tier)", ingredient=ing, amount=amt, note=note)
        for ing in extra_proteins:
            if ing in user_extra_p:
                p = _get_ingredient_price_for_category(ing, df, 'Extra Proteins')
                total_cost += p
                _append_line(
                    lines,
                    group="Extra Proteins (user add-on SKU)",
                    ingredient=ing,
                    amount=p,
                    note="Ingredients.Extra — Extra Proteins row ai_price",
                )

        warm_main = list(bowl.get("Warm sides") or [])
        cold_main = list(bowl.get("Cold sides") or [])
        ew_all = list(bowl.get("Extra Warm sides") or [])
        ec_all = list(bowl.get("Extra Cold sides") or [])
        tier_ow = [x for x in ew_all if x not in user_extra_w]
        tier_oc = [x for x in ec_all if x not in user_extra_c]
        side_rows = []
        for ing in warm_main + tier_ow:
            d_slot = _tier_default_price(ing, df, "Warm sides")
            o_slot = _tier_overflow_price(ing, df, "Warm sides", "Extra Warm sides")
            side_rows.append(("Warm sides", ing, d_slot, o_slot))
        for ing in cold_main + tier_oc:
            d_slot = _tier_default_price(ing, df, "Cold sides")
            o_slot = _tier_overflow_price(ing, df, "Cold sides", "Extra Cold sides")
            side_rows.append(("Cold sides", ing, d_slot, o_slot))
        side_rows.sort(key=lambda x: x[2], reverse=True)
        for idx, (side_cat, ing, default_strict, overflow_price) in enumerate(side_rows):
            amt = default_strict if idx < sides_default_limit else overflow_price
            total_cost += amt
            if idx < sides_default_limit:
                note = (
                    f"sides tier default slot {idx + 1} of {sides_default_limit} "
                    "(warm+cold combined; highest default-category ai_price first)"
                )
            else:
                note = "sides tier overflow (Extra Warm/Cold ai_price, else default category)"
            _append_line(lines, group=f"Sides (tier) — {side_cat}", ingredient=ing, amount=amt, note=note)
        for ing in ew_all:
            if ing in user_extra_w:
                p = _get_ingredient_price_for_category(ing, df, 'Extra Warm sides')
                total_cost += p
                _append_line(
                    lines,
                    group="Extra Warm sides (user add-on SKU)",
                    ingredient=ing,
                    amount=p,
                    note="Ingredients.Extra — Extra Warm sides row ai_price",
                )
        for ing in ec_all:
            if ing in user_extra_c:
                p = _get_ingredient_price_for_category(ing, df, 'Extra Cold sides')
                total_cost += p
                _append_line(
                    lines,
                    group="Extra Cold sides (user add-on SKU)",
                    ingredient=ing,
                    amount=p,
                    note="Ingredients.Extra — Extra Cold sides row ai_price",
                )
    else:
        protein_items = []
        for ing in list(bowl.get('Proteins', [])):
            d_slot = _tier_default_price(ing, df, "Proteins")
            o_slot = _tier_overflow_price(ing, df, "Proteins", "Extra Proteins")
            protein_items.append((ing, d_slot, o_slot))
        protein_items.sort(key=lambda x: x[1])
        for idx, (ing, default_strict, overflow_price) in enumerate(protein_items):
            amt = default_strict if idx < protein_default_limit else overflow_price
            total_cost += amt
            note = (
                f"protein tier slot {idx + 1} (default ≤{protein_default_limit}; Proteins row only)"
                if idx < protein_default_limit
                else "protein tier overflow"
            )
            _append_line(lines, group="Proteins (tier, legacy bowl)", ingredient=ing, amount=amt, note=note)
        for ing in list(bowl.get('Extra Proteins', [])):
            p = _get_ingredient_price_for_category(ing, df, 'Extra Proteins')
            total_cost += p
            _append_line(lines, group="Extra Proteins (SKU each)", ingredient=ing, amount=p, note="legacy")

        side_items = []
        for ing in list(bowl.get('Warm sides', [])):
            d_slot = _tier_default_price(ing, df, "Warm sides")
            o_slot = _tier_overflow_price(ing, df, "Warm sides", "Extra Warm sides")
            side_items.append(("Warm sides", ing, d_slot, o_slot))
        for ing in list(bowl.get('Cold sides', [])):
            d_slot = _tier_default_price(ing, df, "Cold sides")
            o_slot = _tier_overflow_price(ing, df, "Cold sides", "Extra Cold sides")
            side_items.append(("Cold sides", ing, d_slot, o_slot))
        side_items.sort(key=lambda x: x[2], reverse=True)
        for idx, (side_cat, ing, default_strict, overflow_price) in enumerate(side_items):
            amt = default_strict if idx < sides_default_limit else overflow_price
            total_cost += amt
            note = f"sides tier slot {idx + 1}" if idx < sides_default_limit else "sides tier overflow"
            _append_line(lines, group=f"Sides (tier, legacy) — {side_cat}", ingredient=ing, amount=amt, note=note)
        for ing in list(bowl.get('Extra Warm sides', [])):
            p = _get_ingredient_price_for_category(ing, df, 'Extra Warm sides')
            total_cost += p
            _append_line(lines, group="Extra Warm sides (SKU each)", ingredient=ing, amount=p, note="legacy")
        for ing in list(bowl.get('Extra Cold sides', [])):
            p = _get_ingredient_price_for_category(ing, df, 'Extra Cold sides')
            total_cost += p
            _append_line(lines, group="Extra Cold sides (SKU each)", ingredient=ing, amount=p, note="legacy")

    total_rounded = round(total_cost, 2)
    breakdown = {
        "base_price": round(float(base_price), 2),
        "pricing_limits": {
            "Proteins_default_slots": protein_default_limit,
            "Sides_default_slots": sides_default_limit,
        },
        "lines": lines,
        "total": total_rounded,
    }
    return total_rounded, breakdown


def calculate_heybo_bowl_cost(bowl, df, user_input=None, price_config=None):
    """
    Heybo BYB: base price + default ai_price for included items.
    If bowl was passed through apply_pricing_tier_split_to_bowl, tier overflow is in Extra * lists
    and is priced with the default/extra pair; user-requested extras use Extra * SKU rows.
    """
    total, _ = calculate_heybo_bowl_cost_with_breakdown(bowl, df, user_input=user_input, price_config=price_config)
    return total


def pop_pricing_metadata(bowl):
    """Remove internal keys used for pricing alignment (before returning bowl to clients)."""
    for k in list(bowl.keys()):
        if k.startswith("_pricing_"):
            bowl.pop(k, None)


def _heybo_is_only_price_active(user_input) -> bool:
    """True when Price is the only active filter (Salad ``_is_only_price_active`` parity)."""
    if not user_input:
        return False
    price = user_input.get("Price", {})
    if not isinstance(price, dict) or (price.get("Min") is None and price.get("Max") is None):
        return False
    has_nutrient_filters = bool(user_input.get("NutrientFilters"))
    has_diet_filters = bool(user_input.get("DietFilters"))
    has_cuisine_filters = bool(user_input.get("CuisineFilters"))
    flavor_prefs = user_input.get("FlavorPreferences") or {}
    has_flavor_filters = bool(
        isinstance(flavor_prefs, dict)
        and any(pref and str(pref).strip() for pref in flavor_prefs.values())
    )
    has_prep_filters = bool(user_input.get("PreparationMethod"))
    has_allergen_filters = bool(user_input.get("AllergenFilters"))
    ing = user_input.get("Ingredients") or {}
    has_ingredient_filters = bool(ing.get("Include") or ing.get("Exclude") or ing.get("Extra"))
    has_light = str(user_input.get("Light", "false")).lower() == "true"
    has_hearty = str(user_input.get("Hearty", "false")).lower() == "true"
    has_sustainable = str(user_input.get("Sustainable", "false")).lower() == "true"
    has_balanced = str(user_input.get("Balanced", "false")).lower() == "true"
    return not (
        has_nutrient_filters
        or has_diet_filters
        or has_cuisine_filters
        or has_flavor_filters
        or has_prep_filters
        or has_allergen_filters
        or has_ingredient_filters
        or has_light
        or has_hearty
        or has_sustainable
        or has_balanced
    )


def normalize_heybo_price_filters(user_input, price_config, global_validations=None):
    """
    Heybo BYB pricing is base (BYB_Min_Price) + ingredient ai_prices — no .50/.90 rounding.
    Costlier-bowls cap uses ``MaxBowlPrice`` (≈ Salad ``MaxSaladPrice``), not ``BYB_Max_Price``
    (≈ Salad ``CYOPriceMax``). Generation uses ±slack via ``_effective_price_bounds``.
    """
    if not isinstance(user_input, dict) or not isinstance(price_config, dict):
        return
    gv = global_validations if isinstance(global_validations, dict) else {}
    price_filter = user_input.get("Price")
    if not isinstance(price_filter, dict):
        return

    db_max = price_config.get("max_bowl_price")
    min_raw = price_filter.get("Min")
    max_raw = price_filter.get("Max")
    user_price_val = max_raw if max_raw is not None else min_raw
    if (
        db_max is not None
        and user_price_val is not None
        and _heybo_is_only_price_active(user_input)
        and float(user_price_val) > float(db_max)
    ):
        cap = float(db_max)
        user_input["_price_original_before_snap"] = {"Min": min_raw, "Max": max_raw}
        user_input["Price"] = {"Min": cap, "Max": cap}
        user_input["_costlier_bowls_capped"] = True
        user_input["_costlier_bowls_db_max"] = cap
        costlier_msg = "These are the most expensive bowls we can generate."
        gv.setdefault("costlier_bowls_message", []).append(costlier_msg)
        gv.setdefault("filter_summary", []).append(
            f"Requested price above maximum generatable bowl price (${cap:.2f}); "
            "generating at MaxBowlPrice"
        )
