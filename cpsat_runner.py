"""
CYO first-layer fill: generation.py calls this, this calls the solver.

Read this file for the CP-SAT *flow*. Read ``cpsat_feasibility.py`` for the model.

Flow:
  1. Only-mode bowl 1: exact user asks + DB customization mins (no nutrient/price hunt).
  2. Skip if this relaxation fingerprint was already tried.
  3. Merge NutrientFilters + Balanced / Light / Hearty (same suppress rules as random).
  4. Solve with normal slots (ask-aware Extra/Include on overflowing categories only).
  5. If nutrient Mins need more slots, solve again with customization max.
  6. Companion-filter check (flavor / cuisine / prep / price / diet / allergen).
  7. Append accepted bowls. Generation continues with random fill if any remain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import CPSAT_MAX_SOLUTIONS, CPSAT_TIME_LIMIT_SECONDS, dbg_print
from .cpsat_feasibility import (
    _heybo_tight_floor_price,
    build_cpsat_filters_from_preferences,
    filter_cpsat_bowls_by_companion_filters,
    search_feasible_bowls_cpsat,
)
from .diet import (
    get_heybo_relaxed_nutrient_filters,
    heybo_active_nutrient_filter_keys,
)
from .diet_constants import HEYBO_BOWL_COMPONENT_KEYS


@dataclass
class CpsatFillState:
    """Mutable CP-SAT session flags (one per CYO request)."""

    last_fingerprint: Any = None
    used_customization_max: bool = False
    companion_rejected_total: int = 0
    fill_count: int = 0


def _relaxation_fingerprint(
    user_input: dict,
    nutrient_filters_all: list,
    nutrient_relaxation_level: int,
    nutrient_relaxation_levels: dict | None,
    price_relaxation_level: int,
    co2_relaxation_level: int,
    light_hearty_relaxation_level: int,
    cuisine_relaxation_enabled: bool,
    flavor_relaxation_enabled: bool,
    prep_relaxation_enabled: bool,
    variety_context_active: bool,
    only_exact: bool,
) -> tuple:
    """Skip a CP-SAT re-run when price/CO2e/LH/nutrient relaxation has not changed."""
    nf_sig = tuple(
        sorted(
            (
                str(nf.get("Nutrient") or ""),
                str((nf.get("Range") or {}).get("Min")),
                str((nf.get("Range") or {}).get("Max")),
            )
            for nf in (user_input.get("NutrientFilters") or nutrient_filters_all or [])
        )
    )
    return (
        int(user_input.get("price_relaxation_level") or price_relaxation_level or 0),
        int(user_input.get("co2_relaxation_level") or co2_relaxation_level or 0),
        int(user_input.get("light_hearty_relaxation_level") or light_hearty_relaxation_level or 0),
        int(user_input.get("balanced_relaxation_level") or 0),
        int(nutrient_relaxation_level or 0),
        tuple(sorted((nutrient_relaxation_levels or {}).items())),
        nf_sig,
        bool(user_input.get("_numeric_customization_stage")),
        bool(user_input.get("_relax_cuisine_filter") or cuisine_relaxation_enabled),
        bool(user_input.get("_relax_flavor_filter") or flavor_relaxation_enabled),
        bool(user_input.get("_relax_preparation_method_filter") or prep_relaxation_enabled),
        bool(variety_context_active),
        bool(only_exact),
    )


def _track_protein_sauce(bowl: dict, used_protein_sauce_combos: set) -> None:
    proteins = list(bowl.get("Proteins") or []) + list(bowl.get("Extra Proteins") or [])
    protein = str(proteins[0]).strip() if proteins else ""
    sauces = list(bowl.get("Sauces") or [])
    sauce = str(sauces[0]).strip() if sauces else ""
    if protein and sauce:
        used_protein_sauce_combos.add(f"{protein}_{sauce}")
    elif sauce:
        used_protein_sauce_combos.add(sauce)
    elif protein:
        used_protein_sauce_combos.add("Lulu-BYB")


def _bowl_name_set(bowl: dict) -> frozenset:
    names: list[str] = []
    for key in HEYBO_BOWL_COMPONENT_KEYS:
        names.extend(bowl.get(key) or [])
    return frozenset(str(n).strip() for n in names if n and str(n).strip())


def run_heybo_cpsat_fill(
    reason: str,
    *,
    state: CpsatFillState,
    enabled: bool,
    original_only_mode: bool,
    variety_context_active: bool,
    target_bowls: int,
    heybo_bowls: list,
    user_input: dict,
    df,
    heybo_cfg: dict,
    global_validations: dict,
    price_relaxation_level: int,
    co2_relaxation_level: int,
    light_hearty_relaxation_level: int,
    nutrient_relaxation_level: int,
    nutrient_relaxation_levels: dict | None,
    nutrient_filters_all: list,
    numeric_nutrient_filters: list,
    cuisine_relaxation_enabled: bool,
    flavor_relaxation_enabled: bool,
    prep_relaxation_enabled: bool,
    numeric_expand_for_min: bool,
    chase_binding_mins: bool,
    active_nutrient_keys,
    bases: list,
    proteins: list,
    extra_proteins: list,
    warm_sides: list,
    extra_warm_sides: list,
    cold_sides: list,
    extra_cold_sides: list,
    dips: list,
    garnishes: list,
    sauces: list,
    include_list: list,
    extra_list: list,
    exclude_list: list,
    omit_categories: set,
    normal_flow_category_limits: dict,
    customization_flow_extra_category_max: dict,
    incompatible_pairs: dict,
    filtered_ingredients,
    flavor_preferences: dict,
    previous_bowls: set,
    used_protein_sauce_combos: set,
    customization_category_limits: dict | None = None,
) -> int:
    """
    Try to fill remaining bowls with CP-SAT. Returns how many bowls were added.

    Called first as ``reason='initial'``, then again when random-loop relaxation
    steps up (price / CO2e / Light-Hearty / nutrients).
    """
    if not enabled:
        return 0
    only_exact = bool(original_only_mode) and not bool(variety_context_active)
    remaining = max(0, target_bowls - len(heybo_bowls))
    if remaining <= 0:
        return 0
    if only_exact:
        remaining = 1
        dbg_print(
            "DEBUG: CP-SAT Only exact bowl 1 — user asks + customization mins; "
            "nutrient/price mins not enforced"
        )

    user_input["_heybo_cfg"] = heybo_cfg or {}
    user_input["price_relaxation_level"] = price_relaxation_level
    user_input["co2_relaxation_level"] = co2_relaxation_level
    user_input["light_hearty_relaxation_level"] = light_hearty_relaxation_level
    if variety_context_active:
        user_input["_only_mode_variety_active"] = True
        user_input["_heybo_variety_bowls_active"] = True

    fingerprint = _relaxation_fingerprint(
        user_input,
        nutrient_filters_all,
        nutrient_relaxation_level,
        nutrient_relaxation_levels,
        price_relaxation_level,
        co2_relaxation_level,
        light_hearty_relaxation_level,
        cuisine_relaxation_enabled,
        flavor_relaxation_enabled,
        prep_relaxation_enabled,
        variety_context_active,
        only_exact,
    )
    if fingerprint == state.last_fingerprint:
        return 0
    state.last_fingerprint = fingerprint

    existing_nf = list(
        user_input.get("NutrientFilters")
        or numeric_nutrient_filters
        or nutrient_filters_all
        or []
    )
    if nutrient_relaxation_levels or nutrient_relaxation_level:
        existing_nf = get_heybo_relaxed_nutrient_filters(
            existing_nf,
            relaxation_level=nutrient_relaxation_level,
            relaxation_levels=nutrient_relaxation_levels,
        )
    cpsat_nf, cpsat_weight, cpsat_sources = build_cpsat_filters_from_preferences(
        user_input,
        existing_nutrient_filters=existing_nf,
    )
    if not cpsat_sources:
        cpsat_sources = ["CYO"]

    if reason == "initial" and cpsat_sources:
        global_validations.setdefault("filter_summary", []).append(
            "CP-SAT criteria from: " + ", ".join(cpsat_sources)
        )
    dbg_print(
        f"DEBUG: CP-SAT run ({reason}); sources={cpsat_sources}; "
        f"nutrients={heybo_active_nutrient_filter_keys(cpsat_nf)}; "
        f"weight={cpsat_weight}; remaining={remaining}"
    )
    if reason != "initial":
        global_validations.setdefault("filter_summary", []).append(
            f"CP-SAT re-run after {reason} (filling remaining bowls with relaxed constraints)"
        )

    def _uniq(*lists):
        seen: set[str] = set()
        out: list[str] = []
        for lst in lists:
            for item in lst or []:
                name = str(item).strip() if item else ""
                if not name or name in seen:
                    continue
                seen.add(name)
                out.append(name)
        return out

    cpsat_pools = {
        "Bases": _uniq(bases),
        "Proteins": _uniq(proteins, extra_proteins),
        "Warm sides": _uniq(warm_sides, extra_warm_sides),
        "Cold sides": _uniq(cold_sides, extra_cold_sides),
        "Dips": _uniq(dips),
        "Garnish": _uniq(garnishes),
        "Sauces": _uniq(sauces),
    }

    before = len(heybo_bowls)
    reject_this_run = 0
    # Extra slots overflow to paid Extra * SKUs. When Max ≈ BYB min that
    # second pass only produces price-rejects (14.60 / 17.90) and burns time.
    skip_customization_max = (not only_exact) and _heybo_tight_floor_price(user_input)
    if skip_customization_max:
        dbg_print(
            "[CP-SAT] skipping customization_max pass: Price Max is at BYB floor "
            "(Extra slots would fail companion price)"
        )
    expand_passes = (
        (False,)
        if only_exact or skip_customization_max
        else (
            (False, True)
            if (numeric_expand_for_min or chase_binding_mins)
            else (False,)
        )
    )
    cuisine_matched = user_input.get("_cuisine_matched_ingredient_names") or set()
    for expand_max in expand_passes:
        remaining = max(0, target_bowls - len(heybo_bowls))
        if only_exact:
            remaining = max(0, 1 - len(heybo_bowls))
        if remaining <= 0:
            break
        blocked = set(previous_bowls or set())
        prior_ps = set(used_protein_sauce_combos or set())
        variety = bool(user_input.get("_only_mode_variety_active") or variety_context_active)
        cpsat_include = [] if variety else list(include_list or [])
        cpsat_extra = [] if variety else list(extra_list or [])
        apriori_seed = list(include_list or []) + list(extra_list or [])

        cpsat_bowls, cpsat_msg = search_feasible_bowls_cpsat(
            df=df,
            user_input=user_input,
            nutrient_filters=cpsat_nf,
            category_pools=cpsat_pools,
            normal_flow_category_limits=normal_flow_category_limits,
            customization_flow_extra_category_max=customization_flow_extra_category_max,
            incompatible_pairs=incompatible_pairs,
            include_list=cpsat_include,
            extra_list=cpsat_extra,
            exclude_list=exclude_list,
            omit_categories=omit_categories,
            max_solutions=min(remaining, 1 if only_exact else CPSAT_MAX_SOLUTIONS),
            time_limit_seconds=CPSAT_TIME_LIMIT_SECONDS,
            expand_to_customization_max=expand_max,
            weight_range=None if only_exact else cpsat_weight,
            previous_bowls=blocked,
            previous_protein_sauce_combos=prior_ps,
            filtered_ingredients=filtered_ingredients,
            apriori_suggestions=global_validations.get("apriori_suggestions") or [],
            apriori_seed_ingredients=apriori_seed,
            flavor_preferences=flavor_preferences,
            cuisine_matched_names=cuisine_matched,
            only_exact_bowl=only_exact,
            customization_category_limits=customization_category_limits,
        )
        if cpsat_msg and reason == "initial":
            global_validations.setdefault("filter_summary", []).append(cpsat_msg)
        overflow_notices = list(user_input.get("_cpsat_category_limit_notices") or [])
        if overflow_notices:
            for note in overflow_notices:
                if note not in global_validations.setdefault("category_limit_notices", []):
                    global_validations["category_limit_notices"].append(note)

        cpsat_rejected = 0
        if cpsat_bowls:
            cpsat_bowls, cpsat_rejected = filter_cpsat_bowls_by_companion_filters(
                cpsat_bowls,
                user_input=user_input,
                df=df,
                flavor_preferences=flavor_preferences,
                cuisine_matched_names=cuisine_matched,
                include_list=include_list,
                extra_list=extra_list,
                filtered_ingredients=filtered_ingredients,
                skip_nutrient_keys=set(active_nutrient_keys or []),
                nutrient_filters=cpsat_nf,
                only_exact_bowl=only_exact,
                incompatible_pairs=incompatible_pairs,
            )
            reject_this_run += cpsat_rejected
            state.companion_rejected_total += cpsat_rejected

        if not cpsat_bowls:
            continue
        if expand_max:
            state.used_customization_max = True
        remaining = max(0, target_bowls - len(heybo_bowls))
        for bowl in cpsat_bowls[:remaining]:
            heybo_bowls.append(bowl)
            previous_bowls.add(_bowl_name_set(bowl))
            _track_protein_sauce(bowl, used_protein_sauce_combos)
        dbg_print(
            f"DEBUG: CP-SAT ({reason}) supplied bowls "
            f"(customization_max={expand_max}, companion_rejected={cpsat_rejected}); "
            f"{len(heybo_bowls)}/{target_bowls} filled"
        )

    added = len(heybo_bowls) - before
    state.fill_count += added
    global_validations["_cpsat_bowls_generated"] = state.fill_count
    if reject_this_run:
        global_validations["_cpsat_companion_rejected"] = state.companion_rejected_total
    if reason == "initial":
        if state.companion_rejected_total:
            global_validations.setdefault("filter_summary", []).append(
                f"CP-SAT: {state.companion_rejected_total} solver bowl(s) rejected by "
                "companion filters (flavor, cuisine, prep, price, weight, diet, etc.)"
            )
        remaining = max(0, target_bowls - len(heybo_bowls))
        if remaining > 0:
            global_validations.setdefault("filter_summary", []).append(
                f"CP-SAT filled {len(heybo_bowls)} bowl(s); continuing random search "
                "with existing relaxation for the rest"
            )
            dbg_print(
                "DEBUG: CP-SAT did not fill the request; random loop + relaxation continues"
            )
        elif heybo_bowls:
            if state.companion_rejected_total:
                global_validations.setdefault("filter_summary", []).append(
                    "CP-SAT filled the request after companion-filter validation; "
                    "random search not needed for remaining bowls"
                )
            else:
                global_validations.setdefault("filter_summary", []).append(
                    "CP-SAT filled the request using current (unrelaxed) nutrient ranges; "
                    "random search / nutrient relaxation not needed"
                )
            global_validations["category_limits_used"] = (
                "customization" if state.used_customization_max else "normal"
            )
    return added
