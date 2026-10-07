"""
User-facing `message_to_user` for Heybo BYB and signatures.

BYB ``message_to_user`` is composed entirely in this module (all bowl counts).
Post-steps normalize opener phrasing for Include / “great variety” where needed.
"""

import re
from typing import Any, List, Optional, Set

from .diet import RELAXATION_PERCENTAGES, heybo_active_nutrient_filter_keys
from .diet_constants import HEYBO_BOWL_COMPONENT_KEYS
from .nutrition_constraints import DIET_TO_NUTRIENT_MAP

# Must match `heybo.generation.HEYBO_BOWLS_PER_PAGE` (avoid importing generation → circular import).
_HEYBO_BYB_FULL_PAGE = 5
# Must match `heybo.generation.PRICE_RELAXATION_SLACKS`.
# First pass $0.50, then +$1 … +$6. The next level removes the maximum.
_HEYBO_PRICE_RELAXATION_SLACKS = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


# Shown when Only BYB generates a single bowl (legacy / partial page).
_HEYBO_ONLY_MODE_MESSAGE = (
    "Your bowl was generated with your requested ingredients — feel free to customize it."
)
_HEYBO_ONLY_MODE_ADJUSTED_MESSAGE = (
    "Your bowl was generated from your requested ingredients within category limits — "
    "feel free to customize it."
)
_HEYBO_ONLY_MODE_VARIETY_OPENER = (
    "Your first bowl includes exactly the ingredients you requested."
)
_HEYBO_ONLY_MODE_VARIETY_ADJUSTED_OPENER = (
    "Your first bowl prioritizes your requested ingredients within category limits."
)
_HEYBO_ONLY_MODE_VARIETY_SUFFIX = (
    "Bowls 2–5 offer variety and may include additional ingredients while honoring your filters — "
    "feel free to customize any bowl."
)


def _heybo_user_requested_a_base(ui: Optional[dict], bowls: Optional[List[dict]] = None) -> bool:
    """True when Include/Extra names a base that appears on the exact Only bowl (bowl 1)."""
    if not isinstance(ui, dict):
        return False
    requested: Set[str] = set()
    ing = ui.get("Ingredients") or {}
    if isinstance(ing, dict):
        for key in ("Include", "Extra"):
            for x in ing.get(key) or []:
                if x is not None and str(x).strip():
                    requested.add(str(x).strip().lower())
    if not requested:
        return False
    bowl0 = (bowls or [None])[0] if bowls else None
    if isinstance(bowl0, dict):
        for b in bowl0.get("Bases") or []:
            if b is not None and str(b).strip().lower() in requested:
                return True
    return False


def _heybo_clean_note_text(text: str) -> str:
    c = (text or "").strip()
    low = c.lower()
    if low.startswith("- note:"):
        c = c[7:].strip()
    elif low.startswith("note:"):
        c = c[5:].strip()
    return c


def _heybo_bowl_type_label(ui: Optional[dict], bowls: Optional[List[dict]] = None) -> str:
    raw = None
    if bowls and isinstance(bowls[0], dict):
        raw = bowls[0].get("Bowl Type") or bowls[0].get("BowlType")
    if not raw and isinstance(ui, dict):
        raw = ui.get("Bowl Type") or ui.get("BowlType")
    return str(raw or "bowl").replace("_", " ").strip().lower() or "bowl"


def _heybo_only_mode_portion_cap_notes(ui: Optional[dict]) -> List[str]:
    """Only-mode category cap lines (e.g. 2 dips requested, max 1)."""
    out: List[str] = []
    gv = (ui or {}).get("global_validations") or {}
    for n in gv.get("category_limit_notices") or []:
        if not isinstance(n, str) or not n.strip():
            continue
        low = n.lower()
        if "you chose" in low and "allows at most" in low:
            cleaned = _heybo_clean_note_text(n)
            if cleaned and cleaned not in out:
                out.append(cleaned)
    return out


def _heybo_only_mode_min_fill_notes(
    ui: Optional[dict], bowls: Optional[List[dict]]
) -> List[str]:
    """Salad-parity: catalog fill to meet a category minimum (usually Bases)."""
    notes: List[str] = []
    bowl1 = bowls[0] if bowls else None
    sources: List[str] = []
    if isinstance(bowl1, dict):
        sources.extend(bowl1.get("message") or [])
        sources.extend(
            (bowl1.get("Validations") or {})
            .get("bowl_specific", {})
            .get("ingredient_explanations")
            or []
        )
    for m in sources:
        if (
            isinstance(m, str)
            and m.startswith("We added")
            and "requires at least" in m
        ):
            cleaned = _heybo_clean_note_text(m)
            if cleaned and cleaned not in notes:
                notes.append(cleaned)
    if notes:
        return notes
    if not bowls or _heybo_user_requested_a_base(ui, bowls):
        return notes
    bowl0 = bowls[0] if isinstance(bowls[0], dict) else None
    bases = [x for x in ((bowl0 or {}).get("Bases") or []) if x]
    if bases:
        label = _heybo_bowl_type_label(ui, bowls)
        notes.append(f"We added Bases since each {label} requires at least one.")
    return notes


def _heybo_compose_only_mode_message(ui: dict, bowls: List[dict]) -> str:
    """Salad-parity Only copy: opener, optional Note, then variety suffix."""
    is_variety = len(bowls) > 1 or bool(ui.get("_heybo_variety_bowls_active"))
    min_fill = _heybo_only_mode_min_fill_notes(ui, bowls)
    limit_notes = _heybo_only_mode_portion_cap_notes(ui)
    adjusted = bool(min_fill or limit_notes)
    notes: List[str] = []
    for n in limit_notes + min_fill:
        if n not in notes:
            notes.append(n)

    if is_variety:
        opener = (
            _HEYBO_ONLY_MODE_VARIETY_ADJUSTED_OPENER
            if adjusted
            else _HEYBO_ONLY_MODE_VARIETY_OPENER
        )
        if notes:
            return f"{opener} Note: {' '.join(notes)} {_HEYBO_ONLY_MODE_VARIETY_SUFFIX}"
        return f"{opener} {_HEYBO_ONLY_MODE_VARIETY_SUFFIX}"

    base = (
        _HEYBO_ONLY_MODE_ADJUSTED_MESSAGE if adjusted else _HEYBO_ONLY_MODE_MESSAGE
    )
    if notes:
        return f"{base} Note: {' '.join(notes)}"
    return base


def _is_truthy_only_flag(ui: dict) -> bool:
    ing = ui.get("Ingredients") or {}
    if ing.get("Only") is True or ing.get("only") is True:
        return True
    return ui.get("Only") is True or ui.get("only") is True


def _heybo_line_blocked_from_client_message(line: str) -> bool:
    """
    Pipeline / merge diagnostics must not appear in message_to_user.
    Salad uses process_validation_messages_for_user for friendly copy, not raw filter logs.
    """
    low = (line or "").lower()
    if "strictest combined bounds" in low:
        return True
    if "merged" in low and "constraints" in low:
        return True
    if low.startswith("applied diet filters:"):
        return True
    if "skipped nutrient filter" in low:
        return True
    if any(
        p in low
        for p in (
            "capped min to db",
            "raised min to",
            "added max from db",
            "adjusted so min <= max",
        )
    ):
        return True
    return False


def _append_category_limit_notices(msg, global_validations):
    """Append user-facing category cap messages (e.g. Only mode vs customization max)."""
    if not global_validations:
        return msg
    notices = [
        n.strip()
        for n in (global_validations.get("category_limit_notices") or [])
        if isinstance(n, str) and n.strip()
    ]
    if not notices:
        return msg
    blob = " ".join(notices)
    if not (msg or "").strip():
        return f"Note: {blob}"
    if " - Note:" in msg:
        return f"{msg.rstrip(' .')}, and {blob}"
    return f"{msg.rstrip()} - Note: {blob}"


def _dedupe_comma_separated_pref_clause(inner: str) -> str:
    """Remove duplicate segments (case-insensitive) after splitting on commas."""
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    seen = set()
    out = []
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return ", ".join(out)


def _dedupe_salad_your_preferences_clause(message: str) -> str:
    """
    Salad can repeat the same diet/filter phrase in the opener; collapse duplicates in the
    '… your X preferences' segment only (leave the Note section unchanged).
    """
    if not message:
        return message
    note_idx = message.find(" - Note:")
    head = message[:note_idx] if note_idx != -1 else message
    tail = message[note_idx:] if note_idx != -1 else ""
    for before, after in (
        (" for your ", " preferences"),
        (" that match your ", " preferences"),
    ):
        i = head.find(before)
        if i == -1:
            continue
        j = head.find(after, i + len(before))
        if j == -1:
            continue
        inner_start = i + len(before)
        inner = head[inner_start:j]
        deduped = _dedupe_comma_separated_pref_clause(inner)
        head = head[:inner_start] + deduped + head[j:]
        break
    return head + tail


def _all_bowls_meet_applied_nutrient_filters(bowls, nutrient_filters):
    """True if every bowl's Total Nutrients satisfies each NutrientFilters range."""
    if not bowls or not nutrient_filters:
        return False
    for bowl in bowls:
        tn = bowl.get("Total Nutrients") or {}
        for nf in nutrient_filters:
            name = (nf.get("Nutrient") or "").strip()
            if not name:
                continue
            rng = nf.get("Range") or {}
            val = tn.get(name)
            if val is None:
                return False
            try:
                v = float(val)
            except (TypeError, ValueError):
                return False
            mn, mx = rng.get("Min"), rng.get("Max")
            if mn is not None:
                try:
                    if v < float(mn) - 1e-6:
                        return False
                except (TypeError, ValueError):
                    return False
            if mx is not None:
                try:
                    if v > float(mx) + 1e-6:
                        return False
                except (TypeError, ValueError):
                    return False
    return True


def _heybo_numeric_range_drift(orig: dict, applied: dict) -> bool:
    """True if Min/Max on a nutrient row changed between user snapshot and merged Range."""
    if not isinstance(orig, dict) or not isinstance(applied, dict):
        return False
    for key in ("Min", "Max"):
        o, a = orig.get(key), applied.get(key)
        if o is None and a is None:
            continue
        try:
            if o is None or a is None:
                return True
            if abs(float(o) - float(a)) > 1e-6:
                return True
        except (TypeError, ValueError):
            if o != a:
                return True
    return False


def _heybo_has_original_range_snapshot(orig: dict) -> bool:
    """True when _original_range records at least one bound the user or merge captured."""
    if not isinstance(orig, dict):
        return False
    return orig.get("Min") is not None or orig.get("Max") is not None


def _heybo_bound_unchanged(expected, actual) -> bool:
    """True when applied bound equals the user's original (or user did not set one)."""
    if actual is None:
        return expected is None
    if expected is None:
        return False
    try:
        return abs(float(expected) - float(actual)) <= 1e-6
    except (TypeError, ValueError):
        return expected == actual


def _heybo_is_internal_range_bound_fill_in(orig: dict, applied: dict) -> bool:
    """
    True when alignment only auto-added the missing bound (Min-only → Max, or Max-only → Min).
    Not user-facing adjustment — do not show guideline/optimization notes for this alone.
    """
    if not isinstance(orig, dict) or not isinstance(applied, dict):
        return False
    o_min, o_max = orig.get("Min"), orig.get("Max")
    a_min, a_max = applied.get("Min"), applied.get("Max")
    if o_min is not None and o_max is None:
        if _heybo_bound_unchanged(o_min, a_min) and a_max is not None:
            return True
    if o_max is not None and o_min is None:
        if _heybo_bound_unchanged(o_max, a_max) and a_min is not None:
            return True
    return False


def _heybo_format_nutrient_bound(v) -> str:
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else f"{f:g}"
    except (TypeError, ValueError):
        return str(v)


def _heybo_nutrient_filter_user_display_range(nf: dict) -> dict:
    """
    Nutrient Min/Max for message_to_user.

    Normally shows the user's original bounds (from ``_original_range``) so the
    message reflects what they asked for.

    Exception — hard floor / ceiling adjustments: when ``customization_min`` raised
    the user's Min, or ``customization_max`` capped the user's Max, the effective
    bound is shown instead.  Saying "minimum 100" when we're actually enforcing
    "minimum 200" is misleading, especially since the Note already mentions the
    optimisation.
    """
    rng = dict((nf or {}).get("Range") or {})
    orig = (nf or {}).get("_original_range")
    if isinstance(orig, dict) and _heybo_has_original_range_snapshot(orig):
        out = {}
        orig_min = orig.get("Min")
        orig_max = orig.get("Max")
        eff_min = rng.get("Min")
        eff_max = rng.get("Max")

        # When customization_min raised the user's Min, omit both Min and Max from
        # the display — the Note already says the range was optimized, so showing any
        # numbers in the opener would be misleading.  The opener will just name the
        # nutrient (e.g. "Low Calorie") without any bounds.
        min_was_floored = False
        if orig_min is not None:
            try:
                if eff_min is not None and float(eff_min) > float(orig_min):
                    min_was_floored = True   # floor raised it — drop all bounds from display
                else:
                    out["Min"] = orig_min
            except (TypeError, ValueError):
                out["Min"] = orig_min

        # Use effective Max when it was reduced below what the user typed (ceiling applied).
        # Skip when floor was applied — we're already hiding all bounds in that case.
        if not min_was_floored and orig_max is not None:
            try:
                out["Max"] = eff_max if (eff_max is not None and float(eff_max) < float(orig_max)) else orig_max
            except (TypeError, ValueError):
                out["Max"] = orig_max

        if min_was_floored:
            # Intentionally empty — opener shows only the nutrient name, no numbers.
            return {}
        out = {k: v for k, v in out.items() if v is not None}
        if out:
            return out
    return rng


def _heybo_nutrient_opener_name_and_unit(nutrient_key: str) -> tuple:
    """User-facing nutrient label and unit suffix (e.g. protein_g → protein, g)."""
    key = (nutrient_key or "").strip()
    low = key.lower()
    if low.endswith("_g"):
        return low.replace("_g", "").replace("_", " "), "g"
    if low.endswith("_mg"):
        return low.replace("_mg", "").replace("_", " "), "mg"
    if low.endswith("_mcg"):
        return low.replace("_mcg", "").replace("_", " "), "mcg"
    if low.endswith("_kcal"):
        return low.replace("_kcal", "").replace("_", " "), "kcal"
    return key.replace("_", " "), ""


def _heybo_format_nutrient_filter_for_opener(nf: dict) -> str:
    nm = str((nf or {}).get("Nutrient") or "").strip()
    if not nm:
        return ""
    name, unit = _heybo_nutrient_opener_name_and_unit(nm)
    rng = _heybo_nutrient_filter_user_display_range(nf)
    mn, mx = rng.get("Min"), rng.get("Max")

    def _bound_clause(kind: str, val) -> str:
        b = _heybo_format_nutrient_bound(val)
        if unit == "kcal":
            return f"{kind} {b} kcal"
        if unit:
            return f"{kind} {b}{unit} {name}"
        return f"{kind} {b} {name}"

    if mn is not None and mx is not None:
        # Equal Min/Max (or opened point target) → user intent is "at least Min".
        try:
            equal_bounds = abs(float(mn) - float(mx)) <= 1e-6
        except (TypeError, ValueError):
            equal_bounds = False
        if equal_bounds or nf.get("_equal_point_target_opened"):
            # Prefer original Min when we opened Max for a point target.
            show_min = mn
            orig = nf.get("_original_range") if isinstance(nf.get("_original_range"), dict) else None
            if orig and orig.get("Min") is not None:
                show_min = orig.get("Min")
            return _bound_clause("minimum", show_min)
        b_min = _heybo_format_nutrient_bound(mn)
        b_max = _heybo_format_nutrient_bound(mx)
        if unit == "kcal":
            return f"minimum {b_min} maximum {b_max} kcal"
        if unit:
            return f"minimum {b_min}{unit} maximum {b_max}{unit} {name}"
        return f"minimum {b_min} maximum {b_max} {name}"
    if mn is not None:
        return _bound_clause("minimum", mn)
    if mx is not None:
        return _bound_clause("maximum", mx)
    return name


def _heybo_nutrient_filter_range_was_adjusted(nf: dict) -> bool:
    """
    True when DB guideline alignment changed targets (for user-facing notes).

    Pure diet→nutrient conversion (e.g. High Protein → protein_g 45–75) has no
    _original_range snapshot; comparing {} to Range must not count as drift.

    Auto-added Max for Min-only requests (customization ceiling) is not treated
    as adjustment — notes are for relaxation / changed user bounds only.
    """
    if not isinstance(nf, dict):
        return False
    # Point-target soft Max (Min==Max → at least Min) is intentional preprocessing, not drift.
    if nf.get("_equal_point_target_opened"):
        return False
    orig = nf.get("_original_range")
    applied = nf.get("Range") or {}
    if isinstance(orig, dict) and _heybo_has_original_range_snapshot(orig):
        if _heybo_is_internal_range_bound_fill_in(orig, applied):
            return False
        if _heybo_numeric_range_drift(orig, applied):
            return True
        return False
    if nf.get("_guidelines_adjusted"):
        return True
    return False


def _scrub_nutrient_filter_for_messaging(nf):
    out = dict(nf) if isinstance(nf, dict) else nf
    if not isinstance(out, dict):
        return out
    out.pop("_original_range", None)
    out["_guidelines_adjusted"] = False
    return out


def _scrub_user_input_if_guideline_note_redundant(ui, bowls, nutrient_relaxation_level):
    """
    If every bowl already meets applied NutrientFilters at relaxation 0, optionally clear
    redundant guideline messaging — but never when DB alignment changed the user's targets
    (Salad keeps ``message_to_user`` Note + avoids strict “match your …” in that case).
    """
    if nutrient_relaxation_level != 0:
        return
    nfs = ui.get("NutrientFilters") or []
    if not bowls or not nfs or not _all_bowls_meet_applied_nutrient_filters(bowls, nfs):
        return
    gv = ui.get("global_validations") or {}
    for line in gv.get("guideline_adjustments") or []:
        if not isinstance(line, str):
            continue
        low = line.lower()
        if "minimum " in low or "maximum " in low or "adjusted from" in low:
            return
    for nf in nfs:
        if not isinstance(nf, dict):
            continue
        if _heybo_nutrient_filter_range_was_adjusted(nf):
            return
    ui["global_validations"] = {**gv, "guideline_adjustments": []}
    ui["NutrientFilters"] = [_scrub_nutrient_filter_for_messaging(nf) for nf in nfs]


def _heybo_any_nutrient_relaxation(ui: dict) -> bool:
    per = ui.get("nutrient_relaxation_levels") or {}
    if isinstance(per, dict) and any(int(v or 0) > 0 for v in per.values()):
        return True
    return int(ui.get("nutrient_relaxation_level") or 0) > 0


def _heybo_has_relaxation_or_adjustment_like_salad(ui: dict) -> bool:
    """Mirror ``salad._has_any_relaxation_or_adjustment`` for Heybo opener phrasing."""
    gv = ui.get("global_validations") or {}
    if gv.get("price_minimum_violation"):
        return True
    if int(ui.get("balanced_relaxation_level") or 0) > 0:
        return True
    if int(ui.get("nutrient_relaxation_level") or 0) > 0:
        return True
    if _heybo_any_nutrient_relaxation(ui):
        return True
    if int(ui.get("nutrient_relaxation_percentage") or 0) > 0:
        return True
    if ui.get("cuisine_relaxation_enabled") or ui.get("prep_relaxation_enabled") or ui.get(
        "flavor_relaxation_enabled"
    ):
        return True
    if int(ui.get("light_hearty_relaxation_level") or 0) > 0:
        return True
    if int(ui.get("co2_relaxation_level") or 0) > 0:
        return True
    if ui.get("relaxed_filters"):
        return True
    if gv.get("guideline_adjustments"):
        return True
    if ui.get("_costlier_bowls_capped"):
        return True
    if ui.get("_most_bowls_outside_price_range"):
        return True
    if _heybo_filter_summary_mentions_price_relaxation(ui):
        return True
    if int(ui.get("price_relaxation_level") or 0) > 0:
        return True
    return False


def _heybo_will_append_relaxation_note(ui: dict, bowls: List[dict]) -> bool:
    """
    True when the Note section will include a filter-relaxation line.
    Opener must not repeat strict limits alongside that note (LLM / user clarity).
    """
    if int(ui.get("nutrient_relaxation_level") or 0) > 0:
        return True
    if _heybo_any_nutrient_relaxation(ui):
        return True
    if int(ui.get("balanced_relaxation_level") or 0) > 0:
        return True
    lh_level = int(ui.get("light_hearty_relaxation_level") or 0)
    if lh_level > 0 and (ui.get("Light") is True or ui.get("Hearty") is True):
        return True
    if _heybo_price_relaxation_level(ui) > 0:
        return True
    if int(ui.get("co2_relaxation_level") or 0) > 0:
        return True
    return False


def _heybo_relaxation_active(ui: dict) -> bool:
    """True when any user filter was relaxed or adjusted (opener vs Note should not repeat)."""
    if int(ui.get("co2_relaxation_level") or 0) > 0:
        return True
    if _heybo_has_relaxation_or_adjustment_like_salad(ui):
        return True
    if _heybo_guideline_alignment_friendly_notes(ui):
        return True
    gv = ui.get("global_validations") or {}
    if _heybo_flavor_adjustment_user_notes(ui, gv):
        return True
    return False


def _heybo_relaxation_omit_from_opener(ui: dict) -> Set[str]:
    """
    Filter categories explained in Note lines — omit from opener ``match_desc`` and trailing clauses.
    Keys: nutrients, flavor, balanced, light_hearty, sustainable, price.
    """
    omit: Set[str] = set()
    if (
        _heybo_any_nutrient_relaxation(ui)
        or _heybo_diet_fallbacks_mention_nutrient_relaxation(ui)
    ):
        omit.add("nutrients")
    elif _heybo_guideline_alignment_friendly_notes(ui):
        omit.add("nutrients")
    if int(ui.get("balanced_relaxation_level") or 0) > 0:
        omit.add("balanced")
    if ui.get("_balanced_suppressed_by_nutrients"):
        omit.add("balanced")
    if int(ui.get("light_hearty_relaxation_level") or 0) > 0:
        omit.add("light_hearty")
    if ui.get("_light_hearty_suppressed_by_nutrients"):
        omit.add("light_hearty")
    if int(ui.get("co2_relaxation_level") or 0) > 0:
        omit.add("sustainable")
    if ui.get("flavor_relaxation_enabled"):
        omit.add("flavor")
    if _heybo_price_relaxation_level(ui) > 0 or _heybo_filter_summary_mentions_price_relaxation(ui):
        omit.add("price")
    if ui.get("_costlier_bowls_capped") or ui.get("_most_bowls_outside_price_range"):
        omit.add("price")
    return omit


def _heybo_flavor_note_redundant_with_opener(note: str, ui: dict) -> bool:
    """Skip band-fallback flavor notes when the same flavor keys are already in the opener."""
    fp = ui.get("FlavorPreferences") or {}
    if not isinstance(fp, dict):
        return False
    low = note.lower()
    for key, val in fp.items():
        if val and str(val).strip() and str(key).lower() in low:
            return True
    return False


def _clamp_relaxation_level(level):
    if not isinstance(level, int):
        return 0
    if level < 0:
        return 0
    max_i = len(RELAXATION_PERCENTAGES) - 1
    return level if level <= max_i else max_i


def _heybo_request_names_present_in_bowls(names, bowls) -> List[str]:
    """Keep Include/Extra names that actually landed in at least one bowl (order preserved)."""
    present = {n.lower() for n in _heybo_all_bowl_ingredient_names(bowls)}
    out: List[str] = []
    seen = set()
    for x in names or []:
        s = str(x).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        if key not in present:
            continue
        seen.add(key)
        out.append(s)
    return out


def _heybo_requested_names_for_opener(user_input, bowls):
    """Include names for the opener — only those present in generated bowls.

    Names that never landed stay in the Note (``Requested ingredients not available…``),
    so the LLM is not told they were fulfilled in the same sentence.
    """
    inc = ((user_input or {}).get("Ingredients") or {}).get("Include") or []
    if not inc:
        return None
    landed = _heybo_request_names_present_in_bowls(inc, bowls)
    return ", ".join(landed) if landed else None


def _normalize_heybo_message_to_user(msg, bowls, user_input):
    """
    SaladStop strings → Heybo product copy, without editing salad.py:
      - '… - Successfully included requested ingredients: X' → '… with requested X'
      - Generic '… with great variety that match your … preferences' when Include is set →
        '… with requested {includes}' (covers the case where Salad drops the include line as redundant).
    """
    if not msg:
        return msg
    marker = " - Successfully included requested ingredients: "
    if marker in msg:
        prefix, _, rest = msg.partition(marker)
        note_sep = " - Note:"
        if note_sep in rest:
            ing, _, tail = rest.partition(note_sep)
            msg = f"{prefix} with requested {ing.strip()}{note_sep}{tail}"
        else:
            msg = f"{prefix} with requested {rest.strip()}"

    return msg


def _heybo_co2_sustainable_max_kg(ui: dict) -> Optional[float]:
    """Sustainable CO2 upper bound from Heybo preference_filters (kg), if configured."""
    try:
        from .config_loader import get_heybo_config

        rid = (ui.get("recommend_page_id") or "").strip()
        sid = (ui.get("session_id") or "").strip()
        cfg = get_heybo_config(rid, sid)
        mx = ((cfg.get("co2_config") or {}).get("Sustainable") or {}).get("max")
        if mx is None or mx == "":
            return None
        return float(mx)
    except Exception:
        return None


def _heybo_diet_filter_names(ui: dict) -> List[str]:
    """Active diet filters only (dict values that are True, or list entries)."""
    raw = ui.get("DietFilters")
    if isinstance(raw, dict):
        return [str(d).strip() for d, active in raw.items() if active is True and str(d).strip()]
    if isinstance(raw, list):
        return [str(d).strip() for d in raw if str(d).strip()]
    return []


def _heybo_excluded_diet_filter_names(ui: dict) -> List[str]:
    """Diets the user marked False (avoid / do not target), for exclusion messaging."""
    raw = ui.get("DietFilters")
    if isinstance(raw, dict):
        return [str(d).strip() for d, active in raw.items() if active is False and str(d).strip()]
    return []


def _heybo_ingredient_based_diet_names(ui: dict) -> List[str]:
    """Diets enforced on ingredients (Vegan, Vegetarian, …), not nutrient-band conversions."""
    return [d for d in _heybo_diet_filter_names(ui) if d not in DIET_TO_NUTRIENT_MAP]


_HEYBO_ALLERGEN_APPLIED_SUFFIX = "Allergen preferences applied: {labels}"


def _heybo_false_dict_labels(raw: Any, *, suffix: str = "") -> List[str]:
    """Labels for dict filters marked False (DietFilters, CuisineFilters, PreparationMethod, …)."""
    if not isinstance(raw, dict):
        return []
    labels: List[str] = []
    for name, active in raw.items():
        if active is not False:
            continue
        s = str(name).strip()
        if not s:
            continue
        labels.append(f"{s} {suffix}".strip() if suffix else s)
    return labels


def _heybo_allergen_exclusion_names(ui: dict) -> List[str]:
    """Allergens the user asked to avoid (list, or dict keys with True)."""
    af = ui.get("AllergenFilters")
    if isinstance(af, dict):
        return [str(k).strip() for k, v in af.items() if v is True and str(k).strip()]
    if isinstance(af, list):
        return [str(a).strip() for a in af if str(a).strip()]
    return []


def _heybo_excluded_ingredient_names(ui: dict) -> List[str]:
    """Ingredients.Exclude only."""
    names: List[str] = []
    ing = ui.get("Ingredients") or {}
    if isinstance(ing, dict):
        for x in ing.get("Exclude") or []:
            s = str(x).strip()
            if s and s not in names:
                names.append(s)
    return names


def _heybo_excluded_filter_labels(ui: dict) -> List[str]:
    """False DietFilters / CuisineFilters / PreparationMethod (not ingredient Exclude)."""
    labels: List[str] = []
    for d in _heybo_excluded_diet_filter_names(ui):
        if d not in labels:
            labels.append(d)
    for label in _heybo_false_dict_labels(ui.get("CuisineFilters"), suffix="cuisine"):
        if label not in labels:
            labels.append(label)
    for label in _heybo_false_dict_labels(ui.get("PreparationMethod"), suffix="preparation"):
        if label not in labels:
            labels.append(label)
    return labels


def _heybo_full_exclude_covered_ingredient_names(ui: dict) -> Set[str]:
    """Ingredient names dropped because an entire category was excluded."""
    gv = ui.get("global_validations") or {}
    names = gv.get("full_exclude_covered_ingredient_names") or []
    return {str(n).strip() for n in names if str(n).strip()}


def _heybo_exclusion_not_present_clause(ui: dict) -> str:
    """
    Exclusion copy for Ingredients.Exclude and/or False dict filters.
    Wording: ingredients only | filters only | ingredients and filters.
    """
    ingredient_names = _heybo_excluded_ingredient_names(ui)
    covered = _heybo_full_exclude_covered_ingredient_names(ui)
    gv = ui.get("global_validations") or {}
    conflict_covered = {
        str(n).strip()
        for n in (gv.get("full_exclude_conflict_ingredient_names") or [])
        if str(n).strip()
    }
    omit_or_conflict = covered | conflict_covered
    if omit_or_conflict:
        ingredient_names = [n for n in ingredient_names if n not in omit_or_conflict]
    filter_labels = _heybo_excluded_filter_labels(ui)
    if not ingredient_names and not filter_labels:
        return ""
    listed = ingredient_names + filter_labels
    if ingredient_names and filter_labels:
        lead = (
            "The following ingredients and filters were EXCLUDED "
            "and are NOT present in any bowl"
        )
    elif ingredient_names:
        lead = (
            "The following ingredients were EXCLUDED "
            "and are NOT present in any bowl"
        )
    else:
        lead = (
            "The following filters were EXCLUDED "
            "and are NOT present in any bowl"
        )
    return f"{lead}: {', '.join(listed)}"


def _heybo_excluded_ingredients_and_filters_descriptor(ui: dict) -> str:
    """Comma list for legacy callers (ingredient excludes + false dict filters)."""
    return ", ".join(_heybo_excluded_ingredient_names(ui) + _heybo_excluded_filter_labels(ui))


def _heybo_allergen_restriction_clause(ui: dict) -> str:
    """Short allergen line — does not claim bowl contents match colloquial names (e.g. “meat”)."""
    names = _heybo_allergen_exclusion_names(ui)
    if not names:
        return ""
    return _HEYBO_ALLERGEN_APPLIED_SUFFIX.format(labels=", ".join(names))


def _heybo_flavor_preferences_clause(ui: dict) -> str:
    """Short flavor line from FlavorPreferences (e.g. Salty: No)."""
    fp = ui.get("FlavorPreferences") or {}
    if not isinstance(fp, dict):
        return ""
    active = [
        f"{str(k).title()} ({str(v).strip().title()})"
        for k, v in fp.items()
        if v and str(v).strip()
    ]
    if not active:
        return ""
    return f"Flavor preferences applied: {', '.join(active)}"


def _heybo_flavor_follow_preference_note(ui: dict) -> str:
    """Short user line: bowls follow requested flavor(s). No bands or categories."""
    fp = ui.get("FlavorPreferences") or {}
    if not isinstance(fp, dict):
        return ""
    names = [str(k).strip().lower() for k, v in fp.items() if v and str(v).strip()]
    if not names:
        return ""
    if len(names) == 1:
        return f"Bowls follow your {names[0]} flavor preference."
    if len(names) == 2:
        return f"Bowls follow your {names[0]} and {names[1]} flavor preferences."
    return (
        "Bowls follow your "
        + ", ".join(names[:-1])
        + f", and {names[-1]} flavor preferences."
    )


def _heybo_flavor_adjustment_user_notes(ui: dict, gv: Optional[dict] = None) -> List[str]:
    """
    User-facing flavor notes from global flavor_adjustments.
    Internal band/category fallback detail stays in Validations; the customer line
    only says bowls follow the requested flavor.
    """
    gv = gv or {}
    has_band_fallback = False
    for line in gv.get("flavor_adjustments") or []:
        if not isinstance(line, str) or not line.strip():
            continue
        low = line.lower()
        if "flavor filter applied for" in low and "kept flavor-matched" in low:
            continue
        if "no matching flavor columns" in low:
            continue
        if "preferred):" in line and "band" in low:
            has_band_fallback = True
            break
    if not has_band_fallback:
        return []
    friendly = _heybo_flavor_follow_preference_note(ui)
    return [friendly] if friendly else []


def _heybo_excluded_filters_descriptor(ui: dict) -> str:
    """Legacy combined string (ingredient/filter excludes only)."""
    return _heybo_excluded_ingredients_and_filters_descriptor(ui)


def _heybo_append_user_filter_clauses(
    base: str, ui: dict, *, omit_categories: Optional[Set[str]] = None
) -> str:
    """Append exclusion + allergen + flavor sentences after the opener (skip categories in Note)."""
    omit = omit_categories or set()
    exclusion = _heybo_exclusion_not_present_clause(ui)
    allergen = _heybo_allergen_restriction_clause(ui)
    flavor = _heybo_flavor_preferences_clause(ui)
    if exclusion and "exclusion" not in omit:
        base = f"{base}. {exclusion}"
    if allergen and "allergens" not in omit:
        low = base.lower()
        names = _heybo_allergen_exclusion_names(ui)
        if not names or not all(n.lower() in low for n in names):
            base = f"{base}. {allergen}"
    if flavor and "flavor" not in omit and "flavor" not in base.lower():
        base = f"{base}. {flavor}"
    return base


def _heybo_match_filters_descriptor(
    ui: dict,
    *,
    omit_categories: Optional[Set[str]] = None,
    include_ingredients: bool = True,
    bowls: Optional[List[dict]] = None,
) -> str:
    """Positive filters only — for “… match your X preferences” (Salad-style, simplified).

    When ``bowls`` is provided, Include/Extra names that never landed are omitted
    (they belong only in the Note).
    """
    omit = omit_categories or set()
    parts: List[str] = []
    diets = _heybo_diets_for_match_descriptor(ui, omit)
    if diets:
        parts.append(", ".join(diets))

    if include_ingredients:
        ing = ui.get("Ingredients") or {}
        if isinstance(ing, dict):
            inc = [str(x).strip() for x in (ing.get("Include") or []) if str(x).strip()]
            if bowls:
                inc = _heybo_request_names_present_in_bowls(inc, bowls)
            if inc:
                parts.append("requested " + ", ".join(inc))
            ext = [str(x).strip() for x in (ing.get("Extra") or []) if str(x).strip()]
            if bowls:
                ext = _heybo_request_names_present_in_bowls(ext, bowls)
            if ext:
                parts.append("extra " + ", ".join(ext))

    cf = ui.get("CuisineFilters") or {}
    if isinstance(cf, dict):
        inc_c = [str(c).strip() for c, v in cf.items() if v is True]
        if inc_c:
            parts.append(", ".join(inc_c) + " cuisine")

    pm = ui.get("PreparationMethod") or {}
    if isinstance(pm, dict):
        inc_m = [str(m).strip() for m, v in pm.items() if v is True and str(m).strip()]
        if inc_m:
            parts.append(", ".join(inc_m) + " preparation")

    fp = ui.get("FlavorPreferences") or {}
    if isinstance(fp, dict) and "flavor" not in omit:
        active = [f"{k} ({str(v).strip()})" for k, v in fp.items() if v and str(v).strip()]
        if active:
            parts.append(", ".join(active) + " flavor")

    nfs = ui.get("NutrientFilters") or []
    if "nutrients" not in omit and isinstance(nfs, list) and nfs:
        bits = []
        for nf in nfs:
            if not isinstance(nf, dict):
                continue
            bit = _heybo_format_nutrient_filter_for_opener(nf)
            if bit:
                bits.append(bit)
        if bits:
            parts.append(", ".join(bits))

    price = ui.get("Price") or {}
    if "price" not in omit and isinstance(price, dict):
        if price.get("Min") is not None:
            parts.append(f"minimum price (${price['Min']})")
        if price.get("Max") is not None:
            parts.append(f"maximum price (${price['Max']})")

    if "light_hearty" not in omit:
        if ui.get("Light") is True and ui.get("Hearty") is not True:
            parts.append("Light")
        elif ui.get("Hearty") is True and ui.get("Light") is not True:
            parts.append("Hearty")
        elif ui.get("Light") is True and ui.get("Hearty") is True:
            parts.append("Light & Hearty")

    if ui.get("Balanced") is True and "balanced" not in omit:
        parts.append("balanced diet")

    af_names = _heybo_allergen_exclusion_names(ui)
    if af_names:
        parts.append("avoiding " + ", ".join(af_names))

    if ui.get("Sustainable") is True and "sustainable" not in omit:
        parts.append("Sustainable (low CO2e)")

    return ", ".join(parts)


def _heybo_active_filters_descriptor(ui: dict) -> str:
    """Legacy combined descriptor; prefer match + excluded helpers for user messages."""
    match = _heybo_match_filters_descriptor(ui)
    excluded = _heybo_excluded_filters_descriptor(ui)
    if match and excluded:
        return match
    if match:
        return match
    return excluded


def _heybo_diet_fallbacks_mention_nutrient_relaxation(ui: dict) -> bool:
    gv = ui.get("global_validations") or {}
    for line in gv.get("diet_fallbacks") or []:
        if not isinstance(line, str):
            continue
        low = line.lower()
        if "nutrient relaxation" in low or "applied nutrient relaxation" in low:
            return True
    return False


def _heybo_filter_summary_mentions_price_relaxation(ui: dict) -> bool:
    gv = ui.get("global_validations") or {}
    for line in gv.get("filter_summary") or []:
        if isinstance(line, str) and "widened price filter band" in line.lower():
            return True
    return False


def _heybo_price_relaxation_level(ui: dict) -> int:
    lvl = ui.get("price_relaxation_level")
    if isinstance(lvl, int) and lvl > 0:
        return lvl
    max_step = 0
    gv = ui.get("global_validations") or {}
    for line in gv.get("filter_summary") or []:
        if not isinstance(line, str):
            continue
        m = re.search(r"step\s+(\d+)", line, re.I)
        if m and "widened price filter band" in line.lower():
            max_step = max(max_step, int(m.group(1)))
    return max_step


def _heybo_any_bowl_price_outside_requested_range(bowls: List[dict]) -> bool:
    for bowl in bowls or []:
        for line in (
            (bowl.get("Validations") or {}).get("bowl_specific", {}).get("price") or []
        ):
            if isinstance(line, str) and "outside the requested range" in line.lower():
                return True
    return False


def _heybo_user_has_price_filter(ui: dict) -> bool:
    price = ui.get("Price") or {}
    if not isinstance(price, dict):
        return False
    return price.get("Min") is not None or price.get("Max") is not None


def _heybo_price_target_descriptor(ui: dict) -> str:
    """User-facing price target (e.g. ``$23`` or ``minimum $20 and maximum $25``)."""
    orig = ui.get("_price_original_before_snap")
    price = orig if isinstance(orig, dict) and (orig.get("Min") is not None or orig.get("Max") is not None) else (ui.get("Price") or {})
    if not isinstance(price, dict):
        return ""
    mn, mx = price.get("Min"), price.get("Max")
    try:
        if mn is not None and mx is not None and abs(float(mn) - float(mx)) < 1e-6:
            return f"${float(mn):g}"
    except (TypeError, ValueError):
        pass
    parts: List[str] = []
    try:
        if mn is not None:
            parts.append(f"minimum ${float(mn):g}")
        if mx is not None:
            parts.append(f"maximum ${float(mx):g}")
    except (TypeError, ValueError):
        pass
    return " and ".join(parts)


def _heybo_bowl_total_costs(bowls: List[dict]) -> List[float]:
    costs: List[float] = []
    for bowl in bowls or []:
        raw = bowl.get("Total Cost")
        if raw is None:
            continue
        try:
            costs.append(float(raw))
        except (TypeError, ValueError):
            continue
    return costs


def _heybo_price_relaxation_notes(ui: dict, bowls: List[dict]) -> List[str]:
    if not _heybo_user_has_price_filter(ui):
        return []
    gv = ui.get("global_validations") or {}
    if gv.get("price_minimum_violation"):
        return []
    if ui.get("_costlier_bowls_capped"):
        return ["Price adjusted for the best available options."]
    if ui.get("_most_bowls_outside_price_range"):
        return ["Price adjusted to fit available options."]
    target = _heybo_price_target_descriptor(ui)
    lvl = _heybo_price_relaxation_level(ui)
    outside = _heybo_any_bowl_price_outside_requested_range(bowls)
    if lvl <= 0 and not outside:
        return []
    if lvl > 0:
        return ["Price adjusted for the best available options."]
    if outside and target:
        return [
            f"Bowls shown may not match your {target} price target exactly — "
            "try adjusting ingredients or price"
        ]
    return []


def _heybo_diets_for_match_descriptor(ui: dict, omit: Set[str]) -> List[str]:
    """
    Diet names for the opener. When nutrients are covered by a Note, drop nutrient-based diets
    (e.g. Low Sodium) so Sodium is not stated twice.
    """
    diets = _heybo_diet_filter_names(ui)
    if "nutrients" not in omit:
        return diets
    converted: Set[str] = set()
    for nf in ui.get("NutrientFilters") or []:
        if not isinstance(nf, dict):
            continue
        od = nf.get("_original_diet")
        if isinstance(od, str) and od.strip():
            converted.add(od.strip())
    return [d for d in diets if d not in DIET_TO_NUTRIENT_MAP and d not in converted]


def _heybo_opener_segment_overlaps_note(segment: str, note_blob: str, ui: dict) -> bool:
    """True when an opener preference clause repeats a nutrient/diet already explained in the Note."""
    seg_low = segment.lower()
    note_low = note_blob.lower()
    if not seg_low or not note_low:
        return False
    if any(
        tok in note_low and tok in seg_low
        for tok in ("sodium", "nutritional requirements", "nutritional requirement")
    ):
        return True
    if "price" in seg_low and ("price" in note_low or "target" in note_low):
        return True
    for diet, nutrient_key in DIET_TO_NUTRIENT_MAP.items():
        stem = nutrient_key.replace("_mg", "").replace("_g", "").replace("_", " ")
        if diet.lower() in seg_low and (diet.lower() in note_low or stem in note_low):
            return True
        if diet.lower() in seg_low and "optimized your" in note_low:
            return True
    for nf in ui.get("NutrientFilters") or []:
        if not isinstance(nf, dict):
            continue
        nk = (nf.get("Nutrient") or "").strip()
        if not nk:
            continue
        stem = nk.replace("_mg", "").replace("_g", "").replace("_", " ")
        friendly = _heybo_nutrient_display_name(nk).lower()
        if stem in note_low and (stem in seg_low or friendly in seg_low):
            return True
    return False


def _heybo_strip_opener_overlap_with_note(
    base: str, note_blob: str, ui: Optional[dict] = None
) -> str:
    """Remove ``for your …`` segments that duplicate nutrient/diet wording already in the Note."""
    if not ui or not note_blob.strip():
        return base
    phrase = " for your "
    idx = base.find(phrase)
    if idx < 0:
        return base
    end_idx = base.find(" preferences", idx)
    if end_idx < 0:
        return base
    inner = base[idx + len(phrase) : end_idx]
    segments = [s.strip() for s in inner.split(",") if s.strip()]
    kept = [
        s for s in segments if not _heybo_opener_segment_overlaps_note(s, note_blob, ui)
    ]
    if kept == segments:
        return base
    tail_start = end_idx + len(" preferences")
    if not kept:
        return (base[:idx].rstrip() + base[tail_start:]).replace("  ", " ").strip()
    return base[:idx] + phrase + ", ".join(kept) + base[end_idx:]


def _heybo_nutrient_display_name(nutrient_key: str) -> str:
    key = (nutrient_key or "").strip()
    if not key:
        return ""
    if key.endswith("_g"):
        return key.replace("_g", "").replace("_", " ").title() + " (g)"
    if key.endswith("_mg"):
        return key.replace("_mg", "").replace("_", " ").title() + " (mg)"
    if key.endswith("_mcg"):
        return key.replace("_mcg", "").replace("_", " ").title() + " (mcg)"
    if key.endswith("_kcal") or key.endswith("_kCal"):
        base = key.replace("_kcal", "").replace("_kCal", "").replace("_", " ").title()
        return f"{base} Kcal"
    return key.replace("_", " ").title()


def _heybo_collect_active_nutrient_based_diets(ui: dict) -> Set[str]:
    """Diets that map to nutrients — from DietFilters and from diet-converted nutrient rows."""
    out: Set[str] = set()
    df = ui.get("DietFilters")
    if isinstance(df, dict):
        for diet, active in df.items():
            if active and diet in DIET_TO_NUTRIENT_MAP:
                out.add(diet)
    elif isinstance(df, list):
        for diet in df:
            if isinstance(diet, str) and diet.strip() in DIET_TO_NUTRIENT_MAP:
                out.add(diet.strip())
    for nf in ui.get("NutrientFilters") or []:
        if not isinstance(nf, dict):
            continue
        od = nf.get("_original_diet")
        if isinstance(od, str) and od.strip() in DIET_TO_NUTRIENT_MAP:
            out.add(od.strip())
    return out


def _heybo_guideline_alignment_friendly_notes(ui: dict) -> List[str]:
    """
    Salad-style short note when DB guideline alignment changed targets (relaxation level 0).
    Must not depend on DietFilters still listing diets that were converted to NutrientFilters.
    """
    nutrient_filters = ui.get("NutrientFilters") or []
    all_nutrient_based_diets = _heybo_collect_active_nutrient_based_diets(ui)

    nutrient_to_diet_map = {
        nutrient: diet for diet, nutrient in DIET_TO_NUTRIENT_MAP.items() if diet in all_nutrient_based_diets
    }
    adjusted_nutrient_based_diets: Set[str] = set()
    direct_adjusted_nutrients: Set[str] = set()
    direct_non_adjusted_nutrients: Set[str] = set()

    for nf in nutrient_filters:
        if not isinstance(nf, dict):
            continue
        nutrient_name = (nf.get("Nutrient") or "").strip()
        if not nutrient_name:
            continue
        was_adjusted = _heybo_nutrient_filter_range_was_adjusted(nf)
        if nf.get("_diet_converted") and nf.get("_original_diet"):
            original_diet = nf.get("_original_diet")
            if isinstance(original_diet, str) and original_diet in DIET_TO_NUTRIENT_MAP:
                if was_adjusted:
                    adjusted_nutrient_based_diets.add(original_diet)
                continue
        if nutrient_name in nutrient_to_diet_map:
            diet_name = nutrient_to_diet_map[nutrient_name]
            if was_adjusted:
                adjusted_nutrient_based_diets.add(diet_name)
            continue

        friendly_name = _heybo_nutrient_display_name(nutrient_name)
        if not friendly_name:
            continue
        if was_adjusted:
            direct_adjusted_nutrients.add(friendly_name)
        else:
            direct_non_adjusted_nutrients.add(friendly_name)

    if all_nutrient_based_diets or direct_adjusted_nutrients or direct_non_adjusted_nutrients:
        non_adjusted_diets = all_nutrient_based_diets - adjusted_nutrient_based_diets
        all_adjusted = sorted(list(adjusted_nutrient_based_diets) + list(direct_adjusted_nutrients))
        all_non_adjusted = sorted(list(non_adjusted_diets) + list(direct_non_adjusted_nutrients))
        note_parts: List[str] = []
        if all_adjusted:
            note_parts.append(
                f"We optimized your {', '.join(all_adjusted)} requirements to give you more great options"
            )
        if all_non_adjusted and all_adjusted:
            note_parts.append(
                f"your {', '.join(all_non_adjusted)} preferences were met exactly as you wanted"
            )
        if len(note_parts) == 2:
            return [f"{note_parts[0]}, and {note_parts[1]}"]
        if note_parts:
            return [". ".join(note_parts)]

    adjusted_friendly = sorted(
        {
            _heybo_nutrient_display_name((nf.get("Nutrient") or "").strip())
            for nf in nutrient_filters
            if isinstance(nf, dict) and _heybo_nutrient_filter_range_was_adjusted(nf)
        }
    )
    adjusted_friendly = [x for x in adjusted_friendly if x]
    if adjusted_friendly:
        return [f"Your {', '.join(adjusted_friendly)} ranges were adjusted for the best fit"]
    return []


def _heybo_nutrient_message_name(nutrient_key: str) -> str:
    """Short user-facing nutrient label for relaxation notes (e.g. protein_g → Protein)."""
    name, unit = _heybo_nutrient_opener_name_and_unit(nutrient_key)
    if unit == "kcal":
        return "Calories"
    if name:
        return name.title()
    return (nutrient_key or "").replace("_", " ").title()


def _heybo_nutrient_relaxation_tail(relaxation_pct: int) -> str:
    if relaxation_pct == 15:
        return "relaxed to ensure variety"
    if relaxation_pct == 30:
        return "relaxed to provide more options"
    if relaxation_pct == 50:
        return "adjusted to ensure great variety"
    if relaxation_pct == 70:
        return "relaxed to provide the best options"
    if relaxation_pct == 85:
        return "relaxed to ensure bowl generation"
    return "adjusted to provide the best options"


def _heybo_nutrient_relaxation_phrase(relaxation_pct: int) -> str:
    return f"were {_heybo_nutrient_relaxation_tail(relaxation_pct)}"


def _heybo_user_requested_nutrient_keys(ui: dict) -> List[str]:
    """All nutrient constraints the user asked for (filters + nutrient-style diets)."""
    gv = ui.get("global_validations") or {}
    stored = gv.get("nutrient_relaxation_active_keys") or ui.get("nutrient_relaxation_active_keys") or []
    keys: List[str] = []
    seen: Set[str] = set()
    for key in list(stored) + heybo_active_nutrient_filter_keys(ui.get("NutrientFilters") or []):
        k = str(key or "").strip()
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
    for diet in _heybo_collect_active_nutrient_based_diets(ui):
        mapped = DIET_TO_NUTRIENT_MAP.get(diet)
        if mapped and mapped not in seen:
            seen.add(mapped)
            keys.append(mapped)
    return keys


def _heybo_format_nutrient_list_for_message(names: List[str]) -> str:
    names = [n for n in names if n]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _heybo_per_nutrient_relaxation_notes(ui: dict) -> List[str]:
    """User note when only some nutrient filters were relaxed (priority order is internal)."""
    per = ui.get("nutrient_relaxation_levels") or {}
    if not isinstance(per, dict) or not any(int(v or 0) > 0 for v in per.values()):
        return []

    nutrient_filters = ui.get("NutrientFilters") or []
    active_keys = _heybo_user_requested_nutrient_keys(ui)
    if not active_keys:
        active_keys = heybo_active_nutrient_filter_keys(nutrient_filters)
    relaxed_keys = [k for k in active_keys if int(per.get(k, 0) or 0) > 0]
    if not relaxed_keys:
        return []

    strict_keys = [k for k in active_keys if int(per.get(k, 0) or 0) == 0]
    relaxed_names = [_heybo_nutrient_message_name(k) for k in relaxed_keys]
    strict_names = [_heybo_nutrient_message_name(k) for k in strict_keys]
    max_level = max(int(per.get(k, 0) or 0) for k in relaxed_keys)
    max_pct = RELAXATION_PERCENTAGES[min(max_level, len(RELAXATION_PERCENTAGES) - 1)]
    tail = _heybo_nutrient_relaxation_tail(max_pct)
    relaxed_blob = _heybo_format_nutrient_list_for_message(relaxed_names)
    relaxed_verb = "range was" if len(relaxed_names) == 1 else "ranges were"

    if strict_names:
        strict_blob = _heybo_format_nutrient_list_for_message(strict_names)
        req_word = "requirement was" if len(strict_names) == 1 else "requirements were"
        return [
            f"Your {strict_blob} {req_word} met; your {relaxed_blob} {relaxed_verb} {tail}"
        ]
    return [f"Your {relaxed_blob} {relaxed_verb} {tail}"]


def _heybo_nutrient_relaxation_notes(ui: dict) -> List[str]:
    """
    When any nutrient relaxation level > 0: friendly messaging for relaxed filters.
    When level is 0: Salad-style guideline alignment friendly note (separate from raw gv lines).
    """
    per_notes = _heybo_per_nutrient_relaxation_notes(ui)
    if per_notes:
        return per_notes

    lvl = int(ui.get("nutrient_relaxation_level") or 0)
    if lvl <= 0:
        return _heybo_guideline_alignment_friendly_notes(ui)

    relaxation_pct = ui.get("nutrient_relaxation_percentage", 0)

    nutrient_filters = ui.get("NutrientFilters") or []
    if not nutrient_filters:
        if relaxation_pct == 15:
            return ["Your nutritional requirements were relaxed to ensure variety"]
        if relaxation_pct == 30:
            return ["Your nutritional requirements were relaxed to provide more options"]
        if relaxation_pct == 50:
            return ["Your nutritional requirements were adjusted to ensure great variety"]
        if relaxation_pct == 70:
            return ["Your nutritional requirements were relaxed to provide the best options"]
        if relaxation_pct == 85:
            return ["Your nutritional requirements were relaxed to ensure bowl generation"]
        return ["Your nutritional requirements were adjusted to provide the best options"]

    relaxed_nutrients: List[str] = []
    for nf in nutrient_filters:
        if not isinstance(nf, dict):
            continue
        nutrient_name = (nf.get("Nutrient") or "").strip()
        if not nutrient_name:
            continue
        friendly_name = _heybo_nutrient_message_name(nutrient_name)
        if friendly_name and friendly_name.lower() not in {x.lower() for x in relaxed_nutrients}:
            relaxed_nutrients.append(friendly_name)

    if relaxed_nutrients:
        nutrient_list = ", ".join(sorted(relaxed_nutrients))
        tail = _heybo_nutrient_relaxation_tail(int(relaxation_pct or 0))
        if len(relaxed_nutrients) == 1:
            return [f"Your {nutrient_list} range was {tail}"]
        return [f"Your {nutrient_list} ranges were {tail}"]

    return _heybo_guideline_alignment_friendly_notes(ui)


def _heybo_balanced_light_hearty_relaxation_notes(ui: dict) -> List[str]:
    notes: List[str] = []
    balanced_level = int(ui.get("balanced_relaxation_level") or 0)
    if balanced_level > 0:
        pct = [0, 15, 30, 50, 70, 85][balanced_level]
        if pct == 15:
            notes.append("Your balanced diet criteria were relaxed to ensure variety")
        elif pct == 30:
            notes.append("Your balanced diet criteria were relaxed to provide more options")
        elif pct == 50:
            notes.append("Your balanced diet criteria were adjusted to ensure great variety")
        elif pct == 70:
            notes.append("Your balanced diet criteria were relaxed to provide the best options")
        elif pct == 85:
            notes.append("Your balanced diet criteria were relaxed to ensure bowl generation")

    lh_level = int(ui.get("light_hearty_relaxation_level") or 0)
    if lh_level > 0 and (ui.get("Light") is True or ui.get("Hearty") is True):
        pct = [0, 15, 30, 50, 70, 85][lh_level]
        lh_type = "Light" if ui.get("Light") is True and ui.get("Hearty") is not True else "Hearty"
        if pct == 15:
            notes.append(f"Your {lh_type} bowl criteria were relaxed to ensure variety")
        elif pct == 30:
            notes.append(f"Your {lh_type} bowl criteria were relaxed to provide more options")
        elif pct == 50:
            notes.append(f"Your {lh_type} bowl criteria were adjusted to ensure great variety")
        elif pct == 70:
            notes.append(f"Your {lh_type} bowl criteria were relaxed to provide the best options")
        elif pct == 85:
            notes.append(f"Your {lh_type} bowl criteria were relaxed to ensure bowl generation")
    return notes


def _heybo_any_bowl_nutrient_line_says_exceeds(bowls: List[dict]) -> bool:
    """True if any bowl validation still reports exceeding a nutrient target (e.g. sugar max)."""
    for bowl in bowls or []:
        lines = (
            (bowl.get("Validations") or {})
            .get("bowl_specific", {})
            .get("nutritional_targets")
            or []
        )
        for line in lines:
            if isinstance(line, str) and "exceeds" in line.lower():
                return True
    return False


def _heybo_full_page_should_avoid_strict_match_phrase(ui: dict, bowls: List[dict]) -> bool:
    """
    When filters were relaxed or bowls still violate a stated target, do not claim
    bowls 'match' strict filter wording (Salad parity).
    """
    if int(ui.get("nutrient_relaxation_level") or 0) > 0:
        return True
    if _heybo_any_nutrient_relaxation(ui):
        return True
    if _heybo_diet_fallbacks_mention_nutrient_relaxation(ui):
        return True
    if _heybo_any_bowl_nutrient_line_says_exceeds(bowls):
        return True
    if _heybo_price_relaxation_level(ui) > 0:
        return True
    if _heybo_any_bowl_price_outside_requested_range(bowls):
        return True
    if _heybo_has_relaxation_or_adjustment_like_salad(ui):
        return True
    return False


_HEYBO_MISSING_INCLUDE_PREFIX = "Requested ingredients not available or not included: "


def _heybo_all_bowl_ingredient_names(bowls: List[dict]) -> Set[str]:
    """Union of ingredient names across every bowl (main + Extra * slots)."""
    names: Set[str] = set()
    for bowl in bowls or []:
        for key in HEYBO_BOWL_COMPONENT_KEYS:
            items = bowl.get(key) or []
            if isinstance(items, str):
                names.add(items.strip())
            else:
                for item in items:
                    if item:
                        names.add(str(item).strip())
    return names


def _heybo_parse_missing_include_line(line: str) -> List[str]:
    if not line.startswith(_HEYBO_MISSING_INCLUDE_PREFIX):
        return []
    rest = line[len(_HEYBO_MISSING_INCLUDE_PREFIX) :].strip()
    return [p.strip() for p in rest.split(",") if p.strip()]


def _heybo_is_only_mode_ui(ui: Optional[dict]) -> bool:
    """True when Only BYB generation / variety page is active."""
    if not isinstance(ui, dict):
        return False
    return (
        bool(ui.get("_heybo_only_mode_generation"))
        or bool(ui.get("_heybo_variety_bowls_active"))
        or _is_truthy_only_flag(ui)
    )


def _heybo_collect_bowl_inline_messages(
    bowls: List[dict], ui: Optional[dict] = None
) -> List[str]:
    """Per-bowl missing-Include lines are merged: only list items absent from every bowl.

    Only mode: skip these entirely for ``message_to_user``. Bowl 1 is exact Include+Extra;
    bowls 2–5 are intentional variety, so missing-include / Extra-count warnings must not
    surface as Notes.
    """
    if _heybo_is_only_mode_ui(ui):
        return []

    out: List[str] = []
    missing_cited: Set[str] = set()
    present = _heybo_all_bowl_ingredient_names(bowls)
    gv = (ui or {}).get("global_validations") or {}
    suppress_exclude_found = bool(gv.get("full_exclude_conflict_categories"))

    for bowl in bowls or []:
        for m in bowl.get("message") or []:
            if not isinstance(m, str) or not m.strip():
                continue
            ms = m.strip()
            if ms.startswith(_HEYBO_MISSING_INCLUDE_PREFIX):
                missing_cited.update(_heybo_parse_missing_include_line(ms))
                continue
            if suppress_exclude_found and (
                ms.startswith("Excluded ingredients found in bowl")
                or ms.startswith("Included despite exclude")
            ):
                continue
            if ms not in out:
                out.append(ms)

    truly_missing = sorted(missing_cited - present)
    if truly_missing:
        out.append(f"{_HEYBO_MISSING_INCLUDE_PREFIX}{', '.join(truly_missing)}")
    return out


def _heybo_strip_match_preferences_clause(base: str) -> str:
    """When appending a Note, drop ``that match your … preferences`` (``salad.py`` ~11103)."""
    out = base
    for phrase in (" that match your ", " that matches your "):
        idx = out.find(phrase)
        if idx != -1:
            end_idx = out.find(" preferences", idx)
            if end_idx != -1:
                end_idx += len(" preferences")
                out = (out[:idx] + out[end_idx:]).replace("  ", " ").strip()
                break
    return out


def _heybo_strip_for_your_preferences_when_note_clarifies(
    base: str, note_blob: str, ui: Optional[dict] = None
) -> str:
    """Keep the full ``for your … preferences`` opener when a Note is appended (Heybo lists all filters)."""
    _ = (note_blob, ui)
    return base


def _heybo_merge_single_note(base: str, note_parts: List[str], ui: Optional[dict] = None) -> str:
    """Append one ` - Note: …` section (Salad-style join)."""
    parts = [p.strip() for p in note_parts if isinstance(p, str) and p.strip()]
    if not parts:
        return base
    dedup: List[str] = []
    for p in parts:
        key = " ".join(p.split()).lower()
        if any(key == " ".join(x.split()).lower() for x in dedup):
            continue
        dedup.append(p)
    if len(dedup) == 1:
        blob = dedup[0]
    elif len(dedup) == 2:
        blob = f"{dedup[0]}, and {dedup[1]}"
    else:
        blob = f"{', '.join(dedup[:-1])}, and {dedup[-1]}"
    head = base.rstrip()
    if " for your " not in head:
        head = _heybo_strip_match_preferences_clause(head)
    head = _heybo_strip_for_your_preferences_when_note_clarifies(head, blob, ui)
    return f"{head} - Note: {blob}"


def _heybo_collect_byb_note_parts(
    ui: dict, bowls: List[dict], *, omit_from_opener: Optional[Set[str]] = None
) -> List[str]:
    """Collect Note parts for Heybo BYB message composition."""
    parts: List[str] = []
    gv = ui.get("global_validations") or {}
    omit = omit_from_opener
    if omit is None and _heybo_relaxation_active(ui):
        omit = _heybo_relaxation_omit_from_opener(ui)
    omit = omit or set()

    for price_note in _heybo_price_relaxation_notes(ui, bowls):
        if price_note and price_note not in parts:
            parts.append(price_note)
    for nutrient_note in _heybo_nutrient_relaxation_notes(ui):
        if nutrient_note and nutrient_note not in parts:
            parts.append(nutrient_note)
    for relax_note in _heybo_balanced_light_hearty_relaxation_notes(ui):
        if relax_note and relax_note not in parts:
            parts.append(relax_note)

    for flavor_note in _heybo_flavor_adjustment_user_notes(ui, gv):
        if not flavor_note or flavor_note in parts:
            continue
        if "flavor" not in omit and _heybo_flavor_note_redundant_with_opener(flavor_note, ui):
            continue
        parts.append(flavor_note)

    # guideline_adjustments (raw DB alignment strings) stay on Validations.global for debugging;
    # user-facing copy is the friendly line from _heybo_nutrient_relaxation_notes (Salad parity).

    for nw in gv.get("nutrient_warnings") or []:
        if "nutrients" in omit:
            continue
        if isinstance(nw, str) and nw.strip() and nw not in parts:
            parts.append(nw.strip())

    for w in gv.get("weight_violations") or []:
        if isinstance(w, str) and w.strip() and w not in parts:
            parts.append(w.strip())

    for pv in gv.get("price_minimum_violation") or []:
        if isinstance(pv, str) and pv.strip():
            c = pv.replace("Note: ", "").strip()
            if c and c not in parts:
                parts.append(c)

    for cm in gv.get("costlier_bowls_message") or []:
        if isinstance(cm, str) and cm.strip() and cm not in parts:
            parts.append(cm.strip())

    if ui.get("Sustainable") is True:
        max_kg = _heybo_co2_sustainable_max_kg(ui)
        if max_kg is not None:
            level = int(ui.get("co2_relaxation_level") or 0)
            if level <= 0:
                parts.append(
                    f"All bowls meet the low CO2e requirement (less than or equal to {max_kg}kg)"
                )
            elif level == 1:
                parts.append(
                    "All bowls meet the relaxed low CO2e requirement "
                    f"(0.80 - 1.80 kg, originally less than or equal to {max_kg}kg)"
                )
            else:
                parts.append(
                    "All bowls generated with relaxed CO2e requirements "
                    f"(originally less than or equal to {max_kg}kg, relaxed for better variety)"
                )

    for bowl in bowls or []:
        expl = (
            (bowl.get("Validations") or {})
            .get("bowl_specific", {})
            .get("ingredient_explanations")
            or []
        )
        for exp in expl:
            if (
                isinstance(exp, str)
                and "excluded due to allergen restrictions" in exp
                and exp.strip() not in parts
            ):
                parts.append(exp.strip())
                break

    parts.extend(_heybo_collect_bowl_inline_messages(bowls, ui))
    return parts


def _heybo_append_byb_note_parts(
    base: str,
    ui: dict,
    bowls: List[dict],
    *,
    omit_from_opener: Optional[Set[str]] = None,
) -> str:
    """Sustainable + global validation notes + per-bowl messages (subset of Salad `_add_additional_filter_messages`)."""
    parts: List[str] = []
    head = base.rstrip()
    if " - Note:" in head:
        head, _, tail = head.partition(" - Note:")
        if tail.strip():
            parts.append(tail.strip())
    parts.extend(_heybo_collect_byb_note_parts(ui, bowls, omit_from_opener=omit_from_opener))
    return _heybo_merge_single_note(head.rstrip(), parts, ui)


def _heybo_compose_byb_message(n: int, ui: dict, bowls: List[dict]) -> str:
    """
    Single BYB opener for any bowl count: list active input filters in the opener,
    except categories explained in a relaxation Note (e.g. omit price from the main
    line when Note says price was adjusted; keep Include / unmet nutrients).
    """
    relaxation_active = _heybo_relaxation_active(ui) or _heybo_any_bowl_price_outside_requested_range(
        bowls
    )
    note_omit = _heybo_relaxation_omit_from_opener(ui) if relaxation_active else set()
    if _heybo_any_bowl_price_outside_requested_range(bowls) and _heybo_user_has_price_filter(ui):
        note_omit.add("price")
    # Keep request prefs in the opener, but drop anything covered by the Note.
    match_desc = _heybo_match_filters_descriptor(
        ui, omit_categories=note_omit, bowls=bowls if n > 0 else None
    )
    requested_frag = _heybo_requested_names_for_opener(ui, bowls) if n > 0 else None
    has_notes = (
        bool(_heybo_collect_byb_note_parts(ui, bowls, omit_from_opener=note_omit))
        if n > 0
        else False
    )
    avoid_strict = (
        _heybo_full_page_should_avoid_strict_match_phrase(ui, bowls)
        if n > 0
        else bool(match_desc)
    )
    nutrient_relaxed = _heybo_any_nutrient_relaxation(ui)
    use_for_your = avoid_strict or has_notes or nutrient_relaxed

    if n <= 0:
        gv = ui.get("global_validations") or {}
        conflict_cats = gv.get("full_exclude_conflict_categories") or []
        limit_notices = [
            n.strip()
            for n in (gv.get("category_limit_notices") or [])
            if isinstance(n, str) and n.strip()
        ]
        if conflict_cats and limit_notices:
            base = f"No bowls generated — {limit_notices[0]}"
        elif conflict_cats:
            if len(conflict_cats) == 1:
                cat = str(conflict_cats[0]).lower()
                base = (
                    f"No bowls generated — each bowl needs at least one {cat} ingredient "
                    f"(you excluded all available {cat} options; bowls include one when possible)"
                )
            else:
                cats = ", ".join(str(c).lower() for c in conflict_cats)
                base = (
                    f"No bowls generated — each bowl needs ingredients from "
                    f"{cats} (you excluded all available options in those categories)"
                )
        elif limit_notices:
            base = f"No bowls generated — {limit_notices[0]}"
        elif match_desc:
            base = (
                f"No bowls generated matching your {match_desc} preferences — "
                "try adjusting your filters"
            )
        else:
            base = (
                "No bowls generated — insufficient ingredient combinations available — "
                "try different preferences"
            )
        omit: Set[str] = set()
        if conflict_cats:
            omit.add("exclusion")
        return _heybo_append_user_filter_clauses(base, ui, omit_categories=omit)

    bowl_word = "bowl" if n == 1 else "bowls"
    if match_desc:
        if use_for_your:
            base = f"Generated {n} {bowl_word} for your {match_desc} preferences"
        else:
            base = (
                f"Generated {n} {bowl_word} with great variety that match your "
                f"{match_desc} preferences"
            )
    elif requested_frag:
        base = f"Generated {n} {bowl_word} with requested {requested_frag}"
    elif n >= _HEYBO_BYB_FULL_PAGE:
        base = f"Generated {n} {bowl_word} with great variety"
    else:
        base = f"Generated {n} {bowl_word}"

    base = _heybo_append_user_filter_clauses(base, ui, omit_categories=note_omit)
    return _heybo_append_byb_note_parts(base, ui, bowls, omit_from_opener=note_omit)


def _heybo_build_full_page_byb_message(ui: dict, bowls: List[dict]) -> str:
    return _heybo_compose_byb_message(len(bowls), ui, bowls)


def build_heybo_message_to_user(
    bowls,
    global_validations=None,
    *,
    user_input=None,
    fallback_categories=None,
    nutrient_relaxation_level=0,
    nutrient_relaxation_levels=None,
    failed_nutrient_attempts=0,
    price_relaxation_level=0,
    error=None,
):
    """
    Build the client-facing string for Heybo BYB (standalone; no salad.py dependency).
    """
    ui = dict(user_input) if isinstance(user_input, dict) else {}
    if ui.get("Sustainable") is True and ui.get("_heybo_sustainable_no_data") is True:
        return "No data found"
    msg_diet = ui.pop("_heybo_message_diet_filters_dict", None)
    if global_validations is not None:
        ui["global_validations"] = global_validations
    else:
        ui.setdefault("global_validations", {})
    if msg_diet is not None:
        ui["DietFilters"] = msg_diet

    lvl = _clamp_relaxation_level(
        nutrient_relaxation_level if isinstance(nutrient_relaxation_level, int) else 0
    )
    ui["nutrient_relaxation_level"] = lvl
    ui["nutrient_relaxation_percentage"] = RELAXATION_PERCENTAGES[lvl]
    if nutrient_relaxation_levels is not None:
        ui["nutrient_relaxation_levels"] = dict(nutrient_relaxation_levels)
    elif ui.get("nutrient_relaxation_levels") is None and global_validations:
        gv_levels = (global_validations or {}).get("nutrient_relaxation_levels")
        if isinstance(gv_levels, dict):
            ui["nutrient_relaxation_levels"] = dict(gv_levels)
    if isinstance(price_relaxation_level, int):
        ui["price_relaxation_level"] = max(0, price_relaxation_level)
    elif _heybo_filter_summary_mentions_price_relaxation(ui):
        ui["price_relaxation_level"] = _heybo_price_relaxation_level(ui)

    if error:
        return f"Unable to generate bowls: {error}"

    b = list(bowls or [])
    _scrub_user_input_if_guideline_note_redundant(ui, b, lvl)

    # Use flag set in generation (same Only detection as bowl build), not Only is True alone.
    only_mode = _heybo_is_only_mode_ui(ui)
    if only_mode and b:
        # Salad parity: opener [Note: limits / min-fill] [variety suffix]. No "Your filters:".
        msg = _heybo_compose_only_mode_message(ui, b)
    else:
        msg = _heybo_compose_byb_message(len(b), ui, b)

    if msg_diet is not None:
        msg = _dedupe_salad_your_preferences_clause(msg)
    msg = _normalize_heybo_message_to_user(msg, b, ui)
    if only_mode and b:
        return msg
    return _append_category_limit_notices(msg, global_validations)


def _requested_bowl_names_display(ui: dict) -> List[str]:
    """Non-empty BowlName strings from the payload (for user-facing messages)."""
    bn = ui.get("BowlName")
    if bn is None:
        return []
    if isinstance(bn, list):
        return [str(x).strip() for x in bn if x is not None and str(x).strip()]
    if isinstance(bn, str) and bn.strip():
        return [bn.strip()]
    return []


def _bowl_type_display_signature(ui: dict) -> str:
    return (
        (ui.get("BowlType") or ui.get("Bowl Type") or "bowl")
        .lower()
        .replace("_", " ")
        .title()
    )


def _heybo_signature_payload_has_filters(ui: dict) -> bool:
    """True when the request carries ingredient/allergen/price/etc. filters beyond BowlType + name."""
    ing = ui.get("Ingredients") or {}
    if isinstance(ing, dict):
        for key in ("Include", "Exclude", "Extra"):
            v = ing.get(key) or []
            if isinstance(v, list) and any(str(x).strip() for x in v):
                return True
    if ui.get("AllergenFilters"):
        return True
    price = ui.get("Price") or {}
    if isinstance(price, dict) and (
        price.get("Min") is not None or price.get("Max") is not None
    ):
        return True
    if ui.get("NutrientFilters"):
        return True
    df = ui.get("DietFilters")
    if isinstance(df, dict) and any(df.values()):
        return True
    if isinstance(df, list) and any(df):
        return True
    cf = ui.get("CuisineFilters") or {}
    if isinstance(cf, dict) and any(v is True or v is False for v in cf.values()):
        return True
    pm = ui.get("PreparationMethod") or {}
    if isinstance(pm, dict) and any(pm.values()):
        return True
    fp = ui.get("FlavorPreferences") or {}
    if isinstance(fp, dict) and any(str(v).strip() for v in fp.values()):
        return True
    if ui.get("Light") is True or ui.get("Hearty") is True or (
        ui.get("Balanced") is True and not ui.get("_balanced_suppressed_by_nutrients")
    ) or ui.get("Sustainable") is True:
        return True
    return False


def _heybo_empty_signature_message_for_named_request(ui: dict, signature_bowls: list) -> Optional[str]:
    """
    When the user asked for specific BowlName(s) but got zero bowls, use Heybo-specific copy
    explaining name / catalog vs location filters.
    Returns None if there are no specific names, or bowls were returned.
    """
    if signature_bowls:
        return None
    names = _requested_bowl_names_display(ui)
    if not names:
        return None
    reason = ui.get("_heybo_signature_empty_reason")
    bt = _bowl_type_display_signature(ui)
    quoted = ", ".join(f'"{n}"' for n in names)
    has_filters = _heybo_signature_payload_has_filters(ui)

    if reason == "no_candidates":
        if has_filters:
            return (
                f"We couldn't find a signature {bt} matching {quoted} with your current filters. "
                "Try a different name, Bowl Type, or relax filters."
            )
        return (
            f"We couldn't find a signature meal named like {quoted} for this bowl type. "
            "Check the spelling or Bowl Type."
        )

    if reason == "no_bowls_passed":
        if has_filters:
            return (
                f"We couldn't prepare {quoted} right now. "
                "Some ingredients may be unavailable at this location, or your filters may exclude this meal."
            )
        return (
            f"We couldn't prepare {quoted} at this location right now — "
            "some ingredients in this meal may be unavailable here."
        )

    if has_filters:
        return (
            f"We couldn't return a signature {bt} matching {quoted} with your current filters. "
            "Try another location or adjust filters."
        )
    return f"We couldn't return {quoted} right now. Try again later or pick another meal."


def _heybo_named_signature_success_message(ui: dict, signature_bowls: list) -> Optional[str]:
    """
    When BowlName is explicitly requested and we have results, use concise Heybo-specific
    success copy instead of generic Salad text. Especially useful when result is modified.
    """
    names = _requested_bowl_names_display(ui)
    bowls = signature_bowls or []
    if not names or not bowls:
        return None

    n = len(bowls)
    bt = _bowl_type_display_signature(ui)
    q = ", ".join(f'"{n_}"' for n_ in names)
    customized = [b for b in bowls if bool(b.get("customized"))]
    has_mod = bool(customized)

    ing = ui.get("Ingredients") or {}
    inc = [str(x).strip() for x in (ing.get("Include") or []) if str(x).strip()]
    exc = [str(x).strip() for x in (ing.get("Exclude") or []) if str(x).strip()]
    ext = [str(x).strip() for x in (ing.get("Extra") or []) if str(x).strip()]
    inc_l = {x.lower() for x in inc}
    ext_l = {x.lower() for x in ext}
    # Same string in Include and Extra: describe once as an extra portion, not "include X" + extra X.
    inc_standalone = [x for x in inc if x.lower() not in ext_l]
    ext_add_ons = [x for x in ext if x.lower() not in inc_l]
    ext_second_portion = [x for x in ext if x.lower() in inc_l]
    req_parts = []
    if inc_standalone:
        req_parts.append(f"include {', '.join(inc_standalone)}")
    if exc:
        req_parts.append(f"exclude {', '.join(exc)}")
    if ext_add_ons:
        req_parts.append(f"extra {', '.join(ext_add_ons)}")
    if ext_second_portion:
        req_parts.append(f"an extra portion of {', '.join(ext_second_portion)}")

    if n == 1:
        bowl_name = str(bowls[0].get("Bowl Name") or "that bowl")
        if has_mod:
            if req_parts:
                return f"We found {bowl_name} and tailored it to your request ({', '.join(req_parts)})."
            return f"We found {bowl_name} and tailored it to your request."
        return f"We found {bowl_name} for you."

    if has_mod:
        return f"We found {n} signature {bt}s for {q}, including options tailored to your request."
    return f"We found {n} signature {bt}s for {q}."


def _heybo_signature_user_message_native(signature_bowls: list, user_input: dict) -> str:
    """
    Generic signature success / empty copy (mirrors salad.generate_signature_user_message).
    """
    ui = dict(user_input) if isinstance(user_input, dict) else {}
    bowl_type = str(ui.get("BowlType") or ui.get("Bowl Type") or "bowl").lower()
    bowl_type_display = bowl_type.replace("_", " ").title()
    if "bowl" in bowl_type_display.lower():
        bowl_word = ""
        bowl_word_plural = "s"
    else:
        bowl_word = " bowl"
        bowl_word_plural = " bowls"

    exclude_ings = (ui.get("Ingredients") or {}).get("Exclude") or []
    extra_ings = (ui.get("Ingredients") or {}).get("Extra") or []
    allergen_filters = ui.get("AllergenFilters") or []
    price_filter = ui.get("Price") or {}
    cuisine_filters = ui.get("CuisineFilters") or {}
    prep_method_filters = ui.get("PreparationMethod") or {}
    flavor_preferences = ui.get("FlavorPreferences") or {}
    light = ui.get("Light") is True
    hearty = ui.get("Hearty") is True
    diet_filters = ui.get("DietFilters") or {}
    nutrient_filters = ui.get("NutrientFilters") or []

    bowls = signature_bowls or []
    if bowls:
        bowl_count = len(bowls)
        preferences_applied: List[str] = []

        if cuisine_filters and isinstance(cuisine_filters, dict):
            included_cuisines = [c.title() for c, include in cuisine_filters.items() if include is True]
            excluded_cuisines = [c.title() for c, include in cuisine_filters.items() if include is False]
            if included_cuisines:
                preferences_applied.append(f"{', '.join(included_cuisines)} cuisine")
            if excluded_cuisines:
                preferences_applied.append(f"excluded {', '.join(excluded_cuisines)} cuisine")

        if prep_method_filters and isinstance(prep_method_filters, dict):
            included_methods = [m.title() for m, include in prep_method_filters.items() if include is True]
            excluded_methods = [m.title() for m, include in prep_method_filters.items() if include is False]
            if included_methods:
                preferences_applied.append(f"{', '.join(included_methods)} preparation")
            if excluded_methods:
                preferences_applied.append(f"excluded {', '.join(excluded_methods)} preparation")

        if flavor_preferences:
            active_flavors = []
            for flavor, intensity in flavor_preferences.items():
                if intensity and str(intensity).strip():
                    active_flavors.append(f"{str(flavor).title()} ({str(intensity).strip().title()})")
            if active_flavors:
                preferences_applied.append(f"{', '.join(active_flavors)} flavor")

        if light and not hearty and not ui.get("_light_hearty_suppressed_by_nutrients"):
            preferences_applied.append("Light")
        elif hearty and not light and not ui.get("_light_hearty_suppressed_by_nutrients"):
            preferences_applied.append("Hearty")
        elif light and hearty and not ui.get("_light_hearty_suppressed_by_nutrients"):
            preferences_applied.append("Light & Hearty")

        if diet_filters:
            if isinstance(diet_filters, dict):
                active_diets = [d.title() for d, active in diet_filters.items() if active is True]
            elif isinstance(diet_filters, list):
                active_diets = [str(d).title() for d in diet_filters if d]
            else:
                active_diets = []
            if active_diets:
                preferences_applied.append(f"{', '.join(active_diets)} diet")

        if nutrient_filters:
            nutrient_list = []
            for nf in nutrient_filters:
                if not isinstance(nf, dict):
                    continue
                bit = _heybo_format_nutrient_filter_for_opener(nf)
                if bit:
                    nutrient_list.append(bit)
            if nutrient_list:
                preferences_applied.append(f"{', '.join(nutrient_list)}")

        customization_applied: List[str] = []
        if exclude_ings:
            customization_applied.append(f"removed {', '.join(str(x) for x in exclude_ings)}")
        if extra_ings:
            customization_applied.append(f"added {', '.join(str(x) for x in extra_ings)}")
        if allergen_filters:
            customization_applied.append(f"excluded {', '.join(str(x) for x in allergen_filters)} allergens")
        if isinstance(price_filter, dict) and price_filter.get("Max") is not None:
            customization_applied.append(f"under ${price_filter['Max']}")

        all_preferences = preferences_applied + customization_applied

        relaxation_notes: List[str] = []
        if ui.get("_flavor_filter_logic_used") == "OR" and flavor_preferences:
            active_flavor_count = sum(
                1 for pref in flavor_preferences.values() if pref and str(pref).strip()
            )
            if active_flavor_count > 1:
                relaxation_notes.append("flavor preferences")
        if ui.get("_cuisine_filter_logic_used") == "OR" and cuisine_filters:
            if isinstance(cuisine_filters, dict):
                included_count = sum(1 for include in cuisine_filters.values() if include is True)
                if included_count > 1:
                    relaxation_notes.append("cuisine preferences")
        if ui.get("_prep_method_filter_logic_used") == "OR" and prep_method_filters:
            if isinstance(prep_method_filters, dict):
                included_count = sum(1 for include in prep_method_filters.values() if include is True)
                if included_count > 1:
                    relaxation_notes.append("preparation method preferences")

        if all_preferences:
            preferences_text = f" matching your preferences: {', '.join(all_preferences)}"
        else:
            preferences_text = ""

        note_bits: List[str] = []
        if ui.get("_signature_price_relaxed"):
            note_bits.append("Price adjusted for the best available options")
        if relaxation_notes:
            note_bits.append(
                f"Some bowls may match only one of your {', '.join(relaxation_notes)}"
            )
        note_text = f" Note: {'; '.join(note_bits)}." if note_bits else ""

        if bowl_count == 1:
            return (
                f"Perfect! Found 1 signature {bowl_type_display}{bowl_word}{preferences_text}.{note_text}"
            )
        return (
            f"Great! Found {bowl_count} signature {bowl_type_display}{bowl_word_plural}"
            f"{preferences_text}.{note_text}"
        )

    reasons: List[str] = []
    if exclude_ings:
        reasons.append(
            f"all bowls contained your excluded ingredients ({', '.join(str(x) for x in exclude_ings)})"
        )
    if allergen_filters:
        reasons.append(f"all bowls contained {', '.join(str(x) for x in allergen_filters)} allergens")
    if isinstance(price_filter, dict) and price_filter.get("Max") is not None:
        reasons.append(f"all bowls exceeded your ${price_filter['Max']} budget")

    if cuisine_filters and isinstance(cuisine_filters, dict):
        included_cuisines = [c.title() for c, include in cuisine_filters.items() if include is True]
        excluded_cuisines = [c.title() for c, include in cuisine_filters.items() if include is False]
        if included_cuisines:
            reasons.append(
                f"no bowls matched your {', '.join(included_cuisines)} cuisine preference"
            )
        if excluded_cuisines:
            reasons.append(f"all bowls matched excluded cuisines ({', '.join(excluded_cuisines)})")

    if prep_method_filters and isinstance(prep_method_filters, dict):
        included_methods = [m.title() for m, include in prep_method_filters.items() if include is True]
        excluded_methods = [m.title() for m, include in prep_method_filters.items() if include is False]
        if included_methods:
            reasons.append(
                f"no bowls matched your {', '.join(included_methods)} preparation method preference"
            )
        if excluded_methods:
            reasons.append(
                f"all bowls matched excluded preparation methods ({', '.join(excluded_methods)})"
            )

    if light or hearty:
        if light and not hearty:
            reasons.append("no bowls matched your Light preference")
        elif hearty and not light:
            reasons.append("no bowls matched your Hearty preference")

    if flavor_preferences and any(pref and str(pref).strip() for pref in flavor_preferences.values()):
        reasons.append("no bowls matched your flavor preferences")

    if diet_filters:
        if isinstance(diet_filters, dict):
            active_diets = [d.title() for d, active in diet_filters.items() if active is True]
        elif isinstance(diet_filters, list):
            active_diets = [str(d).title() for d in diet_filters if d]
        else:
            active_diets = []
        if active_diets:
            reasons.append(f"no bowls matched your {', '.join(active_diets)} diet requirements")

    if nutrient_filters:
        reasons.append("no bowls matched your nutritional requirements")

    if reasons:
        reason_text = f" because {', '.join(reasons)}"
    else:
        reason_text = " - no signature bowls are currently available"

    suggestions: List[str] = []
    if exclude_ings:
        suggestions.append("try removing some ingredient exclusions")
    if allergen_filters:
        suggestions.append("consider removing some allergen restrictions")
    if isinstance(price_filter, dict) and price_filter.get("Max") is not None:
        suggestions.append("try increasing your budget")
    if cuisine_filters and isinstance(cuisine_filters, dict):
        included_cuisines = [c.title() for c, include in cuisine_filters.items() if include is True]
        if included_cuisines:
            suggestions.append("try broadening your cuisine preferences")
    if prep_method_filters and isinstance(prep_method_filters, dict):
        included_methods = [m.title() for m, include in prep_method_filters.items() if include is True]
        if included_methods:
            suggestions.append("try broadening your preparation method preferences")
    if light or hearty:
        suggestions.append("try adjusting your Light/Hearty preference")
    if flavor_preferences and any(pref and str(pref).strip() for pref in flavor_preferences.values()):
        suggestions.append("try adjusting your flavor preferences")
    if diet_filters:
        suggestions.append("try adjusting your diet requirements")
    if nutrient_filters:
        suggestions.append("try adjusting your nutritional requirements")

    if suggestions:
        suggestion_text = f" Here are some suggestions: {', '.join(suggestions)}."
    else:
        suggestion_text = " Please try again later or contact support."

    return (
        f"Sorry, we couldn't find any signature {bowl_type_display} bowls{reason_text}.{suggestion_text}"
    )


def build_heybo_signature_user_message(signature_bowls, user_input, error=None):
    """
    Client-facing copy for Heybo signature (preset) meals — Salad-shaped strings, Heybo-only.
    """
    ui = dict(user_input) if isinstance(user_input, dict) else {}
    if error:
        bt = (
            (ui.get("BowlType") or ui.get("Bowl Type") or "bowl")
            .lower()
            .replace("_", " ")
            .title()
        )
        return f"Sorry, we couldn't load signature {bt} options right now: {error}"

    named_empty = _heybo_empty_signature_message_for_named_request(ui, signature_bowls or [])
    if named_empty is not None:
        return named_empty
    named_success = _heybo_named_signature_success_message(ui, signature_bowls or [])
    if named_success is not None:
        return named_success

    return _heybo_signature_user_message_native(signature_bowls or [], ui)
