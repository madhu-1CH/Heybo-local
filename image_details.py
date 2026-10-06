"""
Heybo BYB image-details helpers.

Standalone 9-grid image URL logic for bowl responses.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

HEYBO_BYB_STATIC_IMAGE_URL = (
    "https://heybo-prod-ingredients-images.s3.ap-southeast-1.amazonaws.com/empty_image_heybo/Heybo+Logo.webp"
)
_EXTRA_DB_CATEGORIES = frozenset(
    {"Extra Warm sides", "Extra Cold sides", "Extra Proteins"}
)


def build_heybo_image_url_map(df: pd.DataFrame) -> Dict[str, Dict[Any, Optional[str]]]:
    """Build image maps from heybo.ingredients_details."""
    by_category_name: Dict[Tuple[str, str], Optional[str]] = {}
    by_name: Dict[str, Optional[str]] = {}
    if df is None or df.empty or "ingredient_name" not in df.columns:
        return {"by_category_name": by_category_name, "by_name": by_name}
    for _, row in df.iterrows():
        name = row.get("ingredient_name")
        if pd.isna(name):
            continue
        ing = str(name).strip()
        if not ing:
            continue
        cat_raw = row.get("category")
        cat_raw = "" if pd.isna(cat_raw) else str(cat_raw).strip()
        cat = cat_raw
        if cat in ("Warm sides", "Cold sides"):
            cat = "Sides"
        elif cat in ("Extra Warm sides", "Extra Cold sides"):
            cat = "Extra Sides"
        raw_url = row.get("image_url")
        if raw_url is None or (isinstance(raw_url, float) and pd.isna(raw_url)):
            url = None
        else:
            cleaned = str(raw_url).strip()
            url = cleaned if cleaned else None
        if not by_category_name.get((cat, ing)):
            by_category_name[(cat, ing)] = url
        if cat_raw not in _EXTRA_DB_CATEGORIES and not by_name.get(ing):
            by_name[ing] = url
    return {"by_category_name": by_category_name, "by_name": by_name}


def _image_lookup_category_for_ingredient(
    ingredient_name: Optional[str],
    bowl: Dict[str, Any],
) -> str:
    """Bowl placement → DB category (extra rows use -additional image URLs)."""
    if not ingredient_name:
        return "Bases"
    if ingredient_name in (bowl.get("Extra Proteins") or []):
        return "Extra Proteins"
    if ingredient_name in (bowl.get("Extra Warm sides") or []) or ingredient_name in (
        bowl.get("Extra Cold sides") or []
    ):
        return "Extra Sides"
    if ingredient_name in (bowl.get("Proteins") or []):
        return "Proteins"
    if ingredient_name in (bowl.get("Bases") or []):
        return "Bases"
    if ingredient_name in (bowl.get("Dips") or []):
        return "Dips"
    if ingredient_name in (bowl.get("Garnish") or []):
        return "Garnish"
    if ingredient_name in (bowl.get("Sauces") or []):
        return "Sauces"
    return "Sides"


def _lookup_image_url(
    image_maps: Dict[str, Dict[Any, Optional[str]]],
    category: str,
    ingredient_name: Optional[str],
) -> Optional[str]:
    if not ingredient_name:
        return None
    cat = category
    if cat in ("Warm sides", "Cold sides"):
        cat = "Sides"
    elif cat in ("Extra Warm sides", "Extra Cold sides"):
        cat = "Extra Sides"
    by_category_name = image_maps.get("by_category_name", {})
    by_name = image_maps.get("by_name", {})
    return by_category_name.get((cat, ingredient_name), by_name.get(ingredient_name))


def _pick_first(
    *category_lists: Tuple[str, List[str]],
    fallback: Tuple[str, Optional[str]],
) -> Tuple[str, Optional[str]]:
    """Pop first ingredient from the first non-empty list; else return fallback."""
    for category, lst in category_lists:
        if lst:
            return (category, lst.pop(0))
    return fallback


def build_heybo_dynamic_grid_image_details(
    bowl: Dict[str, Any],
    image_maps: Dict[str, Dict[Any, Optional[str]]],
) -> Dict[str, Any]:
    """
    Dynamic 9-grid (slot order): base | sides 2–4 | proteins 5–6 | sides 7–8 | dip.

    Garnish and Sauces are preferred over Bases when filling protein and side slots;
    Bases remain the final fallback. Pick order: 2–3, 5–6, 7–8, then 4 (extra showcase).
    Image URLs use bowl placement (extras → -additional URLs from DB).
    """
    bases = list(bowl.get("Bases") or [])

    warm_sides = list(bowl.get("Warm sides") or [])
    cold_sides = list(bowl.get("Cold sides") or [])

    extra_warm = list(bowl.get("Extra Warm sides") or [])
    extra_cold = list(bowl.get("Extra Cold sides") or [])
    extra_sides = extra_warm + extra_cold

    proteins = list(bowl.get("Proteins") or []) + list(bowl.get("Extra Proteins") or [])

    dips = list(bowl.get("Dips") or [])
    garnish = list(bowl.get("Garnish") or [])
    sauce = list(bowl.get("Sauces") or [])

    base_fallback = bases[0] if bases else None
    base_slot: Tuple[str, Optional[str]] = ("Bases", base_fallback)

    remaining_warm = warm_sides.copy()
    remaining_cold = cold_sides.copy()
    remaining_extra = extra_sides.copy()

    # Slots 2 & 3: Warm -> Cold -> Base
    slot2 = _pick_first(
        ("Warm sides", remaining_warm),
        ("Cold sides", remaining_cold),
        fallback=base_slot,
    )
    slot3 = _pick_first(
        ("Warm sides", remaining_warm),
        ("Cold sides", remaining_cold),
        fallback=base_slot,
    )

    # Slot 5: Protein -> Garnish -> Sauce -> Base
    slot5 = _pick_first(
        ("Proteins", proteins),
        ("Garnish", garnish),
        ("Sauces", sauce),
        fallback=base_slot,
    )

    # Slot 6: next Protein / Extra Protein -> Sauce -> Garnish -> Base
    slot6 = _pick_first(
        ("Extra Proteins", proteins),
        ("Sauces", sauce),
        ("Garnish", garnish),
        fallback=base_slot,
    )

    # Slots 7 & 8: remaining normal sides; slot 8 may use Garnish/Sauce before Base
    slot7 = _pick_first(
        ("Cold sides", remaining_cold),
        ("Warm sides", remaining_warm),
        fallback=base_slot,
    )
    slot8 = _pick_first(
        ("Cold sides", remaining_cold),
        ("Warm sides", remaining_warm),
        ("Garnish", garnish),
        ("Sauces", sauce),
        fallback=base_slot,
    )

    # Slot 4: Extra -> leftover sides -> Garnish -> Sauce -> Base
    slot4 = _pick_first(
        ("Extra Sides", remaining_extra),
        ("Cold sides", remaining_cold),
        ("Warm sides", remaining_warm),
        ("Garnish", garnish),
        ("Sauces", sauce),
        fallback=base_slot,
    )

    slots: List[Tuple[str, Optional[str]]] = [
        ("Bases", bases[0] if bases else base_fallback),
        slot2,
        slot3,
        slot4,
        slot5,
        slot6,
        slot7,
        slot8,
        ("Dips", dips[0] if dips else base_fallback),
    ]

    urls: List[Optional[str]] = []
    for _grid_category, ing in slots:
        lookup_cat = _image_lookup_category_for_ingredient(ing, bowl)
        url = _lookup_image_url(image_maps, lookup_cat, ing)
        if url is None and not bases and HEYBO_BYB_STATIC_IMAGE_URL:
            url = HEYBO_BYB_STATIC_IMAGE_URL
        urls.append(url)
    return {"image_type": "dynamic", "image": urls}
