"""
Load Heybo runtime config from heybo.* tables (no static business defaults in config.py).
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

import pandas as pd

_pkg_dir = Path(__file__).resolve().parent
_repo_root = _pkg_dir.parent
load_dotenv(_repo_root / ".env")
load_dotenv(_pkg_dir / ".env")
load_dotenv()

from .config import DB_CONFIG
from .db import get_db_connection, return_db_connection


def _heybo_meal_persistence_tables() -> List[str]:
    """
    Tables that may hold persisted BYB rows for the same DB connection.

    ``HEYBO_RECOMMENDATION_MEALS_TABLE`` (optional env) overrides **only** this persistence table.
    **Testing only:** point at another schema on the same DB, e.g.
    ``heybo_dev.recommendation_engine_generated_meals``, while the rest of ``get_heybo_config`` still
    uses ``heybo.*``. Unset in production unless writers and readers both use that table.
    When unset, default is ``heybo.recommendation_engine_generated_meals``.
    """
    v = (os.getenv("HEYBO_RECOMMENDATION_MEALS_TABLE") or "").strip()
    if v:
        return [v]
    return ["heybo.recommendation_engine_generated_meals"]


def _heybo_recommendation_meals_table() -> str:
    """Primary persistence table (first entry from ``_heybo_meal_persistence_tables``)."""
    return _heybo_meal_persistence_tables()[0]

# Ingredient columns on ``recommendation_engine_generated_meals`` (snake_case), same coverage as bowl JSON.
_HEYBO_GEN_MEALS_INGREDIENT_COLUMNS = (
    "bases",
    "proteins",
    "extra_proteins",
    "warm_sides",
    "extra_warm_sides",
    "cold_sides",
    "extra_cold_sides",
    "dips",
    "garnish",
    "sauces",
)

_HEYBO_GEN_MEALS_ACTIVE_CLAUSE = " AND (delete_status IS NULL OR delete_status = 0)"

REQUIRED_BOWL_CATEGORIES = (
    "Bases",
    "Proteins",
    "Warm sides",
    "Cold sides",
    "Dips",
    "Garnish",
    "Sauces",
)

# Ingredient flavor score columns (must match heybo.db.process_heybo_ingredient_data).
HEYBO_FLAVOR_COLUMNS = ("sweet", "sour", "salty", "bitter", "spicy", "umami")

_LULU_BYB_ONLY_BASE = "lulu-byb"


def _heybo_session_recommend_pair_where_params(
    recommend_page_id: str, session_id: str
) -> Tuple[str, Tuple[str, str, str, str]]:
    """
    SQL fragment + bind values so one row matches whether the client sent ids aligned with DB
    columns or swapped (same two UUIDs, different JSON keys).
    """
    rid = (recommend_page_id or "").strip()
    sid = (session_id or "").strip()
    wh = (
        "((recommend_page_id = %s AND session_id = %s) "
        "OR (recommend_page_id = %s AND session_id = %s))"
    )
    return wh, (rid, sid, sid, rid)


# --- BYB "Bowl Name" numbering (same session / same flow) -----------------------------------------
# The DB stores one string per saved bowl in column ``bowl_name``, e.g. ``Salmon_Mayo`` or
# ``Salmon_Mayo - 2``. For the next API call we only need to know, per *base* string, the
# largest suffix already used so new bowls become ``… - 3``, ``… - 4``, and do not clash.
# We read prior rows (session_id / recommend_page_id), strip a trailing `` - N`` from
# ``bowl_name``, and keep ``previous_bowl_name_index_by_base`` = { base: highest N }.
# The generator also labels using ``firstProtein_firstSauce`` from ingredients; we store
# the same highest N under that key when ``proteins``/``sauces`` are present so both match.
# ------------------------------------------------------------------------------------------------


def _heybo_cell_to_items(val: Any) -> list:
    """Normalize DB cell (list, CSV string, scalar) to a flat list of non-empty values."""
    if val is None or val == "":
        return []
    if isinstance(val, list):
        return [x for x in val if x is not None and str(x).strip() != ""]
    if isinstance(val, str):
        return [i.strip() for i in val.split(",") if i.strip()]
    return [val]


def _heybo_first_cell_item(val: Any) -> Optional[str]:
    items = _heybo_cell_to_items(val)
    if not items:
        return None
    s = str(items[0]).strip()
    return s or None


def _heybo_ingredient_primary_bowl_key(rd: dict) -> Optional[str]:
    """Build ``Protein_Sauce``-style key from row columns (same rule as ``generation``)."""
    p = None
    for col in ("proteins", "extra_proteins"):
        items = _heybo_cell_to_items(rd.get(col))
        if items:
            p = str(items[0]).strip() or None
            break
    s = _heybo_first_cell_item(rd.get("sauces"))
    if p and s:
        return f"{p}_{s}"
    if s:
        return str(s)
    if p:
        return "Lulu-BYB"
    return None


def _heybo_non_lulu_stored_bowl_name_base_index(name: str) -> Tuple[str, int]:
    """Split ``bowl_name`` into (base without suffix, suffix index). ``Salmon_Mayo - 2`` → (``Salmon_Mayo``, 2)."""
    s = (name or "").strip()
    if not s:
        return "", 0
    sep = " - "
    i = s.rfind(sep)
    if i != -1:
        tail = s[i + len(sep) :].strip()
        if tail.isdigit():
            base = s[:i].strip()
            if base:
                return base, int(tail)
    j = s.rfind("-")
    if j != -1 and j < len(s) - 1:
        tail = s[j + 1 :].strip()
        if tail.isdigit():
            base = s[:j].strip()
            if base:
                return base, int(tail)
    return s, 1


def _heybo_bowl_name_index_map_from_meal_df(df: Any) -> Dict[str, int]:
    """Per base string, highest ``- N`` already seen on rows in this dataframe (one session slice)."""
    out: Dict[str, int] = {}
    if df is None or getattr(df, "empty", True):
        return out
    for _, row in df.iterrows():
        rd = {str(k).lower(): v for k, v in row.items()}
        idx_row = 1
        bases: Set[str] = set()
        bn = rd.get("bowl_name")
        if bn is not None and not (isinstance(bn, float) and pd.isna(bn)):
            raw = str(bn).strip()
            if raw:
                base_bn, idx_row = _heybo_non_lulu_stored_bowl_name_base_index(raw)
                if base_bn:
                    bases.add(base_bn)
        ing_key = _heybo_ingredient_primary_bowl_key(rd)
        if ing_key:
            bases.add(ing_key)
        if not bases:
            continue
        for b in bases:
            out[b] = max(out.get(b, 0), idx_row)
    return out


def _heybo_merge_max_index_dict(dest: Dict[str, int], src: Dict[str, int]) -> None:
    for k, v in src.items():
        dest[k] = max(dest.get(k, 0), v)


def _heybo_max_bowl_name_index_by_base(names: List[str]) -> Dict[str, int]:
    """
    For each logical base name (e.g. ``Salmon_Sauce``), largest used suffix index from DB rows
    in this session/page. Used to continue ``…``, ``… - 2``, ``… - 3`` across requests.
    """
    out: Dict[str, int] = {}
    for raw in names:
        s = (raw or "").strip()
        if not s:
            continue
        low = s.lower()
        if low == _LULU_BYB_ONLY_BASE:
            out["Lulu-BYB"] = max(out.get("Lulu-BYB", 0), 1)
            continue
        if low.startswith(_LULU_BYB_ONLY_BASE):
            tail = low[len(_LULU_BYB_ONLY_BASE) :].lstrip()
            if not tail.startswith("-"):
                continue
            num_part = tail[1:].lstrip()
            digits = ""
            for ch in num_part:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                out["Lulu-BYB"] = max(out.get("Lulu-BYB", 0), int(digits))
            continue
        base, idx = _heybo_non_lulu_stored_bowl_name_base_index(s)
        if base:
            out[base] = max(out.get(base, 0), idx)
    return out


def _heybo_max_lulu_byb_only_mode_index(names: List[str]) -> int:
    """
    Largest used index for Only-mode BYB names in a session/page.
    Treats legacy bare ``Lulu-BYB`` as index 1; ``Lulu-BYB - 2`` / ``Lulu-BYB-2`` as 2, etc.
    """
    return _heybo_max_bowl_name_index_by_base(names).get("Lulu-BYB", 0)


def _column_ci(columns, target: str) -> Optional[str]:
    """Return the actual DataFrame column name matching target (case-insensitive), or None."""
    t = target.lower()
    for c in columns:
        if str(c).lower() == t:
            return str(c)
    return None


def _required_band_int(series: pd.Series, col: str, *, row_label: str, field: str) -> int:
    """Coerce flavor band bound; DDL allows NULL numeric — fail with a clear message if missing."""
    v = series[col]
    if v is None or pd.isna(v):
        raise ValueError(
            f"heybo.flavor_profile_details: {field} is NULL for {row_label}; "
            "all six band columns must be set per flavor row."
        )
    return int(float(v))


def _heybo_parse_positive_int_page(page: Any) -> Optional[int]:
    try:
        n = int(page)
        return n if n >= 1 else None
    except (TypeError, ValueError):
        return None


def _heybo_accumulate_generated_meals_for_dedupe(
    df: Any,
    into_bowls: Set[frozenset],
    into_primary_bowl_keys: Set[str],
) -> None:
    """
    Merge ingredient sets from meal rows, and primary identity keys from ``bowl_name``
    (normalized base without `` - N`` suffix), matching ``generation`` bowl naming.
    """
    if df is None or getattr(df, "empty", True):
        return
    for _, row in df.iterrows():
        rd = {str(k).lower(): v for k, v in row.items()}
        ing: List[Any] = []
        for c in _HEYBO_GEN_MEALS_INGREDIENT_COLUMNS:
            v = rd.get(c)
            if isinstance(v, list):
                ing.extend(v)
            elif v is not None and v != "":
                ing.append(v)
        into_bowls.add(frozenset(ing))
        bn = rd.get("bowl_name")
        if bn is not None and str(bn).strip():
            base, _ = _heybo_non_lulu_stored_bowl_name_base_index(str(bn).strip())
            if base:
                into_primary_bowl_keys.add(base)


def _heybo_load_meal_dedupe_from_db(
    conn,
    recommend_page_id: str,
    session_id: str,
    page: Optional[int],
) -> Tuple[Set[frozenset], Set[str], bool]:
    """
    Returns (previous_bowls, previous_protein_sauce_combos, merged_session_orphans).

    ``previous_protein_sauce_combos`` is a set of **normalized ``bowl_name`` bases** (same keys
    ``generation`` uses for primary-bowl diversity), not raw DB tuples.

    Prior meals are accumulated from:

    - **All rows for ``session_id``** when it is set (stable across pages; ``recommend_page_id`` often
      changes every page, so filtering only on the current page id misses earlier pages).
    - **All rows for ``recommend_page_id``** when set (covers page-scoped saves and complements session).
    - When **both** ids are set, also rows matching the swap-tolerant pair clause (same two UUIDs,
      either column order in the request).
    - When ``Page`` > 1 with both ids: additionally same-session rows with blank ``recommend_page_id``
      and ``page_no`` null or 1 (orphan page-1 saves); tries ``session_id`` from the request first,
      then the other id if the client swapped keys.
    - **Persistence table:** each candidate from ``_heybo_meal_persistence_tables`` is queried in turn
      (missing relation or columns skips that table).
    """
    rid = (recommend_page_id or "").strip()
    sid = (session_id or "").strip()
    page_i = _heybo_parse_positive_int_page(page)
    out: Set[frozenset] = set()
    primary_keys: Set[str] = set()
    merged_orphans = False
    cols = ", ".join(_HEYBO_GEN_MEALS_INGREDIENT_COLUMNS) + ", bowl_name"
    active = _HEYBO_GEN_MEALS_ACTIVE_CLAUSE

    def acc(df: Any) -> None:
        _heybo_accumulate_generated_meals_for_dedupe(df, out, primary_keys)

    for tbl in _heybo_meal_persistence_tables():
        sel = f" SELECT {cols} FROM {tbl} "
        try:
            if sid:
                acc(pd.read_sql(sel + f" WHERE session_id = %s{active}", conn, params=(sid,)))
            if rid:
                acc(
                    pd.read_sql(
                        sel + f" WHERE recommend_page_id = %s{active}",
                        conn,
                        params=(rid,),
                    )
                )
            if rid and sid:
                pair_sql, pair_params = _heybo_session_recommend_pair_where_params(rid, sid)
                acc(
                    pd.read_sql(
                        sel + f" WHERE {pair_sql}{active}",
                        conn,
                        params=pair_params,
                    )
                )
            if rid and sid and page_i is not None and page_i > 1:
                for orphan_sid in (sid, rid):
                    df_o = pd.read_sql(
                        sel
                        + """ WHERE session_id = %s
                              AND (recommend_page_id IS NULL OR TRIM(CAST(recommend_page_id AS TEXT)) = '')
                              AND (page_no IS NULL OR page_no = 1)
                        """
                        + active,
                        conn,
                        params=(orphan_sid,),
                    )
                    acc(df_o)
                merged_orphans = True
        except Exception:
            continue

    return out, primary_keys, merged_orphans


def _require_category_map(
    d: Dict[str, Any],
    label: str,
    flow: str,
) -> Dict[str, tuple]:
    if not d:
        raise ValueError(
            f"heybo.ingredient_wise_count has no rows for flow_type={flow!r} ({label}). "
            "Populate the table or fix flow_type values."
        )
    missing = [c for c in REQUIRED_BOWL_CATEGORIES if c not in d]
    if missing:
        raise ValueError(
            f"heybo.ingredient_wise_count ({label}, flow_type={flow!r}) is missing categories: {missing}"
        )
    return d


def get_heybo_config(
    recommend_page_id: str = "",
    session_id: str = "",
    *,
    page: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Load Heybo config from DB only.
    Returned dict: category_limits, category_limits_customization, category_limits_extras, flavor_thresholds, price_config,
    max_bowl_weight, previous_bowls, previous_protein_sauce_combos, lulu_byb_only_mode_max_index,
    previous_bowl_name_index_by_base.

    **Bowl display names (simple idea):** ``previous_bowl_name_index_by_base`` is a small dict.
    Each key is a bowl *base* string (the part of ``bowl_name`` before any `` - 2`` suffix). Each
    value is the **largest suffix index** from stored ``bowl_name`` values (``1`` means the base
    appeared without a `` - N`` tail, or the parsed number from ``… - N``). The generator bumps
    that counter for each new bowl of the same base and sets ``Bowl Name`` to ``base`` when the
    new count is 1, otherwise ``base - {count}`` (so after ``Salmon_Mayo - 2`` in the DB, the next
    new bowl becomes ``Salmon_Mayo - 3``).

    ``HEYBO_RECOMMENDATION_MEALS_TABLE`` (env) pins one table; otherwise meal history reads
    ``heybo.recommendation_engine_generated_meals``.

    Meal history (ingredient-set dedupe + primary bowl keys from ``bowl_name``):

    - **Session-scoped:** all active rows with ``session_id`` when it is set (pagination often issues
      a new ``recommend_page_id`` each page while the session id stays the same).
    - **Page-scoped:** rows with ``recommend_page_id`` when set.
    - When both ids are set, also rows matching the swap-tolerant ``(recommend_page_id, session_id)``
      pair clause.
    - **Tables:** each candidate from ``_heybo_meal_persistence_tables`` is tried until the
      ingredient ``SELECT`` succeeds (missing relation or columns skips that table).
    - If ``Page`` > 1 with both ids: also merge orphan rows (blank ``recommend_page_id``, ``page_no``
      null or 1), trying ``session_id`` from the request first then the other id for swap tolerance.
    - ``previous_protein_sauce_combos``: set of normalized ``bowl_name`` bases (no `` - N``) used so
      the generator avoids repeating the same primary label when other pairs are still available.

    Name numbering loads ``bowl_name`` plus ``proteins``/``sauces`` from the same table so the
    string derived from columns matches the counter derived from ``bowl_name``.
    That load runs in its own try block so a failure in the meal-dedupe query does not skip name history.
    """
    cfg: Dict[str, Any] = {
        "category_limits": {},
        "category_limits_customization": {},
        "category_limits_extras": {},
        "flavor_thresholds": {},
        "co2_config": {},
        "price_config": {},
        "max_bowl_weight": None,
        "previous_bowls": set(),
        "previous_protein_sauce_combos": set(),
        "lulu_byb_only_mode_max_index": 0,
        "previous_bowl_name_index_by_base": {},
        "_heybo_meal_dedupe_merged_session_orphans": False,
    }

    conn = None
    try:
        conn = get_db_connection(DB_CONFIG)

        # --- ingredient_wise_count: normal (default bowl generation limits) ---
        df = pd.read_sql(
            """
            SELECT display_category_name, min, max
            FROM heybo.ingredient_wise_count
            WHERE LOWER(flow_type) = 'normal'
            """,
            conn,
        )
        if df.empty or "display_category_name" not in df.columns:
            raise ValueError(
                "heybo.ingredient_wise_count has no rows for flow_type='normal' "
                "or is missing display_category_name."
            )
        limits = {}
        for _, row in df.iterrows():
            name = row["display_category_name"]
            lo = int(row["min"]) if pd.notna(row.get("min")) else 1
            hi = int(row["max"]) if pd.notna(row.get("max")) else 1
            limits[name] = (lo, hi)
        cfg["category_limits"] = _require_category_map(limits, "normal", "normal")

        # --- ingredient_wise_count: customization (used when user requested count exceeds normal max) ---
        df = pd.read_sql(
            """
            SELECT display_category_name, min, max
            FROM heybo.ingredient_wise_count
            WHERE LOWER(flow_type) = 'customization'
            """,
            conn,
        )
        limits_customization = {}
        for _, row in df.iterrows():
            name = row["display_category_name"]
            lo = int(row["min"]) if pd.notna(row.get("min")) else 0
            hi = int(row["max"]) if pd.notna(row.get("max")) else 1
            limits_customization[name] = (lo, hi)
        if limits_customization:
            cfg["category_limits_customization"] = _require_category_map(
                limits_customization, "customization", "customization"
            )
        else:
            cfg["category_limits_customization"] = dict(cfg["category_limits"])

        # Backward compatibility for existing extras logic.
        cfg["category_limits_extras"] = dict(cfg["category_limits_customization"])

        # --- flavor_profile_details (salad-aligned shape) ---
        # Table shape: flavor_name + low_*/no_*/high_* min/max scores (see heybo.flavor_profile_details DDL).
        # Loaded as { flavor_name.lower(): {"Low"|"No"|"High": (min, max)}, ... }.
        # Per-flavor rows when flavor_name column exists; else first row bands copied to all ingredient score columns.
        df = pd.read_sql("SELECT * FROM heybo.flavor_profile_details", conn)
        if df.empty:
            raise ValueError("heybo.flavor_profile_details has no rows; add at least one row for flavor bands.")
        band_keys = (
            "low_min_score",
            "low_max_score",
            "no_min_score",
            "no_max_score",
            "high_min_score",
            "high_max_score",
        )
        col_map = {k: _column_ci(df.columns, k) for k in band_keys}
        missing = [k for k, v in col_map.items() if v is None]
        if missing:
            raise ValueError(f"heybo.flavor_profile_details is missing columns: {missing}")

        def row_bands(series: pd.Series, *, row_label: str) -> Dict[str, tuple]:
            cm = col_map
            return {
                "Low": (
                    _required_band_int(series, cm["low_min_score"], row_label=row_label, field="low_min_score"),
                    _required_band_int(series, cm["low_max_score"], row_label=row_label, field="low_max_score"),
                ),
                "No": (
                    _required_band_int(series, cm["no_min_score"], row_label=row_label, field="no_min_score"),
                    _required_band_int(series, cm["no_max_score"], row_label=row_label, field="no_max_score"),
                ),
                "High": (
                    _required_band_int(series, cm["high_min_score"], row_label=row_label, field="high_min_score"),
                    _required_band_int(series, cm["high_max_score"], row_label=row_label, field="high_max_score"),
                ),
            }

        flavor_name_col = _column_ci(df.columns, "flavor_name")
        if flavor_name_col:
            flavor_thresholds: Dict[str, Any] = {}
            for _, row in df.iterrows():
                raw = row.get(flavor_name_col)
                if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                    continue
                name = str(raw).strip().lower()
                if not name:
                    continue
                flavor_thresholds[name] = row_bands(row, row_label=f"flavor_name={name!r}")
            if not flavor_thresholds:
                raise ValueError(
                    "heybo.flavor_profile_details has a flavor_name column but no rows with a non-empty flavor_name; "
                    "check data."
                )
            cfg["flavor_thresholds"] = flavor_thresholds
        else:
            r = df.iloc[0]
            bands = row_bands(r, row_label="global row (no flavor_name column)")
            cfg["flavor_thresholds"] = {f: dict(bands) for f in HEYBO_FLAVOR_COLUMNS}

        # --- ingredient_wise_count: Pricing (tier thresholds) ---
        pricing_df = pd.read_sql(
            """
            SELECT display_category_name, min, max
            FROM heybo.ingredient_wise_count
            WHERE LOWER(flow_type) = 'pricing'
            """,
            conn,
        )
        if pricing_df.empty:
            raise ValueError("heybo.ingredient_wise_count must include flow_type='Pricing' rows")
        pricing_limits = {
            str(row["display_category_name"]): (
                int(row["min"]) if pd.notna(row.get("min")) else 0,
                int(row["max"]) if pd.notna(row.get("max")) else 0,
            )
            for _, row in pricing_df.iterrows()
        }

        # --- preference_filters: pricing + bowl weight + Light/Hearty bands ---
        # Fetching all required preference_filters rows in one query avoids the
        # separate DB round-trip that light_hearty.py would otherwise make lazily.
        df = pd.read_sql(
            """
            SELECT filter_name, filter_value
            FROM heybo.preference_filters
            WHERE filter_name IN (
                'BYB_Min_Price', 'BYB_Max_Price', 'MaxBowlPrice', 'MaxBowlWeight',
                'CO2_Sustainable_Min', 'CO2_Sustainable_Max',
                'Light_Min', 'Light_Max', 'Hearty_Min', 'Hearty_Max',
                'Light_Calories_Min', 'Light_Calories_Max',
                'Light_Protein_Min', 'Light_Protein_Max',
                'Light_Fiber_Min', 'Light_Fiber_Max',
                'Light_Weight_Min', 'Light_Weight_Max',
                'Hearty_Calories_Min', 'Hearty_Calories_Max',
                'Hearty_Protein_Min', 'Hearty_Protein_Max',
                'Hearty_Fiber_Min', 'Hearty_Fiber_Max',
                'Hearty_Weight_Min', 'Hearty_Weight_Max'
            )
            """,
            conn,
        )
        if df.empty:
            raise ValueError(
                "heybo.preference_filters must define BYB_Min_Price and MaxBowlWeight."
            )
        row_map = dict(zip(df["filter_name"], df["filter_value"]))
        required_pf = ("BYB_Min_Price", "MaxBowlWeight")
        missing_pf = [k for k in required_pf if k not in row_map]
        if missing_pf:
            raise ValueError(f"heybo.preference_filters is missing: {missing_pf}")
        cfg["price_config"] = {
            # BYB_Min_Price ≈ Salad CYOPrice (platform floor / base in bowl total)
            "base_price": float(row_map["BYB_Min_Price"]),
            # BYB_Max_Price ≈ Salad CYOPriceMax (CYO / user-facing price band ceiling — not generatable max)
            "byb_max_price": float(row_map["BYB_Max_Price"])
            if row_map.get("BYB_Max_Price") not in (None, "")
            else None,
            # MaxBowlPrice ≈ Salad MaxSaladPrice (highest total the engine can actually build)
            "max_bowl_price": float(row_map["MaxBowlPrice"])
            if row_map.get("MaxBowlPrice") not in (None, "")
            else None,
            "pricing_limits": pricing_limits,
        }
        if row_map.get("CO2_Sustainable_Min") not in (None, "") and row_map.get("CO2_Sustainable_Max") not in (None, ""):
            cfg["co2_config"] = {
                "Sustainable": {
                    "min": float(row_map["CO2_Sustainable_Min"]),
                    "max": float(row_map["CO2_Sustainable_Max"]),
                }
            }
        cfg["max_bowl_weight"] = int(row_map["MaxBowlWeight"])

        # --- Light/Hearty ingredient score bands + bowl validation ranges ---
        # Parsed here so light_hearty.py never needs its own DB call.
        _lh_score_config: dict = {}
        if all(k in row_map for k in ("Light_Min", "Light_Max", "Hearty_Min", "Hearty_Max")):
            _lh_score_config["Light"] = {
                "preferred": (int(float(row_map["Light_Min"])), int(float(row_map["Light_Max"]))),
                "fallback": (int(float(row_map["Hearty_Min"])), int(float(row_map["Hearty_Max"]))),
            }
            _lh_score_config["Hearty"] = {
                "preferred": (int(float(row_map["Hearty_Min"])), int(float(row_map["Hearty_Max"]))),
                "fallback": (int(float(row_map["Light_Min"])), int(float(row_map["Light_Max"]))),
            }
        _lh_bowl_keys_light = (
            "Light_Calories_Min", "Light_Calories_Max",
            "Light_Protein_Min", "Light_Protein_Max",
            "Light_Fiber_Min", "Light_Fiber_Max",
            "Light_Weight_Min", "Light_Weight_Max",
        )
        _lh_bowl_keys_hearty = (
            "Hearty_Calories_Min", "Hearty_Calories_Max",
            "Hearty_Protein_Min", "Hearty_Protein_Max",
            "Hearty_Fiber_Min", "Hearty_Fiber_Max",
            "Hearty_Weight_Min", "Hearty_Weight_Max",
        )
        _light_bowl: dict = {}
        if all(k in row_map for k in _lh_bowl_keys_light):
            _light_bowl = {
                "calories_range": (int(float(row_map["Light_Calories_Min"])), int(float(row_map["Light_Calories_Max"]))),
                "protein_range": (int(float(row_map["Light_Protein_Min"])), int(float(row_map["Light_Protein_Max"]))),
                "fiber_range": (int(float(row_map["Light_Fiber_Min"])), int(float(row_map["Light_Fiber_Max"]))),
                "weight_range": (int(float(row_map["Light_Weight_Min"])), int(float(row_map["Light_Weight_Max"]))),
            }
        _hearty_bowl: dict = {}
        if all(k in row_map for k in _lh_bowl_keys_hearty):
            _hearty_bowl = {
                "calories_range": (int(float(row_map["Hearty_Calories_Min"])), int(float(row_map["Hearty_Calories_Max"]))),
                "protein_range": (int(float(row_map["Hearty_Protein_Min"])), int(float(row_map["Hearty_Protein_Max"]))),
                "fiber_range": (int(float(row_map["Hearty_Fiber_Min"])), int(float(row_map["Hearty_Fiber_Max"]))),
                "weight_range": (int(float(row_map["Hearty_Weight_Min"])), int(float(row_map["Hearty_Weight_Max"]))),
            }
        cfg["light_hearty_config"] = {
            "light_hearty": _lh_score_config,
            "light_hearty_bowl_validation": {
                "Light": _light_bowl,
                "Hearty": _hearty_bowl,
            },
        }

        # --- previous bowls (dedupe) + bowl name suffixes for this session ---
        if recommend_page_id or session_id:
            try:
                pb, pc, merged_orphans = _heybo_load_meal_dedupe_from_db(
                    conn, recommend_page_id, session_id, page
                )
                cfg["previous_bowls"] = pb
                cfg["previous_protein_sauce_combos"] = pc
                cfg["_heybo_meal_dedupe_merged_session_orphans"] = merged_orphans
            except Exception:
                pass

            # Highest `` - N`` per base from ``bowl_name`` (and same count under protein_sauce key
            # from columns so ``generation`` picks the next number correctly). Independent try so a
            # failed dedupe SELECT does not skip this.
            try:
                active = _HEYBO_GEN_MEALS_ACTIVE_CLAUSE
                rid = (recommend_page_id or "").strip()
                sid = (session_id or "").strip()
                by_base: Dict[str, int] = {}
                for tbl in _heybo_meal_persistence_tables():
                    hint_sql = (
                        f"""
                        SELECT bowl_name, proteins, extra_proteins, sauces
                        FROM {tbl}
                        """
                    ).strip()
                    try:
                        if sid:
                            df_s = pd.read_sql(
                                hint_sql + f" WHERE session_id = %s" + active,
                                conn,
                                params=(sid,),
                            )
                            _heybo_merge_max_index_dict(
                                by_base, _heybo_bowl_name_index_map_from_meal_df(df_s)
                            )
                        if rid:
                            df_r = pd.read_sql(
                                hint_sql + f" WHERE recommend_page_id = %s" + active,
                                conn,
                                params=(rid,),
                            )
                            _heybo_merge_max_index_dict(
                                by_base, _heybo_bowl_name_index_map_from_meal_df(df_r)
                            )
                        if sid and rid:
                            pair_sql, pair_params = (
                                _heybo_session_recommend_pair_where_params(rid, sid)
                            )
                            df_p = pd.read_sql(
                                hint_sql + f" WHERE {pair_sql}" + active,
                                conn,
                                params=pair_params,
                            )
                            _heybo_merge_max_index_dict(
                                by_base, _heybo_bowl_name_index_map_from_meal_df(df_p)
                            )
                    except Exception:
                        continue
                if by_base:
                    cfg["previous_bowl_name_index_by_base"] = by_base
                    cfg["lulu_byb_only_mode_max_index"] = int(by_base.get("Lulu-BYB", 0) or 0)
            except Exception:
                pass

    finally:
        if conn:
            return_db_connection(conn, DB_CONFIG)

    return cfg
