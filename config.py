"""
Heybo configuration: database connection settings only.
Runtime rules (category counts, pricing, flavor bands, nutrition constraints) load from heybo.* tables
via config_loader.get_heybo_config and nutrition_constraints.load_heybo_nutrition_constraints_from_db.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from package / repo roots, not only cwd (Salad parity).
# generation.py imports this module before config_loader, so DB_CONFIG must
# already see DB_HOST here — otherwise psycopg2 falls back to localhost.
_pkg_dir = Path(__file__).resolve().parent
_repo_root = _pkg_dir.parent
for _env_path in (
    _repo_root / ".env",
    _pkg_dir / ".env",
    Path.cwd() / ".env",
):
    if _env_path.is_file():
        load_dotenv(_env_path, override=False)
load_dotenv(override=False)

# CP-SAT first-pass search for CYO (not Signatures). Random generation remains the fallback.
CPSAT_ENABLED = True
CPSAT_MAX_SOLUTIONS = 5
CPSAT_TIME_LIMIT_SECONDS = 15.0
# OR-Tools threads per solve (one request). Same value in UAT and prod.
CPSAT_SEARCH_WORKERS = 8

DEBUG_ENABLED = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes", "on")

if DEBUG_ENABLED:
    def dbg_print(*args, **kwargs):
        if args and isinstance(args[0], str) and ("{}" in args[0] or ("{" in args[0] and "}" in args[0])):
            try:
                evaluated_args = []
                for arg in args[1:]:
                    evaluated_args.append(arg() if callable(arg) else arg)
                formatted = args[0].format(*evaluated_args) if evaluated_args else args[0]
                print(formatted, **kwargs)
            except (IndexError, KeyError, TypeError):
                print(*args, **kwargs)
        else:
            print(*args, **kwargs)
else:
    def dbg_print(*args, **kwargs):
        pass

# --- Database Configuration ---
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT", 5432),
}

# Heybo vendor DB (HH_*): availability, gogreenfood_menu image_id for signatures, etc.
HEYBO_DB_CONFIG = {
    "host": os.getenv("HH_DB_HOST"),
    "database": os.getenv("HH_DB_NAME"),
    "user": os.getenv("HH_DB_USER"),
    "password": os.getenv("HH_DB_PASSWORD"),
    "port": os.getenv("HH_DB_PORT", 5432),
}

if not DB_CONFIG.get("host"):
    print(
        "[WARN] DB_HOST is unset after .env load — psycopg2 will try localhost. "
        f"Looked for .env under {_repo_root}, {_pkg_dir}, and {Path.cwd()}"
    )
