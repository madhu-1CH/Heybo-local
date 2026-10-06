"""
Heybo database: connection pools, SKU availability, load and preprocess ingredient data.
"""
import json
import threading

import boto3
import pandas as pd
import psycopg2
from psycopg2 import pool as psycopg2_pool

from .config import DB_CONFIG, HEYBO_DB_CONFIG

# Connection pool size per database (min=1, max=size under the hood)
MAIN_DB_POOL_SIZE = 5
VENDOR_DB_POOL_SIZE = 5


def _connection_is_valid(conn):
    """Test if a database connection is still alive."""
    if conn is None or getattr(conn, "closed", True):
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False


class DatabasePool:
    """Thread-safe PostgreSQL connection pool (Salad parity)."""

    def __init__(self, db_config, min_conn=1, max_conn=20):
        self.db_config = db_config
        self.pool = None
        self.lock = threading.Lock()
        self.min_conn = min_conn
        self.max_conn = max_conn
        self.active_connections = 0
        self.total_connections_created = 0
        self._initialize_pool()

    def _connect_kwargs(self):
        return {
            "host": self.db_config["host"],
            "database": self.db_config["database"],
            "user": self.db_config["user"],
            "password": self.db_config["password"],
            "port": self.db_config["port"],
        }

    def _direct_connect(self):
        self.total_connections_created += 1
        return psycopg2.connect(**self._connect_kwargs())

    def _initialize_pool(self):
        try:
            self.pool = psycopg2_pool.SimpleConnectionPool(
                self.min_conn,
                self.max_conn,
                **self._connect_kwargs(),
            )
            print(
                f"Heybo DB pool initialized for {self.db_config['database']} "
                f"(min={self.min_conn}, max={self.max_conn})"
            )
        except Exception as e:
            print(f"ERROR: Failed to initialize Heybo database pool: {e}")
            raise

    def get_connection(self):
        """Get a connection from the pool; validate and discard stale connections."""
        max_attempts = 3
        for attempt in range(max_attempts):
            with self.lock:
                if self.pool is None:
                    self._initialize_pool()
                try:
                    conn = self.pool.getconn()
                    if conn is None:
                        break
                    self.active_connections += 1
                except Exception as e:
                    print(f"ERROR: Failed to get connection from Heybo pool: {e}")
                    conn = None
                    try:
                        self.active_connections += 1
                        conn = self._direct_connect()
                    except Exception as fallback_e:
                        print(f"ERROR: Heybo fallback connection also failed: {fallback_e}")
                        raise
            if conn is None:
                break
            if _connection_is_valid(conn):
                return conn
            try:
                conn.close()
            except Exception:
                pass
            with self.lock:
                self.active_connections = max(0, self.active_connections - 1)
            print(
                f"WARNING: Discarded stale Heybo connection "
                f"(attempt {attempt + 1}/{max_attempts})"
            )

        with self.lock:
            if self.pool is None:
                self._initialize_pool()
            try:
                conn = self.pool.getconn()
                if conn is None:
                    print("WARNING: Heybo connection pool exhausted, creating temporary connection")
                    self.active_connections += 1
                    return self._direct_connect()
                self.active_connections += 1
            except Exception as e:
                print(f"ERROR: Failed to get connection from Heybo pool: {e}")
                self.active_connections += 1
                return self._direct_connect()

        if conn and _connection_is_valid(conn):
            return conn
        if conn:
            try:
                conn.close()
            except Exception:
                pass
            with self.lock:
                self.active_connections = max(0, self.active_connections - 1)
        with self.lock:
            self.active_connections += 1
        return self._direct_connect()

    def return_connection(self, conn):
        """Return a connection to the pool; discard if dead."""
        if not conn:
            return
        if not self.pool:
            with self.lock:
                try:
                    conn.close()
                except Exception:
                    pass
                self.active_connections = max(0, self.active_connections - 1)
            return
        if not _connection_is_valid(conn):
            with self.lock:
                try:
                    conn.close()
                except Exception:
                    pass
                self.active_connections = max(0, self.active_connections - 1)
            return
        with self.lock:
            if conn and self.pool:
                try:
                    if self.pool.closed:
                        conn.close()
                        self.active_connections = max(0, self.active_connections - 1)
                        return
                    self.pool.putconn(conn)
                    self.active_connections = max(0, self.active_connections - 1)
                except Exception as e:
                    print(f"WARNING: Failed to return connection to Heybo pool: {e}")
                    try:
                        conn.close()
                        self.active_connections = max(0, self.active_connections - 1)
                    except Exception:
                        pass
            elif conn:
                try:
                    conn.close()
                    self.active_connections = max(0, self.active_connections - 1)
                except Exception:
                    pass

    def close_all(self):
        with self.lock:
            if self.pool:
                try:
                    self.pool.closeall()
                    print(f"Heybo DB pool closed for {self.db_config['database']}")
                except Exception as e:
                    print(f"WARNING: Error closing Heybo connection pool: {e}")

    def get_pool_status(self):
        with self.lock:
            if self.pool:
                return {
                    "database": self.db_config["database"],
                    "min_conn": self.min_conn,
                    "max_conn": self.max_conn,
                    "closed": self.pool.closed,
                    "active_connections": self.active_connections,
                    "total_connections_created": self.total_connections_created,
                }
            return None


DB_POOL = None
HEYBO_VENDOR_DB_POOL = None


def initialize_connection_pools():
    """Initialize connection pools for main and vendor databases."""
    global DB_POOL, HEYBO_VENDOR_DB_POOL
    if DB_POOL is None:
        DB_POOL = DatabasePool(DB_CONFIG, min_conn=1, max_conn=MAIN_DB_POOL_SIZE)
    if HEYBO_VENDOR_DB_POOL is None:
        HEYBO_VENDOR_DB_POOL = DatabasePool(
            HEYBO_DB_CONFIG,
            min_conn=1,
            max_conn=VENDOR_DB_POOL_SIZE,
        )
    print("Heybo connection pool initialization complete")


def cleanup_connection_pools():
    """Close all pooled connections on shutdown."""
    global DB_POOL, HEYBO_VENDOR_DB_POOL
    if DB_POOL:
        DB_POOL.close_all()
        DB_POOL = None
    if HEYBO_VENDOR_DB_POOL:
        HEYBO_VENDOR_DB_POOL.close_all()
        HEYBO_VENDOR_DB_POOL = None


def get_connection_pool_status():
    """Return pool status for debugging."""
    status = {}
    if DB_POOL:
        status["main_db"] = DB_POOL.get_pool_status()
    if HEYBO_VENDOR_DB_POOL:
        status["heybo_vendor_db"] = HEYBO_VENDOR_DB_POOL.get_pool_status()
    return status


def _pool_for_config(db_config):
    global DB_POOL, HEYBO_VENDOR_DB_POOL
    db_name = (db_config or {}).get("database")
    if db_name == DB_CONFIG.get("database"):
        if DB_POOL is None:
            initialize_connection_pools()
        return DB_POOL
    if db_name == HEYBO_DB_CONFIG.get("database"):
        if HEYBO_VENDOR_DB_POOL is None:
            initialize_connection_pools()
        return HEYBO_VENDOR_DB_POOL
    return None


def get_db_connection(db_config):
    """Get a database connection from the appropriate pool."""
    pool = _pool_for_config(db_config)
    if pool is not None:
        return pool.get_connection()
    print(f"WARNING: Unknown database {db_config.get('database')}, using direct connection")
    return psycopg2.connect(
        host=db_config["host"],
        database=db_config["database"],
        user=db_config["user"],
        password=db_config["password"],
        port=db_config["port"],
    )


def return_db_connection(conn, db_config):
    """Return a database connection to the appropriate pool."""
    pool = _pool_for_config(db_config)
    if pool is not None:
        pool.return_connection(conn)
        return
    if conn:
        try:
            conn.close()
        except Exception:
            pass


def invoke_vendor_lambda(lambda_name, location_id, location_type):
    """
    Invokes the vendor_ec2_connect Lambda function to fetch available SKUs.
    Returns a list of SKUs if successful, otherwise None.
    """
    try:
        print(f"Attempting to invoke Lambda: {lambda_name}")
        lambda_client = boto3.client("lambda")
        payload = {
            "event_type": lambda_name,
            "location_id": location_id,
            "location_type": location_type,
        }
        print(f"Lambda payload: {payload}")
        response = lambda_client.invoke(
            FunctionName="vendor_ec2_connect",
            InvocationType="RequestResponse",
            Payload=json.dumps(payload),
        )
        print(f"Lambda response status: {response['StatusCode']}")
        if response["StatusCode"] == 200:
            payload_response = json.loads(response["Payload"].read().decode("utf-8"))
            print(f"Lambda payload response: {payload_response}")
            body = json.loads(payload_response.get("body", "{}"))
            if payload_response.get("statusCode") == 200:
                skus = body.get("data", [])
                print(f"Successfully got {len(skus)} SKUs from Lambda")
                return skus
            print(f"Error from Lambda: {body.get('error', 'Unknown error')}")
            return None
        print(f"Lambda invocation failed with status code: {response['StatusCode']}")
        return None
    except Exception as e:
        print(f"Error invoking Lambda {lambda_name}: {e}")
        print(f"Exception type: {type(e)}")
        return None


def get_heybo_available_skus(location_id, location_type):
    """
    Get available SKUs for Heybo location - Lambda first, then database fallback.
    Returns set of available SKUs.
    """
    print(f"\n=== FETCHING AVAILABLE SKUs FOR HEYBO LOCATION {location_id} ===")
    print("Database query for available SKUs")
    query = """
    SELECT i.sku
    FROM public.gogreenfood_ingredient i
    WHERE i.deleted_at IS NULL
      AND i."exclude_in_CYO" = false
      AND i.id NOT IN (
            SELECT ingredient_id
            FROM public.gglmenu_gglocationdisabledingredient
            WHERE gglocation_id = %s
              AND gglocation_type = %s
              AND (
                  expires_eod = false
                  OR (expires_eod = true AND created::date = CURRENT_DATE)
              )
      )
    """
    conn = None
    try:
        print("Connecting to database for fallback query...")
        conn = get_db_connection(HEYBO_DB_CONFIG)
        with conn.cursor() as cursor:
            print(f"Executing query with location_id={location_id}, location_type={location_type}")
            cursor.execute(query, (location_id, location_type))
            results = cursor.fetchall()
            available_skus = {sku[0] for sku in results}
            print(f"Got {len(available_skus)} SKUs from database")
            print("All SKUs:", sorted(available_skus))
            return available_skus
    except Exception as e:
        print(f"Error fetching available SKUs: {e}")
        print(f"Error type: {type(e)}")
        raise
    finally:
        if conn:
            return_db_connection(conn, HEYBO_DB_CONFIG)


def load_heybo_data_from_db(test_input):
    """Load Heybo ingredient data filtered by available SKUs."""
    print("\n=== LOADING HEYBO INGREDIENT DATASSSS ===")
    conn = None
    try:
        if "location_id" not in test_input or "location_type" not in test_input:
            raise ValueError("Input data must contain 'location_id' and 'location_type'")
        location_id = test_input["location_id"]
        location_type = test_input["location_type"]
        available_skus = get_heybo_available_skus(location_id, location_type)
        print(f"\nTOTAL AVAILABLE SKUs IN HEYBO: {len(available_skus)}")
        print("All SKUs:", sorted(available_skus))
        if not available_skus:
            raise ValueError("CRITICAL: No available ingredients after initial filtering")
        conn = get_db_connection(DB_CONFIG)
        query = "SELECT * FROM heybo.ingredients_details WHERE sku_code IN %s"
        print("\nExecuting query with SKU filter...")
        print(f"Query: {query}")
        print(f"Using database: {DB_CONFIG['database']}")
        print(f"Using host: {DB_CONFIG['host']}")
        df = pd.read_sql(query, conn, params=(tuple(available_skus),))
        retrieved_skus = set(df["sku_code"])
        invalid_skus = retrieved_skus - available_skus
        if invalid_skus:
            print(f"\nERROR: Found {len(invalid_skus)} invalid SKUs in results:")
            print(sorted(invalid_skus))
            raise ValueError("Database returned non-allowed ingredients")
        print(f"\nVERIFIED: All {len(retrieved_skus)} retrieved SKUs are allowed")
        df = process_heybo_ingredient_data(df)
        print("\n=== FINAL DATAFRAME ===")
        print(f"Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        print("All retrieved SKUs:")
        print(df[["sku_code", "category"]].to_string(index=False))
        return df
    except Exception as e:
        print(f"\nDATABASE ERROR: {e}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")
        raise
    finally:
        if conn:
            return_db_connection(conn, DB_CONFIG)


def process_heybo_ingredient_data(df):
    """Process and validate the Heybo ingredient DataFrame."""
    flavor_cols = ["sweet", "sour", "salty", "bitter", "spicy", "umami"]
    numeric_cols = [
        "calories_kCal",
        "carbs_g",
        "protein_g",
        "total_fat_g",
        "cholesterol_mg",
        "sodium_mg",
        "fiber_g",
        "sugar_g",
        "ai_price",
        "serving_amount_per_portion_in_grams",
        "co2e_values_per_serving",
        "saturated_fat_g",
        "trans_fat_g",
        "added_sugar_g",
        "calcium_mg",
        "iron_mg",
        "potassium_mg",
        "vitamin_d_mcg",
        "phosphorus_mg",
    ] + flavor_cols
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "does_not_go_well_with_ingredients" in df.columns:
        df["does_not_go_well_with_ingredients"] = df[
            "does_not_go_well_with_ingredients"
        ].astype(str)
    else:
        raise ValueError(
            "Required column 'does_not_go_well_with_ingredients' not found in database"
        )
    weight_col = "serving_amount_per_portion_in_grams"
    if weight_col in df.columns:
        df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0)
    else:
        raise ValueError(f"Required column '{weight_col}' not found in database")
    missing_flavors = [f for f in flavor_cols if f not in df.columns]
    if missing_flavors:
        raise ValueError(f"Missing flavor columns: {missing_flavors}")
    if "light_hearty" in df.columns:
        df["light_hearty"] = pd.to_numeric(df["light_hearty"], errors="coerce")
    return df


def preprocess_heybo_dataframe(df):
    """No preprocessing: compare user input directly against main table values."""
    return df


def load_heybo_signature_catalog(user_input):
    """
    Load preset signature rows from heybo.menu_details (filtered by sub_category vs BowlType).

    Signature matching uses sub_category (not category): LOWER(sub_category) equals
    BowlType lowercased with underscores replaced by spaces (e.g. bowl → "bowl").

    Uses columns on that table: sub_category, sku_code, menu_name,
    ingredients_list_with_sku, amount_g, selling_price, nutrient fields, allergens,
    diet_parameters, cuisine, preparation_method, light_hearty, flavor columns, etc.

    On failure or missing table, returns an empty DataFrame (no exception).
    """
    ui = dict(user_input) if isinstance(user_input, dict) else {}
    bowl_type = (ui.get("BowlType") or ui.get("Bowl Type") or "bowl").lower()
    db_bowl_type = bowl_type.replace("_", " ")
    empty = pd.DataFrame()
    conn = None
    try:
        conn = get_db_connection(DB_CONFIG)
        query = """
            SELECT * FROM heybo.menu_details
            WHERE LOWER(TRIM(sub_category)) = %s
              AND (delete_status IS NULL OR delete_status = 0)
        """
        df = pd.read_sql(query, conn, params=(db_bowl_type,))
        if df is None or df.empty:
            return empty
        return df
    except Exception as e:
        print(f"load_heybo_signature_catalog: {e}")
        return empty
    finally:
        if conn:
            return_db_connection(conn, DB_CONFIG)


def load_heybo_vendor_sku_to_image_id_map():
    """
    sku -> image_id from gogreenfood_menu on the Heybo vendor DB (HEYBO_DB_CONFIG / HH_*).
    Used for Heybo signature bowls on the preset meal sku_code. Returns {} if the query
    fails so generation can continue without image_id.
    """
    query = "SELECT sku, image_id FROM gogreenfood_menu"
    conn = None
    try:
        conn = get_db_connection(HEYBO_DB_CONFIG)
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
        if not rows:
            return {}
        out = {}
        for sku, image_id in rows:
            s = str(sku).strip() if sku is not None else ""
            if not s:
                continue
            out[s] = image_id
        return out
    except Exception as e:
        print(f"load_heybo_vendor_sku_to_image_id_map: {e}")
        return {}
    finally:
        if conn:
            return_db_connection(conn, HEYBO_DB_CONFIG)


def load_heybo_previous_signature_bowl_names(recommend_page_id):
    """
    Bowl names already returned for this recommendation page (pagination / no repeats).

    Salad uses saladstop.display_meal with recommend_page_id; Heybo uses the same
    shape under heybo.display_meal. Returns an empty set if recommend_page_id is
    missing, the query fails, or the table has no rows.
    """
    rid = (recommend_page_id or "").strip() if recommend_page_id else ""
    if not rid:
        return set()
    conn = None
    try:
        conn = get_db_connection(DB_CONFIG)
        query = """
            SELECT DISTINCT bowl_name
            FROM heybo.display_meal
            WHERE recommend_page_id = %s
              AND (delete_status IS NULL OR delete_status = 0)
        """
        df = pd.read_sql(query, conn, params=(rid,))
        if df is None or df.empty or "bowl_name" not in df.columns:
            return set()
        return {str(x) for x in df["bowl_name"].dropna().tolist()}
    except Exception as e:
        print(f"load_heybo_previous_signature_bowl_names: {e}")
        return set()
    finally:
        if conn:
            return_db_connection(conn, DB_CONFIG)


def load_heybo_apriori_rules():
    """
    Load Apriori rules from heybo.apriori_rules.
    Returns empty DataFrame on failure.
    """
    conn = None
    try:
        conn = get_db_connection(DB_CONFIG)
        query = (
            "SELECT antecedent, consequent, support, confidence, lift "
            "FROM heybo.apriori_rules"
        )
        df = pd.read_sql(query, conn)
        if df is None:
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"load_heybo_apriori_rules: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            return_db_connection(conn, DB_CONFIG)
