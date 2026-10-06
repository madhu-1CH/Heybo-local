"""
Standalone SageMaker inference entrypoints for Heybo.

This module is independent of salad.py and can be used as the model server
entry script for a dedicated Heybo endpoint.
"""

import json
import logging
from typing import Any, Dict

try:
    # Package context: imported as heybo.sagemaker_inference
    from .config import DB_CONFIG, HEYBO_DB_CONFIG
except ImportError:
    # Flat source_dir context: config.py is a sibling module
    from config import DB_CONFIG, HEYBO_DB_CONFIG


def model_fn(model_dir: str) -> Dict[str, Any]:
    """
    SageMaker model loader.
    Returns runtime configuration needed by downstream code.
    """
    _ = model_dir
    return {
        "db_config": DB_CONFIG,
        "heybo_db_config": HEYBO_DB_CONFIG,
    }


def input_fn(request_body: Any, request_content_type: str) -> Dict[str, Any]:
    """
    Deserialize incoming request payload into a dict.
    Supports JSON only.
    """
    logging.basicConfig(level=logging.INFO)
    logging.info("Received request body for Heybo inference")

    content_type = (request_content_type or "").split(";")[0].strip().lower()
    if content_type and content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")

    if not request_body:
        raise ValueError("Empty request body received")

    if isinstance(request_body, str):
        try:
            payload = json.loads(request_body) if request_body.strip() else {}
            if isinstance(payload, str):
                payload = json.loads(payload)
        except json.JSONDecodeError as e:
            logging.error("JSON decode error: %s", str(e))
            raise ValueError(f"Invalid JSON: {e}") from e
    elif isinstance(request_body, dict):
        payload = request_body
    else:
        raise ValueError(f"Unsupported request body type: {type(request_body).__name__}")

    for key in ("location_id", "location_type"):
        if key not in payload:
            raise ValueError(f"Missing required key: {key}")

    if not payload.get("ShopName"):
        payload["ShopName"] = ["heybo"]
    return payload


def predict_fn(input_data: Dict[str, Any], model: Any = None) -> Dict[str, Any]:
    """
    Execute Heybo recommendation logic.
    The `model` arg is unused and exists for SageMaker compatibility.
    """
    _ = model
    try:
        # Local import avoids circular import when heyboo/__init__.py is run directly.
        try:
            from . import process_heybo_request
        except ImportError:
            try:
                from heyboo import process_heybo_request
            except ImportError:
                from __init__ import process_heybo_request

        result = process_heybo_request(input_data)
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"result": result}
        return result
    except Exception as e:
        logging.exception("ERROR in Heybo predict_fn")
        return {
            "recommended_meal": [],
            "message_to_user": f"Error processing request: {str(e)}",
        }


def output_fn(prediction: Dict[str, Any], accept: str) -> str:
    """
    Serialize inference response.
    Supports JSON only.
    """
    accept_type = (accept or "").split(";")[0].strip().lower()
    if accept_type and accept_type != "application/json":
        raise ValueError(f"Unsupported content type: {accept}")
    if isinstance(prediction, dict):
        return json.dumps(prediction, indent=4)
    return prediction

