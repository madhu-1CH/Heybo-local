"""
Heybo package: Build Your Bowl (BYB) recommendation for Heybo shop.
Split into modules: config, db, filters, validation, diet, pricing, bowl_utils, generation.
"""
import json
import traceback
import os
import sys

# Restore package context when this file is loaded without a parent package
# (e.g., direct run or SageMaker importing "__init__" from a flat code bundle).
if __package__ is None or __package__ == "":
    package_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(package_dir)
    package_name = os.path.basename(package_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    __package__ = package_name

from .generation import generate_heybo_bowls, _is_truthy_flag
from .signatures import generate_heybo_signature_bowls
from .heybo_inference import input_fn, predict_fn, output_fn
from .user_message import build_heybo_message_to_user


def process_heybo_request(user_input):
    """Main entry point for Heybo requests (standalone or via SageMaker heybo_inference)."""
    shop_name = user_input.get("ShopName", [""])[0].lower()
    if "heybo" in shop_name:
        if _is_truthy_flag(user_input.get("Signatures")):
            return generate_heybo_signature_bowls(dict(user_input))
        return generate_heybo_bowls(user_input)
    err = "Unsupported shop name for Heybo."
    return {
        "error": err,
        "message_to_user": build_heybo_message_to_user(
            [], None, user_input=user_input, error=err
        ),
    }


__all__ = [
    "process_heybo_request",
    "generate_heybo_bowls",
    "generate_heybo_signature_bowls",
    "input_fn",
    "predict_fn",
    "output_fn",
]


if __name__ == "__main__":
    print("=== STARTING HEYBO TEST EXECUTION ===")
    test_input = {
        "location_id": 1,
        "session_id": "ae5b1865-c73e-446a-9cb8-a3d99315df3f",
        "order_time": "2026-02-19T12:50:00+08:00",
        "recommend_page_id": "",
        "location_type": 1,
        "ShopName": ["heybo"],
        "BowlType": "bowl",
        # "BowlName": ["pacha"],
        "FlavorPreferences": {
            "Sweet": "",
            "Sour": "",
            "Salty": "",
            "Bitter": "",
            "Spicy": "",
            "Umami": "",
        },
        "CuisineFilters": {},
        "NutrientFilters": [],
        "Ingredients": {"Include": [], "Exclude": [], "Extra": []},
        "Price": {},
        "AllergenFilters": [],
        "PreparationMethod": {},
        "DietFilters": {},
        "Signatures": False,
        "Balanced": False,
        "Light": False,
        "Hearty": False,
        "Sustainable": False,
        "Only": False,
        "Page": 1,
    }
    print("Starting test with input:", json.dumps(test_input, indent=4))
    try:
        output = process_heybo_request(test_input)
        print("Output:", json.dumps(output, indent=4))
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
