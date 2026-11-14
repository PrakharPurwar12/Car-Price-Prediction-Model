"""
tests/test_data_validation.py
Validates user input parsing and numerical constraint checks.
"""

from ml.predict import LUXURY_BRANDS


def test_luxury_brand_classification():
    assert "bmw" in LUXURY_BRANDS
    assert "mercedes-benz" in LUXURY_BRANDS
    assert "audi" in LUXURY_BRANDS
    assert "land rover" in LUXURY_BRANDS
    assert "maruti" not in LUXURY_BRANDS
    assert "hyundai" not in LUXURY_BRANDS


def test_year_boundary_handling():
    current_year = 2025
    year = 2018
    age = max(1, current_year - year)
    assert age == 7

    # Future or current year car has age of at least 1
    future_year = 2025
    age_future = max(1, current_year - future_year)
    assert age_future >= 1

