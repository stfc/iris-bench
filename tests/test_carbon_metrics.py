import pytest
import requests
import requests_mock
from requests.exceptions import Timeout, ConnectionError as RequestsConnectionError


from iris_gpubench.carbon_metrics import (
    get_carbon_region_names,
    get_carbon_forecast,
    CARBON_INTENSITY_URL,
)
from iris_gpubench.utils.globals import DEFAULT_REGION, LOGGER, TIMEOUT_SECONDS

# --- Mock Data for Successful API Calls ---

# Successful response structure for the /regional endpoint
SUCCESS_REGIONAL_RESPONSE = {
    "data": [
        {
            "regions": [
                {
                    "shortname": "South England",
                    "intensity": {"forecast": 150, "index": "moderate"},
                },
                {
                    "shortname": "Scotland",
                    "intensity": {"forecast": 200, "index": "high"},
                },
                {
                    "shortname": "Wales",
                    "intensity": {"forecast": 50, "index": "low"},
                },
            ]
        }
    ]
}

# Expected region names for get_carbon_region_names test
EXPECTED_REGION_NAMES = ['South England', 'Scotland', 'Wales']

def test_get_carbon_region_names_success(requests_mock):
    """Test successful retrieval of region names."""
    requests_mock.get(
        CARBON_INTENSITY_URL,
        json=SUCCESS_REGIONAL_RESPONSE,
        status_code=200 # OK
    )
    result = get_carbon_region_names()
    assert result == EXPECTED_REGION_NAMES

def test_get_carbon_region_names_http_error(requests_mock):
    """Test handling of HTTP errors (e.g., 404, 500)."""
    requests_mock.get(
        CARBON_INTENSITY_URL,
        status_code=500  # Internal Server Error
    )
    result = get_carbon_region_names()
    assert result == []

def test_get_carbon_region_names_timeout(requests_mock, caplog):
    """Test handling of request Timeout."""
    requests_mock.get(
        CARBON_INTENSITY_URL,
        exc=Timeout  # Raise a Timeout exception
    )
    result = get_carbon_region_names()
    assert result == []
    assert f"Request timed out after {TIMEOUT_SECONDS} seconds." in caplog.text

def test_get_carbon_region_names_connection_error(requests_mock, caplog):
    """Test handling of ConnectionError."""
    requests_mock.get(
        CARBON_INTENSITY_URL,
        exc=RequestsConnectionError  # Raise a ConnectionError
    )
    result = get_carbon_region_names()
    assert result == []
    assert "Network error occurred" in caplog.text

def test_get_carbon_region_names_invalid_json(requests_mock, caplog):
    """Test handling of invalid JSON response."""
    requests_mock.get(
        CARBON_INTENSITY_URL,
        text="This is not JSON",
        status_code=200
    )
    result = get_carbon_region_names()
    assert result == []
    assert "Failed to decode JSON response" in caplog.text

# --- Tests for get_carbon_forecast ---

def test_get_carbon_forecast_success(requests_mock):
    """Test successful retrieval of carbon forecast for a specified region."""
    target_region = "Wales"
    expected_forecast = 50.0
    requests_mock.get(
        CARBON_INTENSITY_URL,
        json=SUCCESS_REGIONAL_RESPONSE,
        status_code=200
    )
    result = get_carbon_forecast(target_region)
    assert result == expected_forecast

def test_get_carbon_forecast_default_region(requests_mock, monkeypatch):
    """Test successful retrieval using the default region."""
    # Temporarily set DEFAULT_REGION for the test
    monkeypatch.setattr('iris_gpubench.utils.globals.DEFAULT_REGION', 'South England')
    expected_forecast = 150.0
    requests_mock.get(
        CARBON_INTENSITY_URL,
        json=SUCCESS_REGIONAL_RESPONSE,
        status_code=200
    )
    # Call without argument to use the mocked default
    result = get_carbon_forecast()
    assert result == expected_forecast

def test_get_carbon_forecast_region_not_found(requests_mock, caplog):
    """Test case where the requested region is not in the response."""
    missing_region = "London"
    requests_mock.get(
        CARBON_INTENSITY_URL,
        json=SUCCESS_REGIONAL_RESPONSE,
        status_code=200
    )
    result = get_carbon_forecast(missing_region)
    assert result is None
    assert f"Region '{missing_region}' not found in the response." in caplog.text

def test_get_carbon_forecast_value_error_non_float(requests_mock, caplog):
    """Test handling of a non-numeric 'forecast' value (raises ValueError on float() conversion)."""
    bad_forecast_data = {
        "data": [
            {
                "regions": [
                    {
                        "shortname": "Scotland",
                        "intensity": {"forecast": "not_a_number", "index": "moderate"},
                    }
                ]
            }
        ]
    }
    requests_mock.get(
        CARBON_INTENSITY_URL,
        json=bad_forecast_data,
        status_code=200
    )
    result = get_carbon_forecast("Scotland")
    assert result is None
    assert "Failed to decode JSON response" in caplog.text

def test_get_carbon_forecast_http_error(requests_mock):
    """Test handling of HTTP errors (e.g., 401 Unauthorized)."""
    requests_mock.get(
        CARBON_INTENSITY_URL,
        status_code=401
    )
    result = get_carbon_forecast("Scotland")
    assert result is None

def test_get_carbon_forecast_timeout(requests_mock):
    """Test handling of request Timeout."""
    requests_mock.get(
        CARBON_INTENSITY_URL,
        exc=Timeout
    )
    result = get_carbon_forecast("Scotland")
    assert result is None