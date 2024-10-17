import requests

# Define the URL where your Flask API is running
url = 'http://localhost:5000/predict'  # If you deploy it, replace 'localhost' with your API's deployed URL

# Define the input data that the API expects
json_body = {
    "api_url": "https://render-ivuy.onrender.com/data",  # Example API URL to fetch data from (replace with your actual URL)
    "start_date": "2023-01-01",  # Define the start date for data filtering
    "end_date": "2023-10-01",    # Define the end date for data filtering
    "n_hours": 744,  # Forecast for the next hours
    "rates": {      # Example rates to calculate the energy bill
        "rate_per_kva": 499,
        "rate_per_kWh": 8.12,
        "wheeling_charge_rate": 0.60,
        "fac_rate": 0.20,
        "electricity_duty_rate": 0.075,
        "tax_on_sale_rate": 0.1904,
        "tod_charge_0-6": -1.50,
        "tod_charge_22-24": -1.50,
        "tod_charge_6-9 & 12-18": 0.00,
        "tod_charge_9-12": 0.80,
        "tod_charge_18-22": 1.10
    }
}

# Send a POST request to the API
response = requests.post(url, json=json_body)

# Print the response from the API (should include forecast results and total charges)
print(response.json())
