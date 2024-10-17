import base64
from flask import Flask, request, jsonify, send_file
import pandas as pd
import requests
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)

# Step 1: Load and preprocess data from the external API
def fetch_data_from_api(api_url):
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            return None
    except Exception as e:
        return None

def load_and_preprocess_data(api_url):
    data = fetch_data_from_api(api_url)
    if data is None:
        return None

    df = pd.DataFrame(data)
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Description'], format='%d-%m-%Y %H:%M')
    df = df.sort_values(by='DateTime')
    df.set_index('DateTime', inplace=True)

    df['kVAh'] = pd.to_numeric(df['kVAh'], errors='coerce')
    df['kVAh'].fillna(0, inplace=True)

    return df

# Step 2: Data filtering and preparation
def filter_data_by_date(df, start_date, end_date):
    mask = (df.index >= start_date) & (df.index <= end_date)
    return df[mask].copy()

def check_stationarity(timeseries, adf_threshold=0.05):
    result = adfuller(timeseries)
    return result[1] < adf_threshold

def prepare_hourly_data(filtered_data):
    filtered_data['kVah_diff'] = filtered_data['kVAh'].diff().abs()
    hourly_kvah = filtered_data['kVah_diff'].dropna().resample('H').sum()
    
    if not check_stationarity(hourly_kvah):
        hourly_kvah_diff = hourly_kvah.diff().dropna()
    else:
        hourly_kvah_diff = hourly_kvah
    
    return hourly_kvah_diff, hourly_kvah

# Step 3: Forecasting with SARIMA
def sarima_forecast(time_series, order, seasonal_order, n_hours):
    model = SARIMAX(time_series, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=n_hours)
    conf_int = forecast.conf_int()

    future_dates = pd.date_range(time_series.index[-1] + pd.Timedelta(hours=1), periods=n_hours, freq='H')

    forecast_df = pd.DataFrame({
        'Date_Hourly': future_dates,
        'Forecasted_kVah': forecast.predicted_mean,
        'Lower_CI_kVah': conf_int.iloc[:, 0],
        'Upper_CI_kVah': conf_int.iloc[:, 1]
    })

    return forecast_df, results

# Step 4: Energy bill calculation
def calculate_energy_bill(forecast_df, rates, total_hours):
    total_predicted_kvah = forecast_df['Forecasted_kVah'].sum()
    predicted_avg_kva = total_predicted_kvah / total_hours
    demand_kva = predicted_avg_kva * 2

    demand_charges = demand_kva * rates['rate_per_kva']
    wheeling_charges = total_predicted_kvah * rates['wheeling_charge_rate']
    energy_charges = total_predicted_kvah * rates['rate_per_kWh']

    tod_intervals = ['0-6', '22-24', '6-9 & 12-18', '9-12', '18-22']
    forecasted_totals = {interval: 0 for interval in tod_intervals}

    for index, row in forecast_df.iterrows():
        hour = row['Date_Hourly'].hour
        kVah = row['Forecasted_kVah']
        
        if 0 <= hour < 6:
            forecasted_totals['0-6'] += kVah
        elif 22 <= hour < 24:
            forecasted_totals['22-24'] += kVah
        elif 6 <= hour < 9 or 12 <= hour < 18:
            forecasted_totals['6-9 & 12-18'] += kVah
        elif 9 <= hour < 12:
            forecasted_totals['9-12'] += kVah
        elif 18 <= hour < 22:
            forecasted_totals['18-22'] += kVah

    tod_charges = sum(forecasted_totals[interval] * rates[f'tod_charge_{interval}'] for interval in tod_intervals)

    fac = total_predicted_kvah * rates['fac_rate']
    electricity_duty = (demand_charges + wheeling_charges + energy_charges + tod_charges + fac) * rates['electricity_duty_rate']
    tax_on_sale = total_predicted_kvah * rates['tax_on_sale_rate']

    total_charges = demand_charges + wheeling_charges + energy_charges + tod_charges + fac + electricity_duty + tax_on_sale
    
    # Return all the calculated charges
    return {
        'demand_charges': demand_charges,
        'wheeling_charges': wheeling_charges,
        'energy_charges': energy_charges,
        'tod_charges': tod_charges,
        'fac': fac,
        'electricity_duty': electricity_duty,
        'tax_on_sale': tax_on_sale,
        'total_charges': total_charges
    }

# Step 5: Flask route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json()

        api_url = data.get('api_url')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        n_hours = data.get('n_hours')
        rates = data.get('rates')

        # Step 1: Load and preprocess data
        df = load_and_preprocess_data(api_url)
        if df is None:
            return jsonify({"error": "Failed to fetch or process data."}), 400

        # Step 2: Filter data
        filtered_data = filter_data_by_date(df, start_date, end_date)

        # Step 3: Prepare hourly data
        hourly_kvah_diff, hourly_kvah = prepare_hourly_data(filtered_data)

        # Step 4: Forecasting
        forecast_df, _ = sarima_forecast(hourly_kvah_diff, (1, 0, 1), (1, 1, 1, 24), n_hours)

        # Step 5: Calculate energy bill
        total_hours = len(forecast_df)
        charges = calculate_energy_bill(forecast_df, rates, total_hours)

        # Return only the calculated charges
        return jsonify({
            "demand_charges": charges['demand_charges'],
            "wheeling_charges": charges['wheeling_charges'],
            "energy_charges": charges['energy_charges'],
            "tod_charges": charges['tod_charges'],
            "fac": charges['fac'],
            "electricity_duty": charges['electricity_duty'],
            "tax_on_sale": charges['tax_on_sale'],
            "total_charges": charges['total_charges']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# New endpoint for generating the forecast plot
@app.route('/forecast-plot', methods=['POST'])
def forecast_plot():
    try:
        # Get input data from request
        data = request.get_json()

        api_url = data.get('api_url')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        n_hours = data.get('n_hours')

        # Step 1: Load and preprocess data
        df = load_and_preprocess_data(api_url)
        if df is None:
            return jsonify({"error": "Failed to fetch or process data."}), 400

        # Step 2: Filter data
        filtered_data = filter_data_by_date(df, start_date, end_date)

        # Step 3: Prepare hourly data
        hourly_kvah_diff, hourly_kvah = prepare_hourly_data(filtered_data)

        # Step 4: Forecasting
        forecast_df, _ = sarima_forecast(hourly_kvah_diff, (1, 0, 1), (1, 1, 1, 24), n_hours)

        # Generate the forecast plot
        plt.figure(figsize=(10, 6))
        plt.plot(hourly_kvah.index, hourly_kvah, label='Historical kVah', color='blue')
        plt.plot(forecast_df['Date_Hourly'], forecast_df['Forecasted_kVah'], label='Forecasted kVah', color='orange')
        plt.fill_between(forecast_df['Date_Hourly'], 
                         forecast_df['Lower_CI_kVah'], 
                         forecast_df['Upper_CI_kVah'], 
                         color='orange', alpha=0.2)
        plt.title('kVah Forecast')
        plt.xlabel('Date Hourly')
        plt.ylabel('kVah')
        plt.legend()
        plt.grid()
        
        # Save the plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()  # Close the figure to free up memory

        # Send the plot as a response
        return send_file(buf, mimetype='image/png')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
