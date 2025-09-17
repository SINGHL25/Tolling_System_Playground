
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import plotly.figure_factory as ff

st.set_page_config(page_title="Pollution Impact", page_icon="🌍", layout="wide")

st.title("🌍 Air Quality & Pollution Impact Analysis")
st.markdown("Monitor environmental impact of toll road operations and vehicle emissions")

# Generate comprehensive pollution data
@st.cache_data
def load_pollution_data():
    np.random.seed(42)
    
    # Generate 30 days of hourly data
    dates = pd.date_range(end=datetime.now(), periods=720, freq='H')
    monitoring_stations = ['Station_North', 'Station_South', 'Station_Center', 'Station_East', 'Station_West']
    
    pollution_data = []
    
    for date in dates:
        hour = date.hour
        weekday = date.weekday()
        month = date.month
        
        # Base pollution levels (seasonal variation)
        if month in [12, 1, 2]:  # Winter - higher pollution
            base_multiplier = 1.4
        elif month in [6, 7, 8]:  # Summer - moderate pollution
            base_multiplier = 1.2
        else:  # Spring/Fall - lower pollution
            base_multiplier = 1.0
        
        # Daily variation based on traffic patterns
        if weekday < 5:  # Weekdays
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                traffic_multiplier = 2.5
            elif 10 <= hour <= 16:  # Daytime
                traffic_multiplier = 1.5
            else:  # Off-peak
                traffic_multiplier = 0.6
        else:  # Weekends
            if 10 <= hour <= 18:
                traffic_multiplier = 1.8
            else:
                traffic_multiplier = 0.8
        
        # Weather impact
        weather_conditions = np.random.choice(['Clear', 'Rain', 'Fog', 'Wind'], p=[0.6, 0.2, 0.1, 0.1])
        if weather_conditions == 'Rain':
            weather_multiplier = 0.7  # Rain cleans air
        elif weather_conditions == 'Fog':
            weather_multiplier = 1.3  # Fog traps pollutants
        elif weather_conditions == 'Wind':
            weather_multiplier = 0.5  # Wind disperses pollutants
        else:
            weather_multiplier = 1.0
        
        for station in monitoring_stations:
            # Station-specific factors based on location
            station_factors = {
                'Station_North': 0.9,   # Less traffic
                'Station_South': 1.2,  # Industrial area
                'Station_Center': 1.5, # Highest traffic
                'Station_East': 1.0,   # Moderate traffic
                'Station_West': 0.8    # Residential area
            }
            
            total_multiplier = base_multiplier * traffic_multiplier * weather_multiplier * station_factors[station]
            
            # PM2.5 (Fine particulate matter) - μg/m³
            base_pm25 = 15  # WHO guideline: 5 μg/m³ annual, 15 μg/m³ daily
            pm25 = max(0, np.random.normal(base_pm25 * total_multiplier, base_pm25 * 0.3))
            
            # PM10 (Coarse particulate matter) - μg/m³
            pm10 = pm25 * np.random.uniform(1.5, 2.5)  # PM10 typically 1.5-2.5x PM2.5
            
            # NO2 (Nitrogen Dioxide) - μg/m³
            base_no2 = 25  # WHO guideline: 10 μg/m³ annual, 25 μg/m³ daily
            no2 = max(0, np.random.normal(base_no2 * total_multiplier, base_no2 * 0.4))
            
            # CO (Carbon Monoxide) - mg/m³
            base_co = 4  # WHO guideline: 10 mg/m³ for 8-hour average
            co = max(0, np.random.normal(base_co * total_multiplier * 0.8, base_co * 0.2))
            
            # O3 (Ozone) - μg/m³ (inverse relationship with NO2)
            base_o3 = 80
            o3_multiplier = max(0.3, 2 - (no2 / base_no2))  # High NO2 reduces O3
            o3 = max(0, np.random.normal(base_o3 * o3_multiplier, base_o3 * 0.3))
            
            # SO2 (Sulfur Dioxide) - μg/m³
            base_so2 = 20  # WHO guideline: 40 μg/m³ daily
            so2 = max(0, np.random.normal(base_so2 * total_multiplier * 0.6, base_so2 * 0.3))
            
            # Traffic-related pollutants
            vehicle_count = max(0, np.random.poisson(50 * traffic_multiplier))
            
            # Calculate AQI (Air Quality Index)
            aqi_pm25 = calculate_aqi_pm25(pm25)
            aqi_no2 = calculate_aqi_no2(no2)
            aqi_overall = max(aqi_pm25, aqi_no2)  # Worst pollutant determines AQI
            
            # Health impact estimation
            health_risk = "Good" if aqi_overall <= 50 else "Moderate" if aqi_overall <= 100 else "Unhealthy for Sensitive Groups" if aqi_overall <= 150 else "Unhealthy"
            
            # Environmental cost calculation (simplified)
            health_cost_per_day = calculate_health_cost(pm25, no2, vehicle_count)
            
            pollution_data.append({
                'timestamp': date,
                'station_id': station,
                'pm25_ugm3': round(pm25, 2),
                'pm10_ugm3': round(pm10, 2),
                'no2_ugm3': round(no2, 2),
                'co_mgm3': round(co, 3),
                'o3_ugm3': round(o3, 2),
                'so2_ugm3': round(so2, 2),
                'aqi': round(aqi_overall, 0),
                'health_risk': health_risk,
                'weather_condition': weather_conditions,
                'temperature_c': np.random.normal(20, 10),
                'humidity_percent': np.random.uniform(30, 90),
                'wind_speed_ms': np.random.exponential(2),
                'vehicle_count_estimated': vehicle_count,
                'health_cost_usd_daily': round(health_cost_per_day, 2),
                'visibility_km': max(1, np.random.normal(15, 5) * weather_multiplier),
                'uv_index': max(0, np.random.normal(5, 2)) if 8 <= hour <= 18 else 0
            })
    
    return pd.DataFrame(pollution_data)

def calculate_aqi_pm25(pm25):
    """Calculate AQI for PM2.5 based on US EPA standards"""
    if pm25 <= 12:
        return (50 / 12) * pm25
    elif pm25 <= 35.4:
        return 50 + ((100 - 50) / (35.4 - 12.1)) * (pm25 - 12.1)
    elif pm25 <= 55.4:
        return 100 + ((150 - 100) / (55.4 - 35.5)) * (pm25 - 35.5)
    elif pm25 <= 150.4:
        return 150 + ((200 - 150) / (150.4 - 55.5)) * (pm25 - 55.5)
    else:
        return min(300, 200 + ((300 - 200) / (250.4 - 150.5)) * (pm25 - 150.5))

def calculate_aqi_no2(no2):
    """Calculate AQI for NO2 (simplified)"""
    if no2 <= 53:
        return (50 / 53) * no2
    elif no2 <= 100:
        return 50 + ((100 - 50) / (100 - 54)) * (no2 - 54)
    elif no2 <= 360:
        return 100 + ((150 - 100) / (360 - 101)) * (no2 - 101)
    else:
        return min(200, 150 + ((200 - 150) / (649 - 361)) * (no2 - 361))

def calculate_health_cost(pm25, no2, vehicle_count):
    """Estimate daily health cost based on pollution levels"""
    # Simplified health cost calculation ($/day per 1000 people)
    pm25_cost = (pm25 / 10) * 2.5  # $2.5 per 10 μg/m³ PM2.5
    no2_cost = (no2 / 25) * 1.8   # $1.8 per 25 μg/m³ NO2
    traffic_cost = (vehicle_count / 1000) * 0.5  # $0.5 per 1000 vehicles
    
    return max(0, pm25_cost + no2_cost + traffic_cost)

# Load pollution data
pollution_df = load_pollution_data()

# Sidebar filters
st.sidebar.header("🌍 Environmental Filters")

# Date range
min_date = pollution_df['timestamp'].min().date()
max_date = pollution_df['timestamp'].max().date()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(max_date - timedelta(days=7), max_date),
    min_value=min_date,
    max_value=max_date
)

# Station selection
selected_stations = st.sidebar.multiselect(
    "Monitoring Stations",
    options=pollution_df['station_id'].unique(),
    default=pollution_df['station_id'].unique()
)

# Health risk filter
risk_levels = st.sidebar.multiselect(
    "Health Risk Levels",
    options=['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy'],
    default=['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy']
)

# Weather condition filter
weather_filter = st.sidebar.multiselect(
    "Weather Conditions",
    options=pollution_df['weather_condition'].unique(),
    default=pollution_df['weather_condition'].unique()
)

# Filter data
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = pollution_df[
        (pollution_df['timestamp'].dt.date >= start_date) &
        (pollution_df['timestamp'].dt.date <= end_date) &
        (pollution_df['station_id'].isin(selected_stations)) &
