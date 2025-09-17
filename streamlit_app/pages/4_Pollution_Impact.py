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
        (pollution_df['health_risk'].isin(risk_levels)) &
        (pollution_df['weather_condition'].isin(weather_filter))
    ]
else:
    filtered_df = pollution_df

# Current air quality status
st.subheader("🌬️ Current Air Quality Status")

# Get latest readings for each station
latest_readings = pollution_df.groupby('station_id').tail(1)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    avg_aqi = latest_readings['aqi'].mean()
    aqi_status = "🟢 Good" if avg_aqi <= 50 else "🟡 Moderate" if avg_aqi <= 100 else "🟠 Unhealthy for Sensitive" if avg_aqi <= 150 else "🔴 Unhealthy"
    st.metric("Average AQI", f"{avg_aqi:.0f}", delta=aqi_status)

with col2:
    avg_pm25 = latest_readings['pm25_ugm3'].mean()
    pm25_status = "Good" if avg_pm25 <= 12 else "Moderate" if avg_pm25 <= 35 else "Unhealthy"
    st.metric("PM2.5", f"{avg_pm25:.1f} μg/m³", delta=pm25_status)

with col3:
    avg_no2 = latest_readings['no2_ugm3'].mean()
    no2_status = "Good" if avg_no2 <= 53 else "Moderate" if avg_no2 <= 100 else "High"
    st.metric("NO2", f"{avg_no2:.1f} μg/m³", delta=no2_status)

with col4:
    total_health_cost = latest_readings['health_cost_usd_daily'].sum()
    st.metric("Daily Health Cost", f"${total_health_cost:.2f}")

with col5:
    visibility = latest_readings['visibility_km'].mean()
    vis_status = "Excellent" if visibility > 10 else "Good" if visibility > 5 else "Poor"
    st.metric("Visibility", f"{visibility:.1f} km", delta=vis_status)

# Air quality trends
st.subheader("📈 Air Quality Trends")

tab1, tab2, tab3 = st.tabs(["AQI Trends", "Pollutant Levels", "Health Impact"])

with tab1:
    # AQI trend over time
    hourly_aqi = filtered_df.groupby([
        filtered_df['timestamp'].dt.floor('H'),
        'station_id'
    ])['aqi'].mean().reset_index()
    
    fig = px.line(
        hourly_aqi,
        x='timestamp',
        y='aqi',
        color='station_id',
        title="Air Quality Index (AQI) Trends",
        labels={'aqi': 'AQI', 'timestamp': 'Time'}
    )
    
    # Add AQI threshold lines
    fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good (≤50)")
    fig.add_hline(y=100, line_dash="dash", line_color="yellow", annotation_text="Moderate (≤100)")
    fig.add_hline(y=150, line_dash="dash", line_color="orange", annotation_text="Unhealthy for Sensitive (≤150)")
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Multiple pollutants comparison
    pollutant_trends = filtered_df.groupby(filtered_df['timestamp'].dt.floor('H')).agg({
        'pm25_ugm3': 'mean',
        'pm10_ugm3': 'mean',
        'no2_ugm3': 'mean',
        'co_mgm3': 'mean',
        'o3_ugm3': 'mean',
        'so2_ugm3': 'mean'
    }).reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PM2.5 & PM10', 'NO2 & O3', 'CO', 'SO2'),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # PM2.5 and PM10
    fig.add_trace(go.Scatter(x=pollutant_trends['timestamp'], y=pollutant_trends['pm25_ugm3'],
                            name='PM2.5', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=pollutant_trends['timestamp'], y=pollutant_trends['pm10_ugm3'],
                            name='PM10', line=dict(color='orange')), row=1, col=1)
    
    # NO2 and O3
    fig.add_trace(go.Scatter(x=pollutant_trends['timestamp'], y=pollutant_trends['no2_ugm3'],
                            name='NO2', line=dict(color='blue')), row=1, col=2)
    fig.add_trace(go.Scatter(x=pollutant_trends['timestamp'], y=pollutant_trends['o3_ugm3'],
                            name='O3', line=dict(color='lightblue')), row=1, col=2)
    
    # CO
    fig.add_trace(go.Scatter(x=pollutant_trends['timestamp'], y=pollutant_trends['co_mgm3'],
                            name='CO', line=dict(color='purple')), row=2, col=1)
    
    # SO2
    fig.add_trace(go.Scatter(x=pollutant_trends['timestamp'], y=pollutant_trends['so2_ugm3'],
                            name='SO2', line=dict(color='brown')), row=2, col=2)
    
    fig.update_layout(height=600, title_text="Pollutant Concentration Trends")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Health cost trends
    daily_health_cost = filtered_df.groupby(filtered_df['timestamp'].dt.date).agg({
        'health_cost_usd_daily': 'sum',
        'aqi': 'mean'
    }).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=daily_health_cost['timestamp'], y=daily_health_cost['health_cost_usd_daily'],
                  mode='lines+markers', name='Daily Health Cost ($)', line=dict(color='red')),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=daily_health_cost['timestamp'], y=daily_health_cost['aqi'],
                  mode='lines+markers', name='Average AQI', line=dict(color='blue')),
        secondary_y=True,
    )
    
    fig.update_layout(title_text="Health Cost vs Air Quality")
    fig.update_xaxis(title_text="Date")
    fig.update_yaxis(title_text="Health Cost ($)", secondary_y=False)
    fig.update_yaxis(title_text="AQI", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

# Spatial analysis
st.subheader("🗺️ Spatial Air Quality Analysis")

col1, col2 = st.columns(2)

with col1:
    # Station comparison
    station_comparison = filtered_df.groupby('station_id').agg({
        'aqi': 'mean',
        'pm25_ugm3': 'mean',
        'no2_ugm3': 'mean',
        'health_cost_usd_daily': 'mean'
    }).reset_index()
    
    fig = px.bar(
        station_comparison,
        x='station_id',
        y='aqi',
        title="Average AQI by Monitoring Station",
        labels={'aqi': 'Average AQI', 'station_id': 'Station'},
        color='aqi',
        color_continuous_scale='RdYlGn_r'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Pollutant distribution by station
    fig = px.box(
        filtered_df,
        x='station_id',
        y='pm25_ugm3',
        title="PM2.5 Distribution by Station",
        labels={'pm25_ugm3': 'PM2.5 (μg/m³)', 'station_id': 'Station'}
    )
    st.plotly_chart(fig, use_container_width=True)

# Weather impact analysis
st.subheader("🌦️ Weather Impact on Air Quality")

col1, col2 = st.columns(2)

with col1:
    # Weather condition impact
    weather_impact = filtered_df.groupby('weather_condition').agg({
        'aqi': 'mean',
        'pm25_ugm3': 'mean',
        'visibility_km': 'mean'
    }).reset_index()
    
    fig = px.bar(
        weather_impact,
        x='weather_condition',
        y='aqi',
        title="Average AQI by Weather Condition",
        labels={'aqi': 'Average AQI', 'weather_condition': 'Weather'},
        color='aqi',
        color_continuous_scale='RdYlGn_r'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Wind speed vs pollution
    fig = px.scatter(
        filtered_df,
        x='wind_speed_ms',
        y='pm25_ugm3',
        color='weather_condition',
        size='aqi',
        title="Wind Speed vs PM2.5 Concentration",
        labels={'wind_speed_ms': 'Wind Speed (m/s)', 'pm25_ugm3': 'PM2.5 (μg/m³)'},
        hover_data=['aqi', 'visibility_km']
    )
    st.plotly_chart(fig, use_container_width=True)

# Traffic correlation analysis
st.subheader("🚗 Traffic-Pollution Correlation")

col1, col2 = st.columns(2)

with col1:
    # Vehicle count vs pollution
    fig = px.scatter(
        filtered_df,
        x='vehicle_count_estimated',
        y='no2_ugm3',
        color='station_id',
        size='aqi',
        title="Vehicle Count vs NO2 Levels",
        labels={'vehicle_count_estimated': 'Estimated Vehicle Count', 'no2_ugm3': 'NO2 (μg/m³)'},
        hover_data=['pm25_ugm3', 'aqi']
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Hourly pattern analysis
    hourly_pattern = filtered_df.groupby(filtered_df['timestamp'].dt.hour).agg({
        'vehicle_count_estimated': 'mean',
        'no2_ugm3': 'mean',
        'pm25_ugm3': 'mean'
    }).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=hourly_pattern['timestamp'], y=hourly_pattern['vehicle_count_estimated'],
                  mode='lines+markers', name='Vehicle Count', line=dict(color='blue')),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=hourly_pattern['timestamp'], y=hourly_pattern['no2_ugm3'],
                  mode='lines+markers', name='NO2 (μg/m³)', line=dict(color='red')),
        secondary_y=True,
    )
    
    fig.update_layout(title_text="Hourly Traffic vs Pollution Pattern")
    fig.update_xaxis(title_text="Hour of Day")
    fig.update_yaxis(title_text="Vehicle Count", secondary_y=False)
    fig.update_yaxis(title_text="NO2 Concentration", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

# Health risk assessment
st.subheader("🏥 Health Risk Assessment")

col1, col2, col3 = st.columns(3)

with col1:
    # Health risk distribution
    risk_distribution = filtered_df['health_risk'].value_counts()
    colors = {'Good': '#2E8B57', 'Moderate': '#FFD700', 'Unhealthy for Sensitive Groups': '#FF8C00', 'Unhealthy': '#DC143C'}
    
    fig = px.pie(
        values=risk_distribution.values,
        names=risk_distribution.index,
        title="Health Risk Distribution",
        color=risk_distribution.index,
        color_discrete_map=colors
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Population at risk estimation
    total_population = 50000  # Estimated population in toll road vicinity
    
    risk_stats = filtered_df['health_risk'].value_counts()
    total_observations = len(filtered_df)
    
    st.write("**Estimated Population at Risk:**")
    for risk_level, count in risk_stats.items():
        percentage = (count / total_observations) * 100
        estimated_people = int((percentage / 100) * total_population)
        
        if risk_level == 'Good':
            st.success(f"✅ {risk_level}: {estimated_people:,} people ({percentage:.1f}%)")
        elif risk_level == 'Moderate':
            st.warning(f"⚠️ {risk_level}: {estimated_people:,} people ({percentage:.1f}%)")
        else:
            st.error(f"🚨 {risk_level}: {estimated_people:,} people ({percentage:.1f}%)")

with col3:
    # Economic impact
    total_health_cost = filtered_df['health_cost_usd_daily'].sum()
    if len(date_range) == 2:
        days = (end_date - start_date).days + 1
        daily_avg_cost = total_health_cost / days if days > 0 else 0
        annual_estimate = daily_avg_cost * 365
    else:
        annual_estimate = total_health_cost * 365 / len(filtered_df) * 24
    
    st.metric("Total Health Cost", f"${total_health_cost:.2f}")
    st.metric("Daily Average", f"${daily_avg_cost:.2f}" if 'daily_avg_cost' in locals() else "N/A")
    st.metric("Annual Estimate", f"${annual_estimate:,.0f}")

# Environmental recommendations
st.subheader("🎯 Environmental Recommendations")

# Analyze current conditions and provide recommendations
high_aqi_stations = station_comparison[station_comparison['aqi'] > 100]
high_pm25_stations = station_comparison[station_comparison['pm25_ugm3'] > 35]
high_health_cost = station_comparison[station_comparison['health_cost_usd_daily'] > 5]

recommendations = []

if not high_aqi_stations.empty:
    recommendations.append(f"🔴 **Critical Air Quality**: Stations {', '.join(high_aqi_stations['station_id'])} have unhealthy AQI levels (>100). Consider traffic restrictions during peak hours.")

if not high_pm25_stations.empty:
    recommendations.append(f"🟡 **PM2.5 Alert**: Stations {', '.join(high_pm25_stations['station_id'])} exceed WHO guidelines (>35 μg/m³). Implement dust control measures and encourage electric vehicle adoption.")

if filtered_df['weather_condition'].value_counts().get('Fog', 0) > len(filtered_df) * 0.1:
    recommendations.append("🌫️ **Weather Impact**: High fog frequency detected. Consider installing air purification systems near toll plazas during low visibility conditions.")

if hourly_pattern['no2_ugm3'].max() > 100:
    recommendations.append("🚗 **Traffic Pollution Peak**: NO2 levels exceed healthy limits during rush hours. Implement congestion pricing or promote carpooling initiatives.")

# Positive recommendations
good_stations = station_comparison[station_comparison['aqi'] <= 50]
if not good_stations.empty:
    recommendations.append(f"✅ **Good Air Quality**: Stations {', '.join(good_stations['station_id'])} maintain excellent air quality. Continue current environmental practices.")

if len(recommendations) == 0:
    recommendations.append("✅ **Environmental Status**: Air quality levels are within acceptable ranges across all monitoring stations.")

for rec in recommendations:
    st.markdown(rec)

# Detailed pollution data table
st.subheader("📊 Pollution Monitoring Data")

# Summary statistics by station
pollution_summary = filtered_df.groupby('station_id').agg({
    'aqi': ['mean', 'max'],
    'pm25_ugm3': ['mean', 'max'],
    'no2_ugm3': ['mean', 'max'],
    'health_cost_usd_daily': ['mean', 'sum'],
    'visibility_km': 'mean'
}).round(2)

# Flatten column names
pollution_summary.columns = ['_'.join(col).strip() for col in pollution_summary.columns]
pollution_summary = pollution_summary.reset_index()

st.dataframe(pollution_summary, use_container_width=True)

# Recent readings table
st.subheader("📋 Recent Air Quality Readings")
recent_readings = filtered_df.head(50)[
    ['timestamp', 'station_id', 'aqi', 'pm25_ugm3', 'no2_ugm3', 
     'health_risk', 'weather_condition', 'visibility_km']
].copy()

# Format timestamp
recent_readings['timestamp'] = recent_readings['timestamp'].dt.strftime('%Y-%m-%d %H:%M')

st.dataframe(recent_readings, use_container_width=True)

# Export options
st.subheader("📤 Export Environmental Data")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Export Pollution Data"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"pollution_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("Generate Air Quality Report"):
        report = f"""
        # Air Quality Analysis Report
        
        **Analysis Period**: {date_range[0]} to {date_range[1]}
        **Monitoring Stations**: {len(selected_stations)}
        
        ## Key Metrics
        - Average AQI: {filtered_df['aqi'].mean():.1f}
        - Average PM2.5: {filtered_df['pm25_ugm3'].mean():.1f} μg/m³
        - Average NO2: {filtered_df['no2_ugm3'].mean():.1f} μg/m³
        - Total Health Cost: ${filtered_df['health_cost_usd_daily'].sum():.2f}
        - Average Visibility: {filtered_df['visibility_km'].mean():.1f} km
        
        ## Health Risk Assessment
        {filtered_df['health_risk'].value_counts().to_string()}
        
        ## Recommendations
        {chr(10).join(['- ' + rec.replace('🔴', '').replace('🟡', '').replace('🌫️', '').replace('🚗', '').replace('✅', '') for rec in recommendations])}
        
        ## Station Summary
        {pollution_summary.to_string(index=False)}
        """
        
        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"air_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

with col3:
    if st.button("Export Health Impact Data"):
        health_data = filtered_df[['timestamp', 'station_id', 'aqi', 'health_risk', 'health_cost_usd_daily']]
        csv = health_data.to_csv(index=False)
        st.download_button(
            label="Download Health Data",
            data=csv,
            file_name=f"health_impact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
