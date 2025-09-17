
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

st.set_page_config(page_title="System Health", page_icon="🔧", layout="wide")

st.title("🔧 System Health & Device Monitoring")
st.markdown("Monitor roadside equipment, system uptime, performance metrics, and maintenance alerts")

# Generate comprehensive system health data
@st.cache_data
def load_system_health_data():
    np.random.seed(42)
    
    # Generate 30 days of data with 15-minute intervals
    dates = pd.date_range(end=datetime.now(), periods=2880, freq='15T')
    
    # System components
    systems = {
        'toll_plazas': ['Plaza_A', 'Plaza_B', 'Plaza_C', 'Plaza_D'],
        'cameras': ['Cam_Entry_1', 'Cam_Entry_2', 'Cam_Exit_1', 'Cam_Exit_2', 'Cam_Overview_1'],
        'sensors': ['Speed_Sensor_1', 'Speed_Sensor_2', 'Weight_Sensor_1', 'Weight_Sensor_2'],
        'payment_terminals': ['Terminal_1', 'Terminal_2', 'Terminal_3', 'Terminal_4'],
        'communication': ['Network_Switch_1', 'Network_Switch_2', 'Fiber_Link_1', 'Wifi_AP_1'],
        'servers': ['Database_Server', 'Application_Server', 'Backup_Server', 'Web_Server'],
        'displays': ['LED_Display_1', 'LED_Display_2', 'Variable_Sign_1', 'Variable_Sign_2']
    }
    
    health_data = []
    
    for date in dates:
        hour = date.hour
        
        for category, devices in systems.items():
            for device in devices:
                # Base reliability varies by system type
                base_reliability = {
                    'toll_plazas': 0.98,
                    'cameras': 0.95,
                    'sensors': 0.97,
                    'payment_terminals': 0.93,
                    'communication': 0.99,
                    'servers': 0.995,
                    'displays': 0.96
                }
                
                # Environmental factors affecting reliability
                weather_impact = np.random.choice(['None', 'Rain', 'Wind', 'Snow'], p=[0.8, 0.15, 0.03, 0.02])
                weather_multiplier = {
                    'None': 1.0,
                    'Rain': 0.95,
                    'Wind': 0.98,
                    'Snow': 0.90
                }
                
                # Maintenance windows (planned downtime)
                is_maintenance = (date.hour == 3 and date.minute == 0 and 
                                np.random.random() < 0.01)  # 1% chance of maintenance at 3 AM
                
                if is_maintenance:
                    uptime = 0
                    status = 'Maintenance'
                    response_time = 0
                else:
                    # Calculate uptime based on reliability
                    reliability = base_reliability[category] * weather_multiplier[weather_impact]
                    uptime = 100 if np.random.random() < reliability else np.random.uniform(0, 95)
                    
                    if uptime >= 99:
                        status = 'Online'
                    elif uptime >= 95:
                        status = 'Degraded'
                    else:
                        status = 'Offline'
                
                # Performance metrics
                if status == 'Online':
                    response_time = np.random.normal(50, 15)  # milliseconds
                    cpu_usage = np.random.normal(25, 10)
                    memory_usage = np.random.normal(40, 15)
                    temperature = np.random.normal(35, 8)
                elif status == 'Degraded':
                    response_time = np.random.normal(150, 50)
                    cpu_usage = np.random.normal(65, 15)
                    memory_usage = np.random.normal(75, 10)
                    temperature = np.random.normal(45, 10)
                elif status == 'Maintenance':
                    response_time = 0
                    cpu_usage = 0
                    memory_usage = 0
                    temperature = 25
                else:  # Offline
                    response_time = 5000  # Timeout
                    cpu_usage = 0
                    memory_usage = 0
                    temperature = np.random.normal(25, 5)
                
                # Ensure realistic ranges
                response_time = max(0, response_time)
                cpu_usage = max(0, min(100, cpu_usage))
                memory_usage = max(0, min(100, memory_usage))
                temperature = max(10, temperature)
                
                # Error counts
                if status == 'Online':
                    error_count = np.random.poisson(0.1)
                    warning_count = np.random.poisson(0.5)
                elif status == 'Degraded':
                    error_count = np.random.poisson(2)
                    warning_count = np.random.poisson(5)
                else:
                    error_count = np.random.poisson(10) if status == 'Offline' else 0
                    warning_count = np.random.poisson(3) if status == 'Offline' else 0
                
                # Network metrics for communication devices
                if category == 'communication':
                    bandwidth_usage = np.random.uniform(10, 90) if status == 'Online' else 0
                    packet_loss = np.random.uniform(0, 2) if status == 'Online' else 100
                else:
                    bandwidth_usage = None
                    packet_loss = None
                
                # Generate alerts
                alerts = []
                if temperature > 50:
                    alerts.append('High Temperature')
                if cpu_usage > 80:
                    alerts.append('High CPU Usage')
                if memory_usage > 85:
                    alerts.append('High Memory Usage')
                if response_time > 1000:
                    alerts.append('Slow Response Time')
                if packet_loss and packet_loss > 5:
                    alerts.append('High Packet Loss')
                
                # Last maintenance date
                days_since_maintenance = np.random.randint(1, 90)
                last_maintenance = date - timedelta(days=days_since_maintenance)
                
                health_data.append({
                    'timestamp': date,
                    'device_id': device,
                    'category': category,
                    'status': status,
                    'uptime_percent': round(uptime, 2),
                    'response_time_ms': round(response_time, 1),
                    'cpu_usage_percent': round(cpu_usage, 1),
                    'memory_usage_percent': round(memory_usage, 1),
                    'temperature_celsius': round(temperature, 1),
                    'error_count': error_count,
                    'warning_count': warning_count,
                    'bandwidth_usage_percent': round(bandwidth_usage, 1) if bandwidth_usage else None,
                    'packet_loss_percent': round(packet_loss, 1) if packet_loss else None,
                    'weather_condition': weather_impact,
                    'alerts': '|'.join(alerts) if alerts else 'None',
                    'last_maintenance_date': last_maintenance,
                    'firmware_version': f"v{np.random.randint(1, 5)}.{np.random.randint(0, 10)}.{np.random.randint(0, 20)}",
                    'power_status': 'Normal' if status != 'Offline' else 'Power Loss'
                })
    
    return pd.DataFrame(health_data)

# Load system health data
health_df = load_system_health_data()

# Sidebar filters
st.sidebar.header("🔧 System Filters")

# Date range
min_date = health_df['timestamp'].min().date()
max_date = health_df['timestamp'].max().date()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(max_date - timedelta(days=3), max_date),
    min_value=min_date,
    max_value=max_date
)

# System category filter
selected_categories = st.sidebar.multiselect(
    "System Categories",
    options=health_df['category'].unique(),
    default=health_df['category'].unique()
)

# Device status filter
status_filter = st.sidebar.multiselect(
    "Device Status",
    options=['Online', 'Degraded', 'Offline', 'Maintenance'],
    default=['Online', 'Degraded', 'Offline', 'Maintenance']
)

# Alert level filter
show_alerts_only = st.sidebar.checkbox("Show Only Devices with Alerts", False)

# Filter data
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = health_df[
        (health_df['timestamp'].dt.date >= start_date) &
        (health_df['timestamp'].dt.date <= end_date) &
        (health_df['category'].isin(selected_categories)) &
        (health_df['status'].isin(status_filter))
    ]
else:
    filtered_df = health_df

if show_alerts_only:
    filtered_df = filtered_df[filtered_df['alerts'] != 'None']

# Current system status overview
st.subheader("🖥️ System Status Overview")

# Get latest status for each device
latest_status = health_df.groupby('device_id').tail(1)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_devices = len(latest_status)
    online_devices = len(latest_status[latest_status['status'] == 'Online'])
    online_percentage = (online_devices / total_devices * 100) if total_devices > 0 else 0
    st.metric("Total Devices", total_devices, f"{online_percentage:.1f}% Online")

with col2:
    avg_uptime = latest_status['uptime_percent'].mean()
    uptime_status = "Excellent" if avg_uptime >= 99 else "Good" if avg_uptime >= 95 else "Poor"
    st.metric("Average Uptime", f"{avg_uptime:.1f}%", uptime_status)

with col3:
    active_alerts = len(latest_status[latest_status['alerts'] != 'None'])
    alert_status = "🟢 Normal" if active_alerts == 0 else f"🟡 {active_alerts} Alerts"
    st.metric("Active Alerts", active_alerts, alert_status)

with col4:
    avg_response_time = latest_status[latest_status['status'] == 'Online']['response_time_ms'].mean()
    response_status = "Fast" if avg_response_time < 100 else "Slow"
    st.metric("Avg Response Time", f"{avg_response_time:.0f} ms", response_status)

with col5:
    maintenance_due = len(latest_status[
        (datetime.now().date() - latest_status['last_maintenance_date'].dt.date).dt.days > 60
    ])
    st.metric("Maintenance Due", maintenance_due, "🔧 Devices")

# System status heatmap
st.subheader("🗺️ System Status Heatmap")

# Create uptime heatmap by category and device
if not filtered_df.empty:
    latest_filtered = filtered_df.groupby('device_id').tail(1)
    heatmap_data = latest_filtered.pivot_table(
        values='uptime_percent',
        index='category',
        columns='device_id',
        fill_value=0
    )
    
    fig = px.imshow(
        heatmap_data,
        title="Device Uptime Heatmap (%)",
        color_continuous_scale='RdYlGn',
        aspect='auto',
        labels=dict(color="Uptime %")
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# System performance trends
st.subheader("📈 Performance Trends")

tab1, tab2, tab3 = st.tabs(["Uptime Trends", "Response Times", "Resource Usage"])

with tab1:
    # Uptime trends by category
    hourly_uptime = filtered_df.groupby([
        filtered_df['timestamp'].dt.floor('H'),
        'category'
    ])['uptime_percent'].mean().reset_index()
    
    fig = px.line(
        hourly_uptime,
        x='timestamp',
        y='uptime_percent',
        color='category',
        title="System Uptime Trends by Category",
        labels={'uptime_percent': 'Uptime (%)', 'timestamp': 'Time'}
    )
    fig.add_hline(y=99, line_dash="dash", line_color="green", annotation_text="SLA Target (99%)")
    fig.add_hline(y=95, line_dash="dash", line_color="orange", annotation_text="Warning Threshold (95%)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Response time analysis
    online_devices = filtered_df[filtered_df['status'] == 'Online']
    if not online_devices.empty:
        hourly_response = online_devices.groupby([
            online_devices['timestamp'].dt.floor('H'),
            'category'
        ])['response_time_ms'].mean().reset_index()
        
        fig = px.line(
            hourly_response,
            x='timestamp',
            y='response_time_ms',
            color='category',
            title="Average Response Time by Category",
            labels={'response_time_ms': 'Response Time (ms)', 'timestamp': 'Time'}
        )
        fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Target (100ms)")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Resource usage
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('CPU Usage', 'Memory Usage', 'Temperature', 'Error Count'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    hourly_resources = filtered_df.groupby(filtered_df['timestamp'].dt.floor('H')).agg({
        'cpu_usage_percent': 'mean',
        'memory_usage_percent': 'mean',
        'temperature_celsius': 'mean',
        'error_count': 'sum'
    }).reset_index()
    
    fig.add_trace(
        go.Scatter(x=hourly_resources['timestamp'], y=hourly_resources['cpu_usage_percent'],
                  mode='lines', name='CPU %', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=hourly_resources['timestamp'], y=hourly_resources['memory_usage_percent'],
                  mode='lines', name='Memory %', line=dict(color='green')),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=hourly_resources['timestamp'], y=hourly_resources['temperature_celsius'],
                  mode='lines', name='Temperature °C', line=dict(color='red')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=hourly_resources['timestamp'], y=hourly_resources['error_count'],
                  mode='lines', name='Errors', line=dict(color='orange')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title_text="System Resource Usage Trends", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Device category analysis
st.subheader("🏗️ System Category Analysis")

col1, col2 = st.columns(2)

with col1:
    # Status distribution by category
    category_status = latest_status.groupby(['category', 'status']).size().unstack(fill_value=0)
    
    fig = px.bar(
        category_status.reset_index(),
        x='category',
        y=['Online', 'Degraded', 'Offline', 'Maintenance'],
        title="Device Status by Category",
        labels={'value': 'Number of Devices', 'category': 'Category'},
        color_discrete_map={
            'Online': '#2E8B57',
            'Degraded': '#FFD700', 
            'Offline': '#DC143C',
            'Maintenance': '#FF8C00'
        }
    )
    fig.update_xaxis(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Average performance by category
    category_performance = latest_status.groupby('category').agg({
        'uptime_percent': 'mean',
        'response_time_ms': 'mean',
        'cpu_usage_percent': 'mean'
    }).reset_index()
    
    fig = px.scatter(
        category_performance,
        x='uptime_percent',
        y='response_time_ms',
        size='cpu_usage_percent',
        color='category',
        title="Performance Overview by Category",
        labels={
            'uptime_percent': 'Uptime (%)',
            'response_time_ms': 'Response Time (ms)',
            'cpu_usage_percent': 'CPU Usage (%)'
        },
        hover_data=['cpu_usage_percent']
    )
    st.plotly_chart(fig, use_container_width=True)

# Active alerts and issues
st.subheader("🚨 Active Alerts & Issues")

# Current alerts
current_alerts = latest_status[latest_status['alerts'] != 'None']

if not current_alerts.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        # Alert frequency
        all_alerts = []
        for alerts_str in current_alerts['alerts']:
            all_alerts.extend(alerts_str.split('|'))
        
        alert_counts = pd.Series(all_alerts).value_counts()
        
        fig = px.bar(
            x=alert_counts.values,
            y=alert_counts.index,
            orientation='h',
            title="Most Common Alert Types",
            labels={'x': 'Number of Devices', 'y': 'Alert Type'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Devices with critical issues
        critical_devices = current_alerts[
            (current_alerts['temperature_celsius'] > 50) |
            (current_alerts['cpu_usage_percent'] > 80) |
            (current_alerts['uptime_percent'] < 95)
        ]
        
        st.write("**🔥 Critical Devices Requiring Immediate Attention:**")
        for _, device in critical_devices.iterrows():
            alerts = device['alerts'].replace('|', ', ')
            status_color = {
                'Online': '🟢',
                'Degraded': '🟡',
                'Offline': '🔴',
                'Maintenance': '🟠'
            }
            
            st.warning(f"""
            **{status_color.get(device['status'], '⚪')} {device['device_id']}** ({device['category']})
            - Status: {device['status']} ({device['uptime_percent']:.1f}% uptime)
            - Alerts: {alerts}
            - Temperature: {device['temperature_celsius']:.1f}°C
            - CPU: {device['cpu_usage_percent']:.1f}%
            """)
else:
    st.success("✅ **All Systems Normal** - No active alerts detected!")

# Maintenance scheduling
st.subheader("🔧 Maintenance Management")

col1, col2, col3 = st.columns(3)

with col1:
    # Devices due for maintenance
    days_since_maintenance = (datetime.now().date() - latest_status['last_maintenance_date'].dt.date).dt.days
    maintenance_due = latest_status[days_since_maintenance > 60]
    maintenance_soon = latest_status[(days_since_maintenance > 45) & (days_since_maintenance <= 60)]
    
    st.metric("Maintenance Overdue", len(maintenance_due), "🔴 Critical")
    st.metric("Maintenance Due Soon", len(maintenance_soon), "🟡 Warning")
    
    if not maintenance_due.empty:
        st.write("**Overdue Devices:**")
        for _, device in maintenance_due.head(5).iterrows():
            days_overdue = (datetime.now().date() - device['last_maintenance_date'].date()).days
            st.text(f"• {device['device_id']}: {days_overdue} days overdue")

with col2:
    # Maintenance history
    maintenance_by_category = latest_status.groupby('category').apply(
        lambda x: (datetime.now().date() - x['last_maintenance_date'].dt.date).dt.days.mean()
    ).reset_index()
    maintenance_by_category.columns = ['category', 'avg_days_since_maintenance']
    
    fig = px.bar(
        maintenance_by_category,
        x='category',
        y='avg_days_since_maintenance',
        title="Average Days Since Last Maintenance",
        labels={'avg_days_since_maintenance': 'Days Since Maintenance', 'category': 'Category'}
    )
    fig.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="Maintenance Due (60 days)")
    fig.update_xaxis(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

with col3:
    # Firmware versions
    firmware_dist = latest_status['firmware_version'].value_counts().head(10)
    
    fig = px.pie(
        values=firmware_dist.values,
        names=firmware_dist.index,
        title="Firmware Version Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# Network and connectivity analysis
st.subheader("🌐 Network & Connectivity")

network_devices = filtered_df[filtered_df['category'] == 'communication']

if not network_devices.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        # Bandwidth usage trends
        network_hourly = network_devices.groupby(network_devices['timestamp'].dt.floor('H')).agg({
            'bandwidth_usage_percent': 'mean',
            'packet_loss_percent': 'mean'
        }).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=network_hourly['timestamp'], y=network_hourly['bandwidth_usage_percent'],
                      mode='lines+markers', name='Bandwidth Usage %', line=dict(color='blue')),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=network_hourly['timestamp'], y=network_hourly['packet_loss_percent'],
                      mode='lines+markers', name='Packet Loss %', line=dict(color='red')),
            secondary_y=True,
        )
        
        fig.update_layout(title_text="Network Performance")
        fig.update_xaxis(title_text="Time")
        fig.update_yaxis(title_text="Bandwidth Usage (%)", secondary_y=False)
        fig.update_yaxis(title_text="Packet Loss (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Network device status
        latest_network = network_devices.groupby('device_id').tail(1)
        
        fig = px.scatter(
            latest_network,
            x='bandwidth_usage_percent',
            y='packet_loss_percent',
            size='response_time_ms',
            color='status',
            hover_data=['device_id'],
            title="Network Device Performance",
            labels={
                'bandwidth_usage_percent': 'Bandwidth Usage (%)',
                'packet_loss_percent': 'Packet Loss (%)'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

# Performance benchmarking
st.subheader("📊 Performance Benchmarks")

# Calculate SLA compliance
sla_compliance = latest_status.groupby('category').agg({
    'uptime_percent': lambda x: (x >= 99).mean() * 100,
    'response_time_ms': lambda x: (x <= 100).mean() * 100
}).reset_index()
sla_compliance.columns = ['category', 'uptime_sla_compliance', 'response_sla_compliance']

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        sla_compliance,
        x='category',
        y='uptime_sla_compliance',
        title="Uptime SLA Compliance (99% Target)",
        labels={'uptime_sla_compliance': 'Compliance (%)', 'category': 'Category'}
    )
    fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Minimum Acceptable (95%)")
    fig.update_xaxis(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(
        sla_compliance,
        x='category',
        y='response_sla_compliance',
        title="Response Time SLA Compliance (100ms Target)",
        labels={'response_sla_compliance': 'Compliance (%)', 'category': 'Category'}
    )
    fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Minimum Acceptable (95%)")
    fig.update_xaxis(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# System health recommendations
st.subheader("💡 System Health Recommendations")

recommendations = []

# Check for critical issues
offline_devices = len(latest_status[latest_status['status'] == 'Offline'])
if offline_devices > 0:
    recommendations.append(f"🔴 **Critical**: {offline_devices} devices are offline. Immediate investigation required.")

degraded_devices = len(latest_status[latest_status['status'] == 'Degraded'])
if degraded_devices > 0:
    recommendations.append(f"🟡 **Warning**: {degraded_devices} devices are in degraded state. Monitor closely and schedule maintenance.")

# Check maintenance
if len(maintenance_due) > 0:
    recommendations.append(f"🔧 **Maintenance**: {len(maintenance_due)} devices are overdue for maintenance. Schedule immediate maintenance to prevent failures.")

# Check performance
high_temp_devices = len(latest_status[latest_status['temperature_celsius'] > 50])
if high_temp_devices > 0:
    recommendations.append(f"🌡️ **Temperature**: {high_temp_devices} devices running hot (>50°C). Check cooling systems and ventilation.")

high_cpu_devices = len(latest_status[latest_status['cpu_usage_percent'] > 80])
if high_cpu_devices > 0:
    recommendations.append(f"💻 **Performance**: {high_cpu_devices} devices have high CPU usage (>80%). Consider load balancing or hardware upgrades.")

# Positive feedback
if len(recommendations) == 0:
    recommendations.append("✅ **All Systems Healthy**: All devices are operating within normal parameters. Continue monitoring.")

for rec in recommendations:
    st.markdown(rec)

# Detailed device status table
st.subheader("📋 Device Status Details")

# Format the latest status for display
display_df = latest_status[[
    'device_id', 'category', 'status', 'uptime_percent', 'response_time_ms',
    'cpu_usage_percent', 'memory_usage_percent', 'temperature_celsius', 'alerts'
]].copy()

# Add status indicators
status_indicators = {
    'Online': '🟢',
    'Degraded': '🟡',
    'Offline': '🔴',
    'Maintenance': '🟠'
}
display_df['status_icon'] = display_df['status'].map(status_indicators)

st.dataframe(
    display_df[['status_icon', 'device_id', 'category', 'uptime_percent', 
               'response_time_ms', 'cpu_usage_percent', 'temperature_celsius', 'alerts']],
    use_container_width=True,
    column_config={
        'status_icon': st.column_config.TextColumn('Status'),
        'uptime_percent': st.column_config.ProgressColumn('Uptime %', min_value=0, max_value=100),
        'cpu_usage_percent': st.column_config.ProgressColumn('CPU %', min_value=0, max_value=100),
        'temperature_celsius': st.column_config.NumberColumn('Temp °C', format="%.1f")
    }
)

# Export options
st.subheader("📤 Export System Data")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Export Health Data"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"system_health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("Generate Health Report"):
        report = f"""
        # System Health Report
        
        **Analysis Period**: {date_range[0]} to {date_range[1]}
        **Total Devices**: {len(latest_status)}
        **Average Uptime**: {latest_status['uptime_percent'].mean():.2f}%
        **Active Alerts**: {len(current_alerts)}
        **Devices Offline**: {offline_devices}
        **Maintenance Overdue**: {len(maintenance_due)}
        
        ## System Categories Analysis
        {category_performance.to_string(index=False)}
        
        ## SLA Compliance
        {sla_compliance.to_string(index=False)}
        
        ## Recommendations
        {chr(10).join(['- ' + rec.replace('🔴', '').replace('🟡', '').replace('🔧', '').replace('🌡️', '').replace('💻', '').replace('✅', '') for rec in recommendations])}
        
        ## Critical Devices
        {critical_devices[['device_id', 'category', 'status', 'uptime_percent', 'alerts']].to_string(index=False) if not critical_devices.empty else 'None'}
        """
        
        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"system_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

with col3:
    if st.button("Export Alert Summary"):
        if not current_alerts.empty:
            alert_summary = current_alerts[['device_id', 'category', 'status', 'alerts', 'uptime_percent']]
            csv = alert_summary.to_csv(index=False)
            st.download_button(
                label="Download Alerts",
                data=csv,
                file_name=f"system_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No active alerts to export")
