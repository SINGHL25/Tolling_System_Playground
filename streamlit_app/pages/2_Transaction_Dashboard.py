import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

st.set_page_config(page_title="Transaction Dashboard", page_icon="💰", layout="wide")

st.title("💰 Transaction & Revenue Analytics")
st.markdown("Comprehensive analysis of toll transactions, revenue streams, and payment patterns")

# Generate comprehensive transaction data
@st.cache_data
def load_transaction_data():
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=720, freq='H')
    lanes = ['Lane_1', 'Lane_2', 'Lane_3', 'Lane_4', 'Lane_5']
    vehicle_types = ['Car', 'Truck', 'Bus', 'Motorcycle', 'Trailer']
    payment_methods = ['ETC', 'Cash', 'Credit_Card', 'Mobile_Pay']
    
    # Toll rates by vehicle type and time of day
    base_toll_rates = {
        'Car': 3.50,
        'Truck': 12.00,
        'Bus': 8.50,
        'Motorcycle': 2.00,
        'Trailer': 15.00
    }
    
    transactions = []
    
    for date in dates:
        hour = date.hour
        weekday = date.weekday()
        
        # Peak pricing multipliers
        peak_multiplier = 1.0
        if weekday < 5:  # Weekdays
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                peak_multiplier = 1.5  # 50% surcharge
                volume_multiplier = 3.0
            elif 10 <= hour <= 16:
                volume_multiplier = 1.5
            else:
                volume_multiplier = 0.4
        else:  # Weekends
            if 10 <= hour <= 18:
                volume_multiplier = 1.8
            else:
                volume_multiplier = 0.6
        
        # Generate transactions for each lane
        for lane in lanes:
            lane_factor = {
                'Lane_1': 1.2,
                'Lane_2': 1.0,
                'Lane_3': 1.0,
                'Lane_4': 0.8,
                'Lane_5': 0.6
            }
            
            base_count = int(40 * volume_multiplier * lane_factor[lane])
            transaction_count = max(0, np.random.poisson(base_count))
            
            for i in range(transaction_count):
                # Vehicle type distribution varies by lane
                if lane == 'Lane_4':  # Commercial lane
                    type_probs = [0.2, 0.6, 0.1, 0.05, 0.05]
                else:
                    type_probs = [0.75, 0.15, 0.05, 0.04, 0.01]
                
                v_type = np.random.choice(vehicle_types, p=type_probs)
                
                # Base toll amount
                base_toll = base_toll_rates[v_type]
                toll_amount = base_toll * peak_multiplier
                
                # Add discount/premium variations
                discount_factor = 1.0
                discount_type = "None"
                
                # Frequent user discounts (30% chance)
                if np.random.random() < 0.3:
                    discount_factor = 0.9
                    discount_type = "Frequent_User"
                
                # Commercial discounts for trucks (20% chance)
                if v_type in ['Truck', 'Trailer'] and np.random.random() < 0.2:
                    discount_factor = 0.85
                    discount_type = "Commercial"
                
                # Senior/disabled discounts (5% chance)
                if np.random.random() < 0.05:
                    discount_factor = 0.5
                    discount_type = "Senior_Disabled"
                
                final_toll = toll_amount * discount_factor
                
                # Payment method probabilities
                payment_probs = [0.65, 0.20, 0.10, 0.05]  # ETC, Cash, Card, Mobile
                payment_method = np.random.choice(payment_methods, p=payment_probs)
                
                # Transaction success rate varies by payment method
                success_rates = {
                    'ETC': 0.98,
                    'Cash': 0.95,
                    'Credit_Card': 0.97,
                    'Mobile_Pay': 0.96
                }
                
                transaction_success = np.random.random() < success_rates[payment_method]
                
                # Processing time varies by payment method
                processing_times = {
                    'ETC': np.random.normal(2.0, 0.5),
                    'Cash': np.random.normal(15.0, 5.0),
                    'Credit_Card': np.random.normal(8.0, 2.0),
                    'Mobile_Pay': np.random.normal(5.0, 1.5)
                }
                
                processing_time = max(0.5, processing_times[payment_method])
                
                # Generate transaction ID
                transaction_id = f"TXN{date.strftime('%Y%m%d')}{np.random.randint(10000, 99999)}"
                
                transaction_time = date + timedelta(minutes=np.random.randint(0, 60))
                
                transactions.append({
                    'transaction_id': transaction_id,
                    'timestamp': transaction_time,
                    'lane_id': lane,
                    'vehicle_type': v_type,
                    'base_toll': base_toll,
                    'peak_multiplier': peak_multiplier,
                    'discount_factor': discount_factor,
                    'discount_type': discount_type,
                    'final_toll': round(final_toll, 2),
                    'payment_method': payment_method,
                    'transaction_success': transaction_success,
                    'processing_time_sec': round(processing_time, 1),
                    'operator_id': f"OP{np.random.randint(1, 10):02d}",
                    'receipt_number': f"R{np.random.randint(100000, 999999)}",
                    'refund_issued': np.random.random() < 0.001,  # 0.1% refund rate
                    'dispute_filed': np.random.random() < 0.005   # 0.5% dispute rate
                })
    
    return pd.DataFrame(transactions)

# Load transaction data
transactions_df = load_transaction_data()

# Sidebar filters
st.sidebar.header("🔍 Transaction Filters")

# Date range
min_date = transactions_df['timestamp'].min().date()
max_date = transactions_df['timestamp'].max().date()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(max_date - timedelta(days=7), max_date),
    min_value=min_date,
    max_value=max_date
)

# Lane selection
selected_lanes = st.sidebar.multiselect(
    "Select Lanes",
    options=transactions_df['lane_id'].unique(),
    default=transactions_df['lane_id'].unique()
)

# Payment method filter
selected_payments = st.sidebar.multiselect(
    "Payment Methods",
    options=transactions_df['payment_method'].unique(),
    default=transactions_df['payment_method'].unique()
)

# Transaction status
transaction_status = st.sidebar.selectbox(
    "Transaction Status",
    options=['All', 'Successful Only', 'Failed Only'],
    index=0
)

# Filter data
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = transactions_df[
        (transactions_df['timestamp'].dt.date >= start_date) &
        (transactions_df['timestamp'].dt.date <= end_date) &
        (transactions_df['lane_id'].isin(selected_lanes)) &
        (transactions_df['payment_method'].isin(selected_payments))
    ]
else:
    filtered_df = transactions_df

# Apply transaction status filter
if transaction_status == 'Successful Only':
    filtered_df = filtered_df[filtered_df['transaction_success'] == True]
elif transaction_status == 'Failed Only':
    filtered_df = filtered_df[filtered_df['transaction_success'] == False]

# Key Revenue Metrics
st.subheader("💰 Revenue Overview")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_revenue = filtered_df[filtered_df['transaction_success']]['final_toll'].sum()
    st.metric("Total Revenue", f"${total_revenue:,.2f}")

with col2:
    total_transactions = len(filtered_df)
    st.metric("Total Transactions", f"{total_transactions:,}")

with col3:
    avg_transaction = filtered_df[filtered_df['transaction_success']]['final_toll'].mean()
    st.metric("Avg Transaction", f"${avg_transaction:.2f}")

with col4:
    success_rate = (filtered_df['transaction_success'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    st.metric("Success Rate", f"{success_rate:.1f}%")

with col5:
    failed_revenue = filtered_df[~filtered_df['transaction_success']]['final_toll'].sum()
    st.metric("Lost Revenue", f"${failed_revenue:.2f}")

# Revenue trends
st.subheader("📈 Revenue Trends")

# Daily revenue trend
daily_revenue = filtered_df[filtered_df['transaction_success']].groupby(
    filtered_df['timestamp'].dt.date
)['final_toll'].sum().reset_index()

daily_transactions = filtered_df.groupby(
    filtered_df['timestamp'].dt.date
).size().reset_index(name='transaction_count')

# Combine revenue and transaction count
daily_stats = daily_revenue.merge(daily_transactions, on='timestamp')

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(x=daily_stats['timestamp'], y=daily_stats['final_toll'],
              mode='lines+markers', name='Revenue ($)', line=dict(color='green')),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=daily_stats['timestamp'], y=daily_stats['transaction_count'],
              mode='lines+markers', name='Transaction Count', line=dict(color='blue')),
    secondary_y=True,
)

fig.update_layout(title_text="Daily Revenue and Transaction Volume")
fig.update_xaxis(title_text="Date")
fig.update_yaxis(title_text="Revenue ($)", secondary_y=False)
fig.update_yaxis(title_text="Transaction Count", secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

# Payment method analysis
st.subheader("💳 Payment Method Analysis")

col1, col2 = st.columns(2)

with col1:
    payment_revenue = filtered_df[filtered_df['transaction_success']].groupby('payment_method')['final_toll'].sum().reset_index()
    fig = px.pie(payment_revenue, values='final_toll', names='payment_method',
                 title="Revenue by Payment Method")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    payment_success = filtered_df.groupby('payment_method').agg({
        'transaction_success': ['count', 'sum']
    }).reset_index()
    payment_success.columns = ['payment_method', 'total_transactions', 'successful_transactions']
    payment_success['success_rate'] = (payment_success['successful_transactions'] / payment_success['total_transactions']) * 100
    
    fig = px.bar(payment_success, x='payment_method', y='success_rate',
                 title="Success Rate by Payment Method",
                 labels={'success_rate': 'Success Rate (%)'})
    fig.update_layout(yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig, use_container_width=True)

# Vehicle type revenue analysis
st.subheader("🚗 Revenue by Vehicle Type")

col1, col2 = st.columns(2)

with col1:
    vehicle_revenue = filtered_df[filtered_df['transaction_success']].groupby('vehicle_type')['final_toll'].sum().reset_index()
    fig = px.bar(vehicle_revenue, x='vehicle_type', y='final_toll',
                 title="Total Revenue by Vehicle Type",
                 labels={'final_toll': 'Revenue ($)'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    vehicle_stats = filtered_df[filtered_df['transaction_success']].groupby('vehicle_type').agg({
        'final_toll': ['count', 'mean']
    }).reset_index()
    vehicle_stats.columns = ['vehicle_type', 'transaction_count', 'avg_toll']
    
    fig = px.scatter(vehicle_stats, x='transaction_count', y='avg_toll', 
                     size='transaction_count', color='vehicle_type',
                     title="Transaction Volume vs Average Toll",
                     labels={'transaction_count': 'Number of Transactions', 'avg_toll': 'Average Toll ($)'})
    st.plotly_chart(fig, use_container_width=True)

# Peak pricing analysis
st.subheader("⏰ Peak Pricing Impact")

peak_analysis = filtered_df[filtered_df['transaction_success']].groupby('peak_multiplier').agg({
    'final_toll': ['sum', 'count', 'mean']
}).reset_index()
peak_analysis.columns = ['peak_multiplier', 'total_revenue', 'transaction_count', 'avg_toll']

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(peak_analysis, x='peak_multiplier', y='total_revenue',
                 title="Revenue by Peak Pricing Multiplier",
                 labels={'peak_multiplier': 'Peak Multiplier', 'total_revenue': 'Total Revenue ($)'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Hourly revenue pattern
    hourly_revenue = filtered_df[filtered_df['transaction_success']].groupby(
        filtered_df['timestamp'].dt.hour
    )['final_toll'].sum().reset_index()
    
    fig = px.line(hourly_revenue, x='timestamp', y='final_toll',
                  title="Hourly Revenue Pattern",
                  labels={'timestamp': 'Hour of Day', 'final_toll': 'Revenue ($)'})
    fig.update_layout(xaxis=dict(tickmode='linear'))
    st.plotly_chart(fig, use_container_width=True)

# Discount analysis
st.subheader("🎫 Discount & Promotion Analysis")

discount_analysis = filtered_df[filtered_df['transaction_success']].groupby('discount_type').agg({
    'final_toll': ['sum', 'count'],
    'discount_factor': 'mean'
}).reset_index()
discount_analysis.columns = ['discount_type', 'total_revenue', 'transaction_count', 'avg_discount_factor']

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(discount_analysis, x='discount_type', y='transaction_count',
                 title="Transaction Count by Discount Type",
                 labels={'discount_type': 'Discount Type', 'transaction_count': 'Transactions'})
    fig.update_xaxis(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Calculate revenue impact of discounts
    no_discount_revenue = discount_analysis[discount_analysis['discount_type'] == 'None']['total_revenue'].sum()
    total_discounted_revenue = discount_analysis[discount_analysis['discount_type'] != 'None']['total_revenue'].sum()
    
    # Estimate what revenue would have been without discounts
    discounted_transactions = filtered_df[
        (filtered_df['transaction_success']) & 
        (filtered_df['discount_type'] != 'None')
    ]
    potential_revenue = (discounted_transactions['final_toll'] / discounted_transactions['discount_factor']).sum()
    discount_impact = potential_revenue - total_discounted_revenue
    
    st.metric("Discount Impact", f"-${discount_impact:.2f}")
    st.metric("Discount Savings %", f"{(discount_impact / potential_revenue * 100):.1f}%")

# Lane performance
st.subheader("🛣️ Lane Performance Analysis")

lane_performance = filtered_df[filtered_df['transaction_success']].groupby('lane_id').agg({
    'final_toll': ['sum', 'count', 'mean'],
    'processing_time_sec': 'mean'
}).reset_index()
lane_performance.columns = ['lane_id', 'total_revenue', 'transaction_count', 'avg_toll', 'avg_processing_time']

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(lane_performance, x='lane_id', y='total_revenue',
                 title="Revenue by Lane",
                 labels={'lane_id': 'Lane', 'total_revenue': 'Total Revenue ($)'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(lane_performance, x='avg_processing_time', y='total_revenue',
                     size='transaction_count', color='lane_id',
                     title="Processing Time vs Revenue by Lane",
                     labels={'avg_processing_time': 'Avg Processing Time (sec)', 'total_revenue': 'Total Revenue ($)'})
    st.plotly_chart(fig, use_container_width=True)

# Financial metrics
st.subheader("📊 Financial Performance Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Revenue Breakdown**")
    # Calculate revenue by time period
    if len(date_range) == 2:
        days = (end_date - start_date).days + 1
        daily_avg = total_revenue / days if days > 0 else 0
        st.metric("Daily Average", f"${daily_avg:.2f}")
        
        weekly_avg = total_revenue / (days / 7) if days >= 7 else total_revenue
        st.metric("Weekly Average", f"${weekly_avg:.2f}")

with col2:
    st.write("**Transaction Metrics**")
    avg_processing_time = filtered_df['processing_time_sec'].mean()
    st.metric("Avg Processing Time", f"{avg_processing_time:.1f} sec")
    
    refund_rate = (filtered_df['refund_issued'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    st.metric("Refund Rate", f"{refund_rate:.3f}%")

with col3:
    st.write("**Quality Metrics**")
    dispute_rate = (filtered_df['dispute_filed'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    st.metric("Dispute Rate", f"{dispute_rate:.3f}%")
    
    failed_transactions = len(filtered_df[~filtered_df['transaction_success']])
    st.metric("Failed Transactions", f"{failed_transactions:,}")

# Recent transactions table
st.subheader("📋 Recent Transactions")

# Show recent transactions
recent_transactions = filtered_df.head(100)[
    ['timestamp', 'transaction_id', 'lane_id', 'vehicle_type', 'final_toll', 
     'payment_method', 'transaction_success', 'discount_type']
].copy()

# Format timestamp
recent_transactions['timestamp'] = recent_transactions['timestamp'].dt.strftime('%Y-%m-%d %H:%M')

# Add success indicator
recent_transactions['status'] = recent_transactions['transaction_success'].map({True: '✅', False: '❌'})

st.dataframe(
    recent_transactions[['timestamp', 'transaction_id', 'lane_id', 'vehicle_type', 
                        'final_toll', 'payment_method', 'status', 'discount_type']],
    use_container_width=True
)

# Export functionality
st.subheader("📤 Export Options")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Export Transaction Data"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("Generate Revenue Report"):
        report = f"""
        # Revenue Analysis Report
        
        **Period**: {date_range[0]} to {date_range[1]}
        **Total Revenue**: ${total_revenue:,.2f}
        **Total Transactions**: {total_transactions:,}
        **Average Transaction**: ${avg_transaction:.2f}
        **Success Rate**: {success_rate:.1f}%
        
        ## Top Performing Payment Methods
        {payment_revenue.to_string(index=False)}
        
        ## Lane Performance
        {lane_performance.to_string(index=False)}
        """
        
        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"revenue_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

with col3:
    if st.button("Export Failed Transactions"):
        failed_transactions = filtered_df[~filtered_df['transaction_success']]
        if not failed_transactions.empty:
            csv = failed_transactions.to_csv(index=False)
            st.download_button(
                label="Download Failed Transactions",
                data=csv,
                file_name=f"failed_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No failed transactions in selected period")
