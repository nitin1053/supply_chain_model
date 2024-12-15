# import streamlit as st
# import pandas as pd
# import joblib

# # Load the model and scaler
# model = joblib.load("best_rf_model.pkl")
# scaler = joblib.load("scaler.pkl")

# # Define the feature order as used during training
# FEATURE_ORDER = [
#     "Region", "Lead_Time", "Supplier_Reliability", "Seasonality_Factor", 
#     "Stock_Remaining", "Stock_Needed", "Transportation_Cost", 
#     "Distance_to_Be_Covered", "Profit", "Loss", "Carbon_Footprint", 
#     "Transportation_Risk", "Warehouse_Capacity", "Order_Frequency", 
#     "Promotion"
# ]

# # Streamlit App
# st.title("Supply Chain Demand Prediction")
# st.write("This app predicts the demand for supply chain items based on input parameters.")

# # Input Form
# with st.form("prediction_form"):
#     st.subheader("Input the Supply Chain Parameters")

#     # Collecting user inputs
#     region = st.number_input("Region", min_value=0, step=1)
#     lead_time = st.number_input("Lead Time", min_value=0.0, step=0.1)
#     supplier_reliability = st.number_input("Supplier Reliability", min_value=0.0, max_value=1.0, step=0.01)
#     seasonality_factor = st.number_input("Seasonality Factor", min_value=0.0, max_value=1.0, step=0.01)
#     stock_remaining = st.number_input("Stock Remaining", min_value=0, step=1)
#     stock_needed = st.number_input("Stock Needed", min_value=0, step=1)
#     transportation_cost = st.number_input("Transportation Cost", min_value=0.0, step=0.1)
#     distance_to_be_covered = st.number_input("Distance to Be Covered", min_value=0.0, step=0.1)
#     profit = st.number_input("Profit", min_value=0.0, step=0.1)
#     loss = st.number_input("Loss", min_value=0.0, step=0.1)
#     carbon_footprint = st.number_input("Carbon Footprint", min_value=0.0, step=0.1)
#     transportation_risk = st.number_input("Transportation Risk", min_value=0.0, max_value=1.0, step=0.01)
#     warehouse_capacity = st.number_input("Warehouse Capacity", min_value=0, step=1)
#     order_frequency = st.number_input("Order Frequency", min_value=0, step=1)
#     promotion = st.number_input("Promotion (Binary: 0 or 1)", min_value=0, max_value=1, step=1)

#     submitted = st.form_submit_button("Predict")

# if submitted:
#     try:
#         # Preparing input data for prediction
#         input_data = {
#             "Region": region,
#             "Lead_Time": lead_time,
#             "Supplier_Reliability": supplier_reliability,
#             "Seasonality_Factor": seasonality_factor,
#             "Stock_Remaining": stock_remaining,
#             "Stock_Needed": stock_needed,
#             "Transportation_Cost": transportation_cost,
#             "Distance_to_Be_Covered": distance_to_be_covered,
#             "Profit": profit,
#             "Loss": loss,
#             "Carbon_Footprint": carbon_footprint,
#             "Transportation_Risk": transportation_risk,
#             "Warehouse_Capacity": warehouse_capacity,
#             "Order_Frequency": order_frequency,
#             "Promotion": promotion
#         }

#         # Create DataFrame with correct feature order
#         new_data = pd.DataFrame([input_data], columns=FEATURE_ORDER)

#         # Preprocess the data
#         new_data[FEATURE_ORDER] = scaler.transform(new_data[FEATURE_ORDER])

#         # Make predictions
#         prediction = model.predict(new_data)[0]

#         # Display the prediction result
#         st.success(f"Predicted Demand: {prediction}")

#     except Exception as e:
#         st.error(f"Error: {str(e)}")







import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Load the model and scaler
@st.cache_resource
def load_models():
    model = joblib.load("best_rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_models()

# Define regions mapping
REGIONS = {
    'North': 0,
    'South': 1,
    'East': 2,
    'West': 3
}

# Define the features that need scaling
FEATURES_TO_SCALE = [
    'Lead_Time', 'Supplier_Reliability', 'Seasonality_Factor',
    'Stock_Remaining', 'Stock_Needed', 'Transportation_Cost',
    'Distance_to_Be_Covered', 'Profit', 'Loss', 'Carbon_Footprint',
    'Transportation_Risk', 'Warehouse_Capacity', 'Order_Frequency'
]

# Function to create radar chart
def create_radar_chart(input_data):
    categories = ['Supplier Reliability', 'Seasonality', 'Stock Level', 
                 'Transport Risk', 'Cost Efficiency', 'Environmental Impact']
    
    # Normalize values for radar chart
    stock_level = (input_data['Stock_Remaining'].iloc[0] / input_data['Stock_Needed'].iloc[0]) * 100
    cost_efficiency = 100 - (input_data['Transportation_Cost'].iloc[0] / 500 * 100)  # Normalized to 0-100
    environmental_impact = 100 - (input_data['Carbon_Footprint'].iloc[0] / 500 * 100)  # Normalized to 0-100
    
    values = [
        input_data['Supplier_Reliability'].iloc[0] * 100,
        input_data['Seasonality_Factor'].iloc[0] * 66.7,  # Scale 1.5 to 100
        stock_level,
        (1 - input_data['Transportation_Risk'].iloc[0]) * 100,
        cost_efficiency,
        environmental_impact
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Supply Chain Metrics'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Supply Chain Performance Radar"
    )
    return fig

# Function to create gauge chart
def create_gauge_chart(value, title, max_value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={'axis': {'range': [None, max_value]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, max_value/3], 'color': "lightgray"},
                   {'range': [max_value/3, 2*max_value/3], 'color': "gray"},
                   {'range': [2*max_value/3, max_value], 'color': "darkgray"}
               ]}))
    fig.update_layout(height=200)
    return fig

# Streamlit App
st.title("Supply Chain Demand Prediction")
st.write("""
This app predicts supply chain demand based on various parameters. 
Please input the values below and click 'Predict' to get the demand forecast.
""")

# Create two columns for better layout
col1, col2 = st.columns(2)

with st.form("prediction_form"):
    # First column of inputs
    with col1:
        st.subheader("Basic Parameters")
        region = st.selectbox("Region", options=list(REGIONS.keys()))
        lead_time = st.slider("Lead Time (days)", min_value=1, max_value=15, value=7)
        supplier_reliability = st.slider("Supplier Reliability", min_value=0.8, max_value=1.0, value=0.9, step=0.01)
        promotion = st.radio("Promotion Active", options=["No", "Yes"])
        seasonality_factor = st.slider("Seasonality Factor", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
        stock_remaining = st.number_input("Stock Remaining", min_value=0, max_value=500, value=100)
        stock_needed = st.number_input("Stock Needed", min_value=50, max_value=1000, value=200)

    # Second column of inputs
    with col2:
        st.subheader("Advanced Parameters")
        transportation_cost = st.slider("Transportation Cost ($)", min_value=100, max_value=500, value=250)
        distance = st.slider("Distance (km)", min_value=10, max_value=1000, value=500)
        profit = st.slider("Expected Profit ($)", min_value=500, max_value=5000, value=2500)
        loss = st.slider("Expected Loss ($)", min_value=0, max_value=1000, value=100)
        carbon_footprint = st.slider("Carbon Footprint (kg CO2)", min_value=50, max_value=500, value=200)
        transportation_risk = st.slider("Transportation Risk", min_value=0.1, max_value=0.9, value=0.5)
        warehouse_capacity = st.slider("Warehouse Capacity", min_value=100, max_value=1000, value=500)
        order_frequency = st.slider("Order Frequency (per month)", min_value=1, max_value=30, value=15)

    submitted = st.form_submit_button("Predict Demand")

if submitted:
    try:
        # Convert inputs to proper format
        input_data = pd.DataFrame({
            'Region': [REGIONS[region]],
            'Lead_Time': [lead_time],
            'Supplier_Reliability': [supplier_reliability],
            'Promotion': [1 if promotion == "Yes" else 0],
            'Seasonality_Factor': [seasonality_factor],
            'Stock_Remaining': [stock_remaining],
            'Stock_Needed': [stock_needed],
            'Transportation_Cost': [transportation_cost],
            'Distance_to_Be_Covered': [distance],
            'Profit': [profit],
            'Loss': [loss],
            'Carbon_Footprint': [carbon_footprint],
            'Transportation_Risk': [transportation_risk],
            'Warehouse_Capacity': [warehouse_capacity],
            'Order_Frequency': [order_frequency]
        })

        # Scale the features that need scaling
        input_data_scaled = input_data.copy()
        input_data_scaled[FEATURES_TO_SCALE] = scaler.transform(input_data[FEATURES_TO_SCALE])

        # Make prediction
        prediction = model.predict(input_data_scaled)[0]
        
        # Display results
        st.success(f"Predicted Demand: {int(prediction)} units")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Key Metrics", "Performance Radar", "Cost Analysis"])
        
        with tab1:
            # Display gauge charts in three columns
            gcol1, gcol2, gcol3 = st.columns(3)
            
            with gcol1:
                inventory_ratio = (stock_remaining / stock_needed) * 100
                fig_inventory = create_gauge_chart(inventory_ratio, "Inventory Coverage (%)", 100)
                st.plotly_chart(fig_inventory, use_container_width=True)
                
            with gcol2:
                capacity_usage = (prediction / warehouse_capacity) * 100
                fig_capacity = create_gauge_chart(capacity_usage, "Capacity Usage (%)", 100)
                st.plotly_chart(fig_capacity, use_container_width=True)
                
            with gcol3:
                cost_efficiency = ((profit - loss) / profit) * 100
                fig_efficiency = create_gauge_chart(cost_efficiency, "Cost Efficiency (%)", 100)
                st.plotly_chart(fig_efficiency, use_container_width=True)
        
        with tab2:
            # Display radar chart
            radar_fig = create_radar_chart(input_data)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with tab3:
            # Create cost breakdown pie chart
            cost_data = {
                'Category': ['Transportation', 'Storage', 'Risk-related', 'Other'],
                'Cost': [transportation_cost, 
                        warehouse_capacity * 0.1,  # Assumed storage cost
                        loss * transportation_risk,
                        loss * (1 - transportation_risk)]
            }
            fig_costs = px.pie(cost_data, values='Cost', names='Category', 
                             title='Cost Breakdown Analysis')
            st.plotly_chart(fig_costs, use_container_width=True)

            # Create bar chart for cost metrics
            cost_metrics = {
                'Metric': ['Cost per Unit', 'Cost per km', 'Cost per Order'],
                'Value': [transportation_cost/prediction if prediction > 0 else 0,
                         transportation_cost/distance,
                         transportation_cost/order_frequency]
            }
            fig_metrics = px.bar(cost_metrics, x='Metric', y='Value',
                               title='Cost Efficiency Metrics')
            st.plotly_chart(fig_metrics, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check if all input values are within valid ranges and try again.")

# Add explanatory notes
with st.expander("About this Prediction Model"):
    st.write("""
    This model uses a Random Forest Regressor trained on historical supply chain data to predict demand. 
    The prediction takes into account multiple factors including:
    - Regional variations and seasonality
    - Supply chain metrics (lead time, reliability)
    - Inventory levels and capacity
    - Cost and environmental factors
    - Risk factors and operational parameters
    
    The visualizations provide:
    1. Key Metrics: Gauge charts showing crucial performance indicators
    2. Performance Radar: A comprehensive view of supply chain performance across multiple dimensions
    3. Cost Analysis: Detailed breakdown of costs and efficiency metrics
    
    The model has been trained on normalized data to ensure consistent predictions across different scales of input.
    """)