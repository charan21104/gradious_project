import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

st.set_page_config(page_title="Fuel Efficiency Predictor", layout="wide")

@st.cache_data
def load_and_train_model():
    df = pd.read_csv('automobile_performance.csv')

    df['power_output'] = pd.to_numeric(df['power_output'], errors='coerce')
    df.dropna(inplace=True)

    numeric_cols = ['fuel_efficiency', 'engine_volume', 'power_output', 'vehicle_mass', 'accel_capability']
    
    df_cleaned = df.copy()
    for col in numeric_cols:
        q1 = df_cleaned[col].quantile(0.25)
        q3 = df_cleaned[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

    df_final = df_cleaned.copy()
    
    features = [
        'engine_volume', 'power_output', 'vehicle_mass', 'accel_capability', 
        'release_year', 'engine_config', 'manufacture_region'
    ]
    X = df_final[features]
    y = df_final['fuel_efficiency']
    
    X_encoded = pd.get_dummies(X, columns=['engine_config', 'manufacture_region'], drop_first=True)
    
    model_columns = X_encoded.columns
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_encoded, y)
    
    return model, model_columns, df

model, model_columns, original_df = load_and_train_model()

st.title('🚗 Vehicle Fuel Efficiency Predictor')
st.markdown("Enter the vehicle's attributes to predict its fuel efficiency (in MPG).")

st.sidebar.header("Vehicle Attributes")

def user_inputs():
    col1, col2 = st.sidebar.columns(2)

    engine_volume = col1.slider(
        'Engine Volume (cubic inches)', 
        min_value=float(original_df['engine_volume'].min()), 
        max_value=float(original_df['engine_volume'].max()),
        value=150.0,
        step=1.0
    )
    power_output = col2.slider(
        'Power Output (horsepower)', 
        min_value=float(original_df['power_output'].min()), 
        max_value=float(original_df['power_output'].max()),
        value=100.0,
        step=1.0
    )
    vehicle_mass = col1.slider(
        'Vehicle Mass (lbs)', 
        min_value=float(original_df['vehicle_mass'].min()),
        max_value=float(original_df['vehicle_mass'].max()),
        value=3000.0,
        step=10.0
    )
    accel_capability = col2.slider(
        'Acceleration Capability (0-60 mph in sec)', 
        min_value=float(original_df['accel_capability'].min()),
        max_value=float(original_df['accel_capability'].max()),
        value=15.0,
        step=0.1
    )

    release_year = st.sidebar.slider(
        'Release Year',
        min_value=int(original_df['release_year'].min()),
        max_value=int(original_df['release_year'].max()),
        value=78,
        step=1
    )
    
    engine_config_map = {
        '3 Cylinders': 3, '4 Cylinders': 4, '5 Cylinders': 5, 
        '6 Cylinders': 6, '8 Cylinders': 8
    }
    engine_config_str = st.sidebar.selectbox(
        'Engine Configuration (Cylinders)', 
        options=list(engine_config_map.keys()),
        index=1
    )
    engine_config = engine_config_map[engine_config_str]

    manufacture_region_map = {'USA': 1, 'Europe': 2, 'Japan': 3}
    manufacture_region_str = st.sidebar.selectbox(
        'Manufacture Region', 
        options=list(manufacture_region_map.keys()),
        index=2
    )
    manufacture_region = manufacture_region_map[manufacture_region_str]
    
    data = {
        'engine_volume': engine_volume,
        'power_output': power_output,
        'vehicle_mass': vehicle_mass,
        'accel_capability': accel_capability,
        'release_year': release_year,
        'engine_config': engine_config,
        'manufacture_region': manufacture_region
    }
    
    return pd.DataFrame(data, index=[0])

input_df = user_inputs()

prediction_df = pd.DataFrame(columns=model_columns)

input_encoded = pd.get_dummies(input_df, columns=['engine_config', 'manufacture_region'], drop_first=True)

prediction_aligned = input_encoded.reindex(columns=model_columns, fill_value=False)

prediction = model.predict(prediction_aligned)

st.subheader("Prediction")

st.write("Based on the following vehicle attributes:")
col1, col2, col3 = st.columns(3)
col1.metric("Engine Volume", f"{input_df['engine_volume'].iloc[0]} cu in")
col2.metric("Power Output", f"{input_df['power_output'].iloc[0]} HP")
col3.metric("Vehicle Mass", f"{input_df['vehicle_mass'].iloc[0]} lbs")
col1.metric("Acceleration", f"{input_df['accel_capability'].iloc[0]} sec")
col2.metric("Year", f"19{input_df['release_year'].iloc[0]}")
col3.metric("Cylinders", f"{input_df['engine_config'].iloc[0]}")

st.markdown("---")
st.markdown("### Predicted Fuel Efficiency:")
st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>{prediction[0]:.2f} MPG</h2>", unsafe_allow_html=True)
st.markdown("---")

st.info(
    "**Disclaimer:** This prediction is based on a machine learning model trained on the *Automobile Performance* dataset. "
    "Actual fuel efficiency may vary based on driving conditions, vehicle maintenance, and other factors."
)