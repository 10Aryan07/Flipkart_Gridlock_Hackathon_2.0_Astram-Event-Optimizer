import streamlit as st
import requests
import pandas as pd
import pydeck as pdk
from datetime import datetime
import os

st.set_page_config(page_title="ASTraM Event Optimizer", layout="wide")
API_URL = "http://127.0.0.1:8000/simulate_event"

st.title("🚦 Gridlock Phase 2: ASTraM Event Optimizer")
st.markdown("Predict localized traffic breakdowns and optimize police resource deployment in real-time.")

@st.cache_data
def load_zones():
    try:
        frontend_dir = os.path.dirname(os.path.abspath(__file__))

        project_root = os.path.dirname(frontend_dir)

        csv_path = os.path.join(project_root, "data", "processed", "events_cleaned.csv")
        
        df = pd.read_csv(csv_path)
        return df.groupby('zone')[['latitude', 'longitude']].mean().reset_index()
    except Exception as e:
        st.sidebar.error(f"Data Load Error: {e}") 
        return pd.DataFrame({"zone": ["South Zone 1"], "latitude": [12.92], "longitude": [77.58]})

zone_data = load_zones()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Event Parameters")
    with st.form("event_form"):
        event_name = st.text_input("Event Name", "Severe Waterlogging at Junction")
        event_cause = st.selectbox("Event Cause", ["water_logging", "public_event", "procession", "protest", "construction", "vehicle_breakdown"])
        selected_zone = st.selectbox("Bengaluru Zone", zone_data['zone'].tolist())
        
        zone_lat = zone_data[zone_data['zone'] == selected_zone]['latitude'].values[0]
        zone_lng = zone_data[zone_data['zone'] == selected_zone]['longitude'].values[0]
        
        event_date = st.date_input("Date", datetime.today())
        event_time = st.time_input("Time", datetime.now().time())
        
        st.divider()
        st.markdown("Interactive Simulation Mode")
        st.caption("Bypass the ML baseline to stress-test the resource optimization algorithms.")
        override_severity_flag = st.checkbox("Enable Manual Severity Override")
        custom_severity = st.slider("Simulate Extreme Severity", 0.0, 1.0, 0.9) if override_severity_flag else None

        submit = st.form_submit_button("Simulate Impact & Optimize")

with col2:
    st.header("Impact Visualization & Deployment")
    
    if submit:
        start_dt = f"{event_date}T{event_time}"
        payload = {
            "event_name": event_name, "event_cause": event_cause,
            "latitude": float(zone_lat), "longitude": float(zone_lng),
            "zone": selected_zone, "start_datetime": start_dt,
            "override_severity": custom_severity
        }
        
        with st.spinner("Calculating Spatiotemporal Impact..."):
            try:
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    result = response.json()

                    m1, m2, m3 = st.columns(3)
                    speed_pct = int(result['predictions']['predicted_speed_multiplier'] * 100)
                    m1.metric("Traffic Speed", f"{speed_pct}% of Normal", "Degraded Flow", delta_color="inverse")
                    
                    time_saved = result['predictions']['duration_without_astram'] - result['predictions']['estimated_duration_minutes']
                    m2.metric("Est. Clearance Time", f"{result['predictions']['estimated_duration_minutes']} mins", f"-{time_saved} mins (ASTraM Impact)")
                    
                    m3.metric("Severity Index", f"{int(result['predictions']['severity_index'] * 100)} / 100")
                    
                    st.divider()

                    st.subheader("ASTraM Dispatch Manifest")
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Officers Required", result['resource_allocation']['recommended_officers'])
                    r2.metric("Barricades to Deploy", result['resource_allocation']['barricades_count'])
                    r3.metric("Diversion Perimeter", f"{result['resource_allocation']['diversion_radius_km']} km")

                    st.subheader("📍 Target Location & Diversion Perimeter")
                    map_data = pd.DataFrame({'lat': [zone_lat], 'lon': [zone_lng]})
                    
                    color = [255, 0, 0, 160] if result['predictions']['severity_index'] >= 0.7 else \
                            [255, 165, 0, 160] if result['predictions']['severity_index'] >= 0.4 else \
                            [0, 255, 0, 160]

                    impact_layer = pdk.Layer(
                        'ScatterplotLayer', data=map_data, get_position='[lon, lat]',
                        get_fill_color=color, get_radius=400, 
                    )
                    
                    diversion_radius_meters = result['resource_allocation']['diversion_radius_km'] * 1000
                    diversion_layer = pdk.Layer(
                        'ScatterplotLayer', data=map_data, get_position='[lon, lat]',
                        get_fill_color=[0, 0, 0, 0], get_line_color=[255, 204, 0, 255], 
                        get_radius=diversion_radius_meters, stroked=True, line_width_min_pixels=3,
                    )

                    view_state = pdk.ViewState(latitude=zone_lat, longitude=zone_lng, zoom=13, pitch=45)
                    st.pydeck_chart(pdk.Deck(layers=[impact_layer, diversion_layer], initial_view_state=view_state))
                    
                else:
                    st.error(f"Backend Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to Backend! Is FastAPI running in the other terminal?")
    else:
        st.info("Enter event details and click 'Simulate Impact & Optimize' to generate ASTraM deployment plans.")

        st.subheader("Regional Overview")
        
        default_view = pdk.ViewState(latitude=12.9716, longitude=77.5946, zoom=10, pitch=30)

        st.pydeck_chart(pdk.Deck(initial_view_state=default_view))