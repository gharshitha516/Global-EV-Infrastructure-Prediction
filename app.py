import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# --- Load Data ---
@st.cache_data
def load_data():
    stations = pd.read_csv("clean_ev_stations.csv")
    models = pd.read_csv("clean_ev_models.csv")
    country_summary = pd.read_csv("clean_country_summary.csv")
    world_summary = pd.read_csv("clean_world_summary.csv")
    return stations, models, country_summary, world_summary

stations, models, country_summary, world_summary = load_data()

# --- Sidebar Navigation ---
st.sidebar.markdown(
    """
    <h2 style='margin-bottom:0; font-weight:600;'> → EV Insights</h2>
    <hr style='margin:0.5rem 0;'>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "",
    [
        "Overview",
        "Global Insights",
        "EV Models",
        "Charging Infrastructure",
        "EV Forecasts"
    ]
)

# --- Overview ---
if page == "Overview":
    st.markdown(
        """
        <h1 style='font-size:32px; margin-bottom:0;'><b>🌐 Global EV Infrastructure Insights</b></h1>
        <p style='font-size:16px; color:#555;'>
        <b>A global analytics platform for Electric Vehicle adoption, charging networks, and growth forecasts</b>.
        </p>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Charging Stations", f"{stations.shape[0]:,}")
    col2.metric("EV Models", f"{models.shape[0]:,}")
    col3.metric("Countries", f"{country_summary.shape[0]:,}")

    # Show table directly instead of expander
    st.subheader("Global Data Overview")
    st.dataframe(world_summary.head())


# --- Global Insights ---
elif page == "Global Insights":
    st.title("Global Charging Infrastructure")
    st.caption("Global overview of EV charging density, hotspots, and geographic trends.")

    stations_count = stations.groupby("country_code").size().reset_index(name="station_count")

    tab1, tab2, tab3 = st.tabs(["Choropleth Map", "Top 10 Countries", "Station Locations"])

    with tab1:
        fig = px.choropleth(
            stations_count,
            locations="country_code",
            color="station_count",
            hover_name="country_code",
            color_continuous_scale="Viridis",
            title="EV Charging Stations by Country"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        top10 = stations_count.sort_values(by="station_count", ascending=False).head(10)
        fig2 = px.bar(
            top10,
            x="country_code",
            y="station_count",
            color="station_count",
            color_continuous_scale="Plasma",
            title="Top 10 Countries by Charging Infrastructure"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        sampled = stations.sample(n=min(5000, len(stations)), random_state=42)
        fig3 = px.scatter_geo(
            sampled,
            lat="latitude",
            lon="longitude",
            color="is_fast_dc",
            color_discrete_map={True: "red", False: "blue"},
            hover_name="name",
            title="Sampled Charging Stations (Fast vs Slow)",
            projection="natural earth"
        )
        st.plotly_chart(fig3, use_container_width=True)

# --- EV Models ---
elif page == "EV Models":
    st.title("EV Model Analytics")
    st.caption("Explore EV models by manufacturer and launch year.")

    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Select Manufacturer:", ["All"] + sorted(models["make"].unique().tolist()))
    with col2:
        year = st.selectbox("Select Launch Year:", ["All"] + sorted(models["first_year"].unique().tolist()))

    filtered = models.copy()
    if brand != "All":
        filtered = filtered[filtered["make"] == brand]
    if year != "All":
        filtered = filtered[filtered["first_year"] == int(year)]

    st.write(f"Displaying {len(filtered)} matching EV models")
    st.dataframe(filtered)

# --- Charging Insights ---
elif page == "Charging Infrastructure":
    st.title("EV Charging Network Overview")
    st.caption("Analysis of charger distribution and country level infrastructure.")

    tab1, tab2 = st.tabs(["Charger Mix", "Country Clustering"])

    with tab1:
        st.subheader("Charger Type Distribution")
        charger_counts = stations["is_fast_dc"].value_counts().reset_index()
        charger_counts.columns = ["Charger Type", "Count"]
        charger_counts["Charger Type"] = charger_counts["Charger Type"].map({True: "Fast DC", False: "Slow AC"})

        fig = px.bar(
            charger_counts,
            x="Charger Type",
            y="Count",
            color="Charger Type",
            color_discrete_map={"Fast DC": "red", "Slow AC": "blue"},
            title="Fast vs Slow Charger Distribution",
            text="Count"
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Country Level Clustering")
        infra = country_summary.copy()
        kmeans = KMeans(n_clusters=3, random_state=42)
        infra["cluster"] = kmeans.fit_predict(infra[["stations"]])

        fig2 = px.scatter(
            infra,
            x="country_code",
            y="stations",
            color="cluster",
            title="Clusters of Countries by Charging Infrastructure",
            size="stations",
            hover_data=["stations"]
        )
        st.plotly_chart(fig2, use_container_width=True)

# --- EV Forecasts ---
elif page == "EV Forecasts":
    st.title("EV Market Forecasts")
    st.caption("ML based predictions for EV growth and infrastructure expansion.")

    tab1, tab2 = st.tabs(["EV Model Forecast", "Charging Station Forecast"])

    with tab1:
        st.subheader("Predicted EV Model Growth")
        growth = models.groupby("first_year").size().reset_index(name="count")
        X = growth["first_year"].values.reshape(-1, 1)
        y = growth["count"].values
        model = LinearRegression().fit(X, y)
        future_years = pd.DataFrame({"first_year": range(growth["first_year"].min(), 2030)})
        future_years["prediction"] = model.predict(future_years[["first_year"]])

        fig = px.line(
            future_years,
            x="first_year",
            y="prediction",
            markers=True,
            line_shape="spline",
            color_discrete_sequence=["green"],
            title="Forecast: EV Models Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Predicted Charging Network Growth")
        X2 = world_summary.index.values.reshape(-1, 1)
        y2 = world_summary["count"].values
        model2 = LinearRegression().fit(X2, y2)
        future_idx = pd.DataFrame({"idx": range(len(world_summary) + 5)})
        future_idx["prediction"] = model2.predict(future_idx[["idx"]])

        fig2 = px.line(
            future_idx,
            x="idx",
            y="prediction",
            markers=True,
            line_shape="spline",
            color_discrete_sequence=["orange"],
            title="Forecast: Charging Stations (Timeline Index)"
        )
        st.plotly_chart(fig2, use_container_width=True)

