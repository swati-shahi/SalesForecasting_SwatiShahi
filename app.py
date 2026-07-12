 
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
 

# PAGE CONFIG

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
 

# LOAD DATA (cached so it only loads once, not on every click)
# ----------------------------------------------------------------------------

@st.cache_data
def load_data():
    train = pd.read_csv("train.csv")
    train["Order Date"] = pd.to_datetime(train["Order Date"], dayfirst=True)
    train["Year"] = train["Order Date"].dt.year
 
    monthly_sales = pd.read_csv("monthly_sales.csv", index_col=0, parse_dates=True)["Sales"]
    weekly_sales = pd.read_csv("weekly_sales.csv", index_col=0, parse_dates=True)["Sales"]
    anomaly_summary = pd.read_csv("anomaly_summary.csv", parse_dates=["Week"])
    cluster_segments = pd.read_csv("cluster_segments.csv", index_col=0)
    model_metrics = pd.read_csv("model_metrics.csv")
    forecast_results = pd.read_csv("forecast_results.csv", parse_dates=["Date"])
 
    return train, monthly_sales, weekly_sales, anomaly_summary, cluster_segments, model_metrics, forecast_results
 
train, monthly_sales, weekly_sales, anomaly_summary, cluster_segments, model_metrics, forecast_results = load_data()
 


# SIDEBAR NAVIGATION -----------------------------------------------------------------------

st.sidebar.title("Sales Forecasting System")
page = st.sidebar.radio(
    "Navigate to:",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"]
)
 

# PAGE 1 — SALES OVERVIEW DASHBOARD------------------------------------------------------------

if page == "Sales Overview":
    st.title("Sales Overview Dashboard")
 
    col1, col2 = st.columns(2)
 
    with col1:
        st.subheader("Total Sales by Year")
        yearly_sales = train.groupby("Year")["Sales"].sum().reset_index()
        fig_year = px.bar(yearly_sales, x="Year", y="Sales", text_auto=".2s")
        st.plotly_chart(fig_year, use_container_width=True)
 
    with col2:
        st.subheader("Monthly Sales Trend")
        monthly_df = monthly_sales.reset_index()
        monthly_df.columns = ["Date", "Sales"]
        fig_month = px.line(monthly_df, x="Date", y="Sales", markers=True)
        st.plotly_chart(fig_month, use_container_width=True)
 
    st.subheader("Sales by Region and Category")
    col3, col4 = st.columns(2)
    with col3:
        region_filter = st.multiselect(
            "Filter by Region", options=train["Region"].unique(), default=list(train["Region"].unique())
        )
    with col4:
        category_filter = st.multiselect(
            "Filter by Category", options=train["Category"].unique(), default=list(train["Category"].unique())
        )
 
    filtered = train[train["Region"].isin(region_filter) & train["Category"].isin(category_filter)]
    grouped = filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
    fig_grouped = px.bar(grouped, x="Region", y="Sales", color="Category", barmode="group")
    st.plotly_chart(fig_grouped, use_container_width=True)
 


# PAGE 2 — FORECAST EXPLORER------------------------------------------------------------------------------------

elif page == "Forecast Explorer":
    st.title("Forecast Explorer")
    st.caption("Forecasts generated using SARIMA - the best-performing model identified in Task 3.")
 
    segment_options = forecast_results["Segment"].unique().tolist()
    selected_segment = st.selectbox("Select Category or Region", segment_options)
 
    horizon = st.slider("Forecast horizon (months ahead)", min_value=1, max_value=3, value=3)
 
    segment_forecast = forecast_results[forecast_results["Segment"] == selected_segment].sort_values("Date")
    segment_forecast_display = segment_forecast.head(horizon)
 
    # Show relevant history for context: company-wide history for Company-Wide,
    # otherwise the model doesn't have segment history saved, so we note that.
    if selected_segment == "Company-Wide":
        history_df = monthly_sales.reset_index()
        history_df.columns = ["Date", "Sales"]
        history_df = history_df.tail(12)
 
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history_df["Date"], y=history_df["Sales"], name="Actual (last 12 months)", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=segment_forecast_display["Date"], y=segment_forecast_display["Forecast"],
                                  name="Forecast", mode="lines+markers", line=dict(dash="dash", color="red")))
        fig.update_layout(title=f"{selected_segment}: Actual vs Forecast", xaxis_title="Date", yaxis_title="Sales ($)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.line(segment_forecast_display, x="Date", y="Forecast", markers=True,
                       title=f"{selected_segment}: {horizon}-Month Forecast")
        st.plotly_chart(fig, use_container_width=True)
 
    st.subheader(f"Forecast Values ({horizon} month{'s' if horizon > 1 else ''})")
    st.dataframe(segment_forecast_display[["Date", "Forecast"]].style.format({"Forecast": "${:,.2f}"}))
 
    st.subheader("Model Accuracy (SARIMA, evaluated on Task 3's held-out test months)")
    sarima_metrics = model_metrics[model_metrics["Model"] == "SARIMA"].iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"${sarima_metrics['MAE']:,.2f}")
    m2.metric("RMSE", f"${sarima_metrics['RMSE']:,.2f}")
    m3.metric("MAPE", f"{sarima_metrics['MAPE']:.2f}%")
 


# PAGE 3 — ANOMALY REPORT-----------------------------------------------------------------------------------------------

elif page == "Anomaly Report":
    st.title("Anomaly Report")
    st.caption("Weeks flagged as unusual by Isolation Forest and/or Z-Score methods (Task 5).")
 
    weekly_df = weekly_sales.reset_index()
    weekly_df.columns = ["Date", "Sales"]
 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_df["Date"], y=weekly_df["Sales"], name="Weekly Sales", mode="lines"))
 
    iso_anomalies = anomaly_summary[anomaly_summary["Flagged_by_IsolationForest"]]
    zscore_anomalies = anomaly_summary[anomaly_summary["Flagged_by_ZScore"]]
 
    fig.add_trace(go.Scatter(x=iso_anomalies["Week"], y=iso_anomalies["Sales"], mode="markers",
                              name="Isolation Forest Anomaly", marker=dict(color="red", size=10)))
    fig.add_trace(go.Scatter(x=zscore_anomalies["Week"], y=zscore_anomalies["Sales"], mode="markers",
                              name="Z-Score Anomaly", marker=dict(color="orange", size=10, symbol="diamond")))
 
    fig.update_layout(title="Weekly Sales with Detected Anomalies", xaxis_title="Date", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)
 
    st.subheader("Detected Anomalies")
    display_table = anomaly_summary.copy()
    display_table["Week"] = display_table["Week"].dt.strftime("%Y-%m-%d")
    st.dataframe(display_table.style.format({"Sales": "${:,.2f}"}))
 


# PAGE 4 — PRODUCT DEMAND SEGMENTS----------------------------------------------------------------------------------------------

elif page == "Product Demand Segments":
    st.title("Product Demand Segments")
    st.caption("Sub-categories grouped by K-Means clustering on volume, growth, volatility, and order value (Task 6).")
 
    fig = px.scatter(
        cluster_segments.reset_index(),
        x="PCA1", y="PCA2",
        color="Cluster_Label",
        text="Sub-Category",
        size="Total_Sales",
        hover_data=["Total_Sales", "Growth_Rate_Pct", "Volatility", "Avg_Order_Value"],
        title="Product Sub-Category Clusters (PCA 2D Projection)"
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)
 
    st.subheader("Sub-Categories by Cluster")
    for label in cluster_segments["Cluster_Label"].unique():
        with st.expander(f"{label}"):
            members = cluster_segments[cluster_segments["Cluster_Label"] == label]
            st.dataframe(
                members[["Total_Sales", "Growth_Rate_Pct", "Volatility", "Avg_Order_Value"]]
                .style.format({
                    "Total_Sales": "${:,.2f}",
                    "Growth_Rate_Pct": "{:.1f}%",
                    "Volatility": "${:,.2f}",
                    "Avg_Order_Value": "${:,.2f}",
                })
            )
 