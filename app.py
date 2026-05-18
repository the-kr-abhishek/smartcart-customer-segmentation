import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

# PAGE CONFIG 
st.set_page_config(
    page_title="SmartCart Customer Segmentation",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 SmartCart Customer Segmentation")
st.write(
    "Upload dataset and analyze customer purchasing behavior using clustering.")
#  DATASET UPLOAD 
uploaded_file = st.file_uploader(
    "Upload SmartCart Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload the dataset to continue.")
    st.stop()

data = pd.read_csv(uploaded_file)

#  PREPROCESSING 
def preprocess_data(df):

    data = df.copy()

    # Missing value handling
    si = SimpleImputer(strategy="median")
    data["Income"] = si.fit_transform(data[["Income"]])

    # Drop ID
    if "ID" in data.columns:
        data.drop("ID", axis=1, inplace=True)

    # Age Feature
    data["Age"] = 2026 - data["Year_Birth"]
    data.drop("Year_Birth", axis=1, inplace=True)

    # Customer Tenure
    data["Dt_Customer"] = pd.to_datetime(
        data["Dt_Customer"],
        format="%d-%m-%Y"
    )

    data["Customer_tenure"] = (
        data["Dt_Customer"].max()
        - data["Dt_Customer"]
    ).dt.days

    data.drop("Dt_Customer", axis=1, inplace=True)

    # Total Spent
    data["Total_Spent"] = (
        data["MntFruits"]
        + data["MntMeatProducts"]
        + data["MntFishProducts"]
        + data["MntWines"]
        + data["MntSweetProducts"]
        + data["MntGoldProds"]
    )

    # Children
    data["Children"] = (
        data["Kidhome"]
        + data["Teenhome"]
    )

    # Drop unnecessary columns
    data.drop([
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntWines",
        "MntSweetProducts",
        "Kidhome",
        "Teenhome",
        "MntGoldProds"
    ], axis=1, inplace=True)

    # Category Mapping
    data["Marital_Status"] = data["Marital_Status"].map({
        "Single": "Single",
        "Married": "Together",
        "Together": "Together",
        "Divorced": "Single",
        "Widow": "Single",
        "Alone": "Single",
        "Absurd": "Single",
        "YOLO": "Single"
    })

    data["Education"] = data["Education"].map({
        "PhD": "PostGraduate",
        "Master": "PostGraduate",
        "Graduation": "Graduate",
        "Basic": "UnderGraduate",
        "2n Cycle": "UnderGraduate"
    })

    # Remove outliers
    data = data[data["Age"] < 100]
    data = data[data["Income"] < 300000]

    # Encoding
    data_encoded = pd.get_dummies(
        data,
        columns=["Education", "Marital_Status"],
        dtype=int
    )

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_encoded)

    # PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # Clustering
    model = AgglomerativeClustering(n_clusters=5)
    labels = model.fit_predict(pca_data)

    data["Cluster"] = labels

    return data, pca_data


processed_data, pca_data = preprocess_data(data)

# DASHBOARD 
st.header("📊 Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Customers",
    processed_data.shape[0]
)

col2.metric(
    "Average Income",
    f"${processed_data['Income'].mean():,.0f}"
)

col3.metric(
    "Average Spending",
    f"${processed_data['Total_Spent'].mean():,.0f}"
)

#  DATA PREVIEW 
st.subheader("Dataset Preview")
st.dataframe(processed_data.head())

#  EDA 
st.header("📈 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        processed_data,
        x="Income",
        title="Income Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(
        processed_data,
        x="Age",
        title="Age Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# Spending vs Income
st.subheader("Income vs Total Spending")

fig = px.scatter(
    processed_data,
    x="Income",
    y="Total_Spent",
    color="Cluster"
)

st.plotly_chart(fig, use_container_width=True)

#  PCA CLUSTER 
st.header("🎯 Customer Segmentation")

pca_df = pd.DataFrame({
    "PCA1": pca_data[:, 0],
    "PCA2": pca_data[:, 1],
    "Cluster": processed_data["Cluster"]
})

fig = px.scatter(
    pca_df,
    x="PCA1",
    y="PCA2",
    color="Cluster",
    title="Customer Clusters"
)

st.plotly_chart(fig, use_container_width=True)

#  CLUSTER SUMMARY 
st.subheader("Cluster Summary")

cluster_summary = processed_data.groupby(
    "Cluster"
).mean(numeric_only=True)

st.dataframe(cluster_summary)

st.success(
    "Customers successfully segmented into 5 groups."
)

st.write("Made by Kumar Abhishek 🚀")