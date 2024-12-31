import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import plotly.express as px

# Fungsi untuk memuat model
def load_model(file_name):
    try:
        with open(file_name, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"File model {file_name} tidak ditemukan.")
    except Exception as e:
        st.error(f"Terjadi error saat memuat model {file_name}: {e}")

# Load models
regression_model = load_model("model.pkl")  # Model regresi
clustering_model = load_model("modelClustering.pkl")  # Model clustering

if not regression_model or not clustering_model:
    st.stop()

# Load dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("wines_SPA.csv")  # Gunakan dataset raw
        return data
    except FileNotFoundError:
        st.error("File dataset wines_SPA.csv tidak ditemukan.")
        return pd.DataFrame()  # Return dataframe kosong jika error

data = load_data()

if data.empty:
    st.stop()

# Pastikan tipe data konsisten di setiap kolom
data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
data["body"] = pd.to_numeric(data["body"], errors="coerce")
data["year"] = pd.to_numeric(data["year"], errors="coerce")
data["region"] = data["region"].astype(str)
data["type"] = data["type"].astype(str)
data["price"] = pd.to_numeric(data["price"], errors="coerce")

# Extract unique options for dropdowns
rating_options = sorted(data["rating"].dropna().unique())
region_options = sorted(data["region"].dropna().unique())
type_options = sorted(data["type"].dropna().unique())
body_options = sorted(data["body"].dropna().unique())
year_options = sorted(data["year"].dropna().unique())

# Streamlit UI
st.title("Wine Dashboard")
st.write("Analyze wine data using regression and clustering.")

# Sidebar Inputs for Regression
st.sidebar.header("Input Wine Attributes for Regression")
rating = st.sidebar.selectbox("Rating", rating_options)
region = st.sidebar.selectbox("Region", region_options)
wine_type = st.sidebar.selectbox("Type", type_options)
body = st.sidebar.selectbox("Body", body_options)
year = st.sidebar.selectbox("Year", year_options)

if st.sidebar.button("Predict Price"):
    # Prepare Input for Prediction
    input_data = pd.DataFrame({
        'rating': [rating],
        'region': [region],
        'type': [wine_type],
        'body': [body],
        'year': [year]
    })

    st.write("Input Data:")
    st.dataframe(input_data)

    try:
        # Perform OneHotEncoding for categorical columns
        categorical_columns = ["region", "type"]
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(data[categorical_columns])  # Fit encoder to the original dataset

        encoded_cats = encoder.transform(input_data[categorical_columns])
        encoded_cat_columns = encoder.get_feature_names_out(categorical_columns)

        # Combine numerical and encoded categorical features
        input_data = input_data.drop(columns=categorical_columns)
        input_data = pd.concat(
            [input_data, pd.DataFrame(encoded_cats, columns=encoded_cat_columns)],
            axis=1
        )

        # Cocokkan urutan kolom dengan model
        trained_features = regression_model.feature_names_in_  # Fitur yang digunakan saat pelatihan
        input_data = input_data.reindex(columns=trained_features, fill_value=0)

        # Predict Price
        prediction = regression_model.predict(input_data)[0]
        st.write(f"Predicted Price: €{prediction:.2f}")
    except Exception as e:
        st.error(f"Error during regression prediction: {e}")

# Clustering Analysis
st.header("Clustering Analysis")
selected_region = st.selectbox("Select a Region for Clustering Analysis", region_options)

# Prepare Data for Clustering
features = ["price", "rating", "body", "acidity", "year"]  # Sesuaikan dengan dataset
filtered_data = data.dropna(subset=features)  # Hanya gunakan data lengkap untuk clustering

try:
    # Scaling only on the filtered data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(filtered_data[features])

    # Predict clusters for filtered data
    filtered_data["Cluster"] = clustering_model.predict(scaled_features)

    # Filter data by selected region
    region_data = filtered_data[filtered_data["region"] == selected_region]

    if not region_data.empty:
        # Calculate cluster percentages
        cluster_counts = region_data["Cluster"].value_counts(normalize=True) * 100
        cluster_df = cluster_counts.reset_index()
        cluster_df.columns = ["Cluster", "Percentage"]

        # Create pie chart
        fig = px.pie(cluster_df, names="Cluster", values="Percentage",
                     title=f"Cluster Distribution for Region: {selected_region}",
                     labels={"Cluster": "Cluster", "Percentage": "Percentage"})
        st.plotly_chart(fig)
    else:
        st.write(f"No data available for the selected region: {selected_region}")
except Exception as e:
    st.error(f"Error during clustering analysis: {e}")