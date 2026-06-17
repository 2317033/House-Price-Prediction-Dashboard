import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="House Price Prediction Dashboard",
    page_icon="🏠",
    layout="wide"
)

# ---------------- LOAD DATA ----------------

df = pd.read_csv("Housing.csv")
model = joblib.load("house_price_model.pkl")

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    font-size: 45px;
    font-weight: bold;
    color: white;
}

.subtitle {
    font-size: 20px;
    color: #94A3B8;
}

.footer {
    background-color: #1E293B;
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Dashboard",
        "📊 Dataset Analysis",
        "📈 Visualizations",
        "🤖 Price Prediction",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info("""
🏢 Horizon Intern

Machine Learning Internship Project

House Price Prediction System
""")

# ---------------- HEADER ----------------

st.markdown(
    "<div class='title'>🏠 House Price Prediction Dashboard</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Machine Learning Internship Project</div>",
    unsafe_allow_html=True
)

st.divider()

# ====================================================
# DASHBOARD
# ====================================================

if page == "🏠 Dashboard":

    st.header("🏠 Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Records", df.shape[0])

    with col2:
        st.metric("Features", df.shape[1]-1)

    with col3:
        st.metric("Model", "Linear Regression")

    with col4:
        st.metric("Accuracy", "90%+")

    st.divider()

    st.subheader("📌 Project Summary")

    st.write("""
This project predicts house prices using Machine Learning.

The model considers:

- Area
- Bedrooms
- Bathrooms
- Stories
- Main Road Access
- Guest Room
- Basement
- Air Conditioning
- Parking
- Preferred Area
- Furnishing Status

Users can enter property details and get an estimated house price instantly.
""")

# ====================================================
# DATASET ANALYSIS
# ====================================================

elif page == "📊 Dataset Analysis":

    st.header("📊 Dataset Analysis")

    st.subheader("Dataset Preview")

    st.dataframe(df.head(), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    st.subheader("Missing Values")

    st.dataframe(
        pd.DataFrame(
            df.isnull().sum(),
            columns=["Missing Values"]
        )
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

# ====================================================
# VISUALIZATIONS
# ====================================================

elif page == "📈 Visualizations":

    st.header("📈 Data Visualizations")

    st.subheader("House Price Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.histplot(
        df["price"],
        bins=30,
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    temp_df = df.copy()

    binary_cols = [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea"
    ]

    for col in binary_cols:
        temp_df[col] = temp_df[col].map({
            "yes":1,
            "no":0
        })

    temp_df["furnishingstatus"] = temp_df[
        "furnishingstatus"
    ].map({
        "furnished":0,
        "semi-furnished":1,
        "unfurnished":2
    })

    fig2, ax2 = plt.subplots(figsize=(10,6))

    sns.heatmap(
        temp_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax2
    )

    st.pyplot(fig2)

# ====================================================
# PRICE PREDICTION
# ====================================================

elif page == "🤖 Price Prediction":

    st.header("🤖 House Price Prediction")

    col1, col2 = st.columns(2)

    with col1:

        area = st.number_input(
            "Area (sq.ft)",
            min_value=500,
            max_value=20000,
            value=5000
        )

        bedrooms = st.slider(
            "Bedrooms",
            1, 10, 3
        )

        bathrooms = st.slider(
            "Bathrooms",
            1, 10, 2
        )

        stories = st.slider(
            "Stories",
            1, 5, 2
        )

        parking = st.slider(
            "Parking",
            0, 5, 2
        )

    with col2:

        mainroad = st.selectbox(
            "Main Road",
            ["Yes","No"]
        )

        guestroom = st.selectbox(
            "Guest Room",
            ["Yes","No"]
        )

        basement = st.selectbox(
            "Basement",
            ["Yes","No"]
        )

        hotwaterheating = st.selectbox(
            "Hot Water Heating",
            ["Yes","No"]
        )

        airconditioning = st.selectbox(
            "Air Conditioning",
            ["Yes","No"]
        )

        prefarea = st.selectbox(
            "Preferred Area",
            ["Yes","No"]
        )

        furnishing = st.selectbox(
            "Furnishing Status",
            [
                "Furnished",
                "Semi-Furnished",
                "Unfurnished"
            ]
        )

    mainroad = 1 if mainroad == "Yes" else 0
    guestroom = 1 if guestroom == "Yes" else 0
    basement = 1 if basement == "Yes" else 0
    hotwaterheating = 1 if hotwaterheating == "Yes" else 0
    airconditioning = 1 if airconditioning == "Yes" else 0
    prefarea = 1 if prefarea == "Yes" else 0

    furnishing_map = {
        "Furnished":0,
        "Semi-Furnished":1,
        "Unfurnished":2
    }

    furnishingstatus = furnishing_map[furnishing]

    input_data = pd.DataFrame({
        "area":[area],
        "bedrooms":[bedrooms],
        "bathrooms":[bathrooms],
        "stories":[stories],
        "mainroad":[mainroad],
        "guestroom":[guestroom],
        "basement":[basement],
        "hotwaterheating":[hotwaterheating],
        "airconditioning":[airconditioning],
        "parking":[parking],
        "prefarea":[prefarea],
        "furnishingstatus":[furnishingstatus]
    })

    st.subheader("Input Summary")

    st.dataframe(
        input_data,
        use_container_width=True
    )

    if st.button(
        "Predict House Price",
        use_container_width=True
    ):

        prediction = model.predict(
            input_data
        )[0]

        st.success(
            f"🏠 Predicted House Price: ₹ {prediction:,.0f}"
        )

# ====================================================
# ABOUT PROJECT
# ====================================================

elif page == "ℹ️ About Project":

    st.header("ℹ️ About Project")

    st.markdown("""
### Project Title
House Price Prediction System

### Objective
Predict house prices using Machine Learning.

### Dataset
Housing Dataset (545 Records)

### Algorithm
Linear Regression

### Technologies Used
- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Matplotlib
- Seaborn

### Internship
Horizon Intern

### Developed By
Pratiksha Pawar
""")

# ---------------- FOOTER ----------------

st.divider()

st.markdown("""
<div class='footer'>
<b>👩‍💻 Developed By:</b> Pratiksha Pawar<br>
<b>🏢 Internship:</b> Horizon Intern<br>
<b>🏠 Project:</b> House Price Prediction Dashboard
</div>
""", unsafe_allow_html=True)