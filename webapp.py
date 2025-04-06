import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import tempfile
import streamlit.components.v1 as components
st.set_page_config("Churn EDA & Predictor", layout="wide")
st.title("📊 Telco Churn Dashboard")

# Load model
model = XGBClassifier()
model.load_model("xgboost.json")
expected_cols = joblib.load("expected_columns.joblib")

# Label encoder setup
labelencoder = LabelEncoder()
label_fit_values = {
    'gender': ['Female', 'Male'],
    'Partner': ['No', 'Yes'],
    'Dependents': ['No', 'Yes'],
    'PhoneService': ['No', 'Yes'],
    'PaperlessBilling': ['No', 'Yes'],
    "SeniorCitizen": ["No", "Yes"],
}
def encode_label(col, val):
    labelencoder.fit(label_fit_values[col])
    return labelencoder.transform([val])[0]

# One-hot encoding options
one_hot_values = {
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ],
}

# --- TABS ---
tab1, tab2 = st.tabs(["📈 EDA Dashboard", "🧠 Churn Predictor"])

# 📈 EDA TAB
with tab1:
    st.subheader("📊 Interactive Churn Dataset EDA")
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.drop(columns=["customerID"], errors="ignore", inplace=True)

    st.write("### 🔍 Data Overview")
    st.dataframe(df.head())
    st.write(df.describe(include="all"))

    st.write("### 🧼 Missing Values")
    st.dataframe(df.isna().sum().reset_index().rename(columns={0: "Missing Values", "index": "Column"}))

    st.sidebar.header("🔍 Filters")
    gender_filter = st.sidebar.multiselect("Gender", df["gender"].dropna().unique(), default=df["gender"].dropna().unique())
    contract_filter = st.sidebar.multiselect("Contract", df["Contract"].dropna().unique(), default=df["Contract"].dropna().unique())
    tenure_range = st.sidebar.slider("Tenure Range", 0, int(df["tenure"].max()), (0, int(df["tenure"].max())))

    df_filtered = df[
        (df["gender"].isin(gender_filter)) &
        (df["Contract"].isin(contract_filter)) &
        (df["tenure"].between(tenure_range[0], tenure_range[1]))
    ]

    st.write("### 📊 Univariate Distributions")
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 2.5), constrained_layout=True)
            sns.histplot(df_filtered[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"{col} Histogram", fontsize=10)
            ax.tick_params(labelsize=8)
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(5, 2.5), constrained_layout=True)
            sns.boxplot(x=df_filtered[col], ax=ax)
            ax.set_title(f"{col} Boxplot", fontsize=10)
            ax.tick_params(labelsize=8)
            st.pyplot(fig)

    st.write("### 🔄 Churn vs Categorical Feature")
    cat_feature = st.selectbox("Select a categorical variable", df.select_dtypes(include="object").columns)
    fig, ax = plt.subplots(figsize=(5, 2.5), constrained_layout=True)
    sns.countplot(data=df_filtered, x=cat_feature, hue="Churn", ax=ax)
    ax.set_title(f"Churn by {cat_feature}", fontsize=10)
    ax.tick_params(labelsize=8)
    plt.xticks(rotation=45, fontsize=7)
    st.pyplot(fig)


with tab2:
    st.subheader("🧠 Churn Prediction")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure", 0, 72, 12)
    phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
    multiplelines = st.selectbox("Multiple Lines", one_hot_values["MultipleLines"])
    internet = st.selectbox("Internet Service", one_hot_values["InternetService"])
    onlinesecurity = st.selectbox("Online Security", one_hot_values["OnlineSecurity"])
    onlinebackup = st.selectbox("Online Backup", one_hot_values["OnlineBackup"])
    device = st.selectbox("Device Protection", one_hot_values["DeviceProtection"])
    tech = st.selectbox("Tech Support", one_hot_values["TechSupport"])
    tv = st.selectbox("Streaming TV", one_hot_values["StreamingTV"])
    movies = st.selectbox("Streaming Movies", one_hot_values["StreamingMovies"])
    contract = st.selectbox("Contract", one_hot_values["Contract"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", one_hot_values["PaymentMethod"])
    monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=800.0)

    if st.button("🔍 Predict Churn"):
        base = {
            "gender": encode_label("gender", gender),
            "SeniorCitizen": encode_label("SeniorCitizen", senior),
            "Partner": encode_label("Partner", partner),
            "Dependents": encode_label("Dependents", dependents),
            "tenure": tenure,
            "PhoneService": encode_label("PhoneService", phoneservice),
            "PaperlessBilling": encode_label("PaperlessBilling", paperless),
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }
        df_base = pd.DataFrame([base])

        onehot_input = {
            "MultipleLines": multiplelines,
            "InternetService": internet,
            "OnlineSecurity": onlinesecurity,
            "OnlineBackup": onlinebackup,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaymentMethod": payment
        }
        df_onehot = pd.DataFrame([onehot_input])
        df_onehot = pd.get_dummies(df_onehot)

        for col, options in one_hot_values.items():
            for opt in options:
                colname = f"{col}_{opt}"
                if colname not in df_onehot.columns:
                    df_onehot[colname] = 0

        df = pd.concat([df_base, df_onehot], axis=1)
        df["has_both_streaming"] = df.get("StreamingMovies_Yes", 0) & df.get("StreamingTV_Yes", 0)
        df["senior_married_with_dependents"] = df["SeniorCitizen"] & df["Partner"] & df["Dependents"]
        df["discount_factor"] = df["MonthlyCharges"] * df["tenure"] - df["TotalCharges"]

        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]

        st.write("📦 Model Input Preview")
        st.dataframe(df)

        prob = model.predict_proba(df)[0][1]
        st.metric("Churn Probability", f"{prob:.2%}")
        if prob > 0.5:
            st.error("⚠️ High risk of churn")
        else:
            st.success("✅ Low churn risk")

        #display SHAP Beeswarm plot of probability more than 0.5
        # display SHAP bar plot if churn probability is greater than 0.5
        if prob > 0.5:
            # Calculate SHAP values for the input DataFrame as before.
            explainer = shap.Explainer(model)
            shap_values = explainer(df)
            local_shap = shap_values[0]  # Selecting the first prediction's SHAP values
            
            # Create a DataFrame that maps features to their SHAP values
            shap_df = pd.DataFrame({
                'Feature': df.columns,
                'SHAP_Value': local_shap.values
            })
            # Sort by SHAP value descending (i.e. those features pushing churn higher)
            shap_df_sorted = shap_df.sort_values(by='SHAP_Value', ascending=False)
            
            st.write("### Feature Impact Ranking")
            st.dataframe(shap_df_sorted)
            # Display the SHAP bar plot for the top contributing features.
            fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
            # Render the bar plot without immediately showing it so we can capture and show via Streamlit.
            shap.plots.bar(local_shap, max_display=10, show=False, ax=ax)
            ax.set_title("Local Feature Importance", fontsize=10)
            plt.setp(ax.get_xticklabels(), fontsize=8)
            plt.setp(ax.get_yticklabels(), fontsize=8)
            st.pyplot(fig)
    
            # Generate suggestions based on feature contributions
            suggestions = []
            for _, row in shap_df_sorted.iterrows():
                feature = row['Feature']
                shap_val = row['SHAP_Value']
                
                # Only consider features that contribute positively to churn risk
                if shap_val > 0:
                    if "MonthlyCharges" in feature:
                        suggestions.append("• High monthly charges are a major churn driver. Consider offering discounts, flexible billing, or bundled promotion plans.")
                    elif "tenure" in feature:
                        suggestions.append("• A low tenure contribution indicates that customer loyalty might be low. Introduce loyalty programs or retention incentives.")
                    elif "discount_factor" in feature:
                        suggestions.append("• The discount factor suggests misalignment between spending and usage. Evaluate pricing strategies or personalized offers.")
                    elif "Streaming" in feature:
                        suggestions.append("• Streaming service features are impacting churn. Tailor bundled streaming packages to provide more value.")
                    elif "Contract" in feature:
                        suggestions.append("• The contract type is influencing churn. Consider offering flexible contract terms or incentives for longer commitments.")
                    elif any(val in feature for val in ["TechSupport_No", "Online_Security_No", "Online_Backup_No"]):
                        suggestions.append("• Additional Services is a significant churn driver. Provide these services to enhance customer satisfaction.")
            st.write("### Suggested Actions to Reduce Churn")
            #remove duplicates and empty suggestions
            suggestions = list(set(suggestions))
            suggestions = [s for s in suggestions if s.strip() != ""]
            if suggestions:
                for s in suggestions:
                    st.write(s)
            else:
                st.write("No actionable suggestions based on the current SHAP values.")
                

