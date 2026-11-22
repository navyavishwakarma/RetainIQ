
import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import csv
from io import StringIO
from pathlib import Path

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="RetainIQ | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "best_gradient_boosting.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()


@st.cache_resource
def load_explainer():
    return shap.TreeExplainer(
        model.named_steps["classifier"]
    )

explainer = load_explainer()


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background-color: #F7F6F2;
    color: #24364B;
    font-family: 'Inter', sans-serif;
}

[data-testid="stMain"] {
    background-color: #F7F6F2;
}

[data-testid="stHeader"] {
    background: #F7F6F2;
}

[data-testid="stSidebar"] {
    background-color: #EAEFEA;
    border-right: 1px solid #D9E2DC;
}

[data-testid="stSidebar"] * {
    color: #24364B !important;
}

h1, h2, h3, p, label {
    color: #24364B;
}

.hero {
    background-color: #E5EFEA;
    border-left: 6px solid #438C76;
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 28px;
}

.hero h1 {
    color: #244C40 !important;
    font-size: 38px;
    font-weight: 800;
}

.hero p {
    color: #526D62 !important;
    font-size: 15px;
}

.metric-card {
    border-radius: 15px;
    padding: 22px;
    min-height: 125px;
    border: 1px solid #E0E5E0;
    background: #FFFFFF;
    box-shadow: 0 3px 12px rgba(36, 54, 75, 0.05);
}

.metric-card:nth-child(1) {
    background: #E4F1EC;
}

.metric-label {
    color: #53685E !important;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.metric-value {
    color: #24364B !important;
    font-size: 30px;
    font-weight: 800;
    margin-top: 12px;
}

.section-title {
    color: #24364B !important;
    font-size: 24px;
    font-weight: 700;
    margin-top: 32px;
    margin-bottom: 6px;
}

.section-subtitle {
    color: #687789 !important;
    font-size: 14px;
    margin-bottom: 20px;
}

.info-card {
    background: #FFFFFF;
    border: 1px solid #E3E7E2;
    border-radius: 15px;
    padding: 22px;
    min-height: 145px;
    box-shadow: 0 3px 12px rgba(36, 54, 75, 0.04);
}

.info-card h3 {
    color: #347B66 !important;
}

.info-card p {
    color: #596779 !important;
    font-size: 14px;
    line-height: 1.6;
}

div.stButton > button {
    background: #438C76;
    color: #FFFFFF !important;
    border: none;
    border-radius: 9px;
    font-weight: 600;
}

div.stButton > button:hover {
    background: #326D5B;
    color: #FFFFFF !important;
}

.stSelectbox label,
.stTextInput label,
.stNumberInput label,
.stSlider label {
    color: #24364B !important;
    font-weight: 600;
}

[data-baseweb="select"] > div {
    background-color: #FFFFFF;
    color: #24364B;
    border-color: #D5DDD8;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #788579 !important;
    font-size: 12px;
    padding: 25px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown("## 📊 RetainIQ")
    st.caption("CUSTOMER INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Overview", "Churn Prediction", "About Project"]
    )

    st.divider()

    st.markdown("### 🤖 Model Information")
    st.success("Machine Learning Powered")
    st.caption("IBM Telco Customer Churn Dataset")


# ==================================================
#                    OVERVIEW
# ==================================================

if page == "Overview":

    st.markdown("""
    <div class="hero">
        <h1>RetainIQ</h1>
        <p>Predict churn. Understand risk. Retain customers.</p>
        <p>AI-powered Customer Churn Intelligence & Retention System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Business Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">A snapshot of customer retention performance</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        (col1, "TOTAL CUSTOMERS", "7,043"),
        (col2, "CHURNED CUSTOMERS", "1,869"),
        (col3, "OVERALL CHURN RATE", "26.54%"),
        (col4, "HIGH RISK CUSTOMERS", "2,812")
    ]

    for column, label, value in metrics:

        with column:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Why RetainIQ?</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class="info-card">
            <h3>🎯 Predict</h3>
            <p>Identify customers who may be likely to leave.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="info-card">
            <h3>🔍 Understand</h3>
            <p>Explore factors influencing customer churn.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class="info-card">
            <h3>💡 Retain</h3>
            <p>Explore retention strategies for at-risk customers.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="footer">RetainIQ | Customer Churn Intelligence Project</div>',
        unsafe_allow_html=True
    )


# ==================================================
#                 CHURN PREDICTION
# ==================================================

elif page == "Churn Prediction":

    st.markdown("""
    <div class="hero">
        <h1>Customer Churn Prediction</h1>
        <p>Enter customer details to estimate their likelihood of leaving.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Customer Information")

    # ---------------- CUSTOMER INPUT FORM ----------------

    with st.form("churn_prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            partner = st.selectbox(
                "Has Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Has Dependents",
                ["Yes", "No"]
            )

            tenure = st.number_input(
                "Tenure (months)",
                0, 72, 12
            )

            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )

        with col2:

            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

        with col3:

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=840.0
            )

        predict_button = st.form_submit_button(
            "Predict Churn",
            use_container_width=True
        )


    # ---------------- PREDICTION ----------------

    if predict_button:

        # Correct dependent service selections

        if phone == "No":
            multiple_lines = "No phone service"

        if internet == "No":

            online_security = "No internet service"
            online_backup = "No internet service"
            device_protection = "No internet service"
            tech_support = "No internet service"
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"


        # ---------------- CUSTOMER DATAFRAME ----------------

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])


        # ---------------- MODEL PREDICTION ----------------

        churn_probability = model.predict_proba(customer)[0][
            list(model.classes_).index("Yes")
        ]

        prediction = model.predict(customer)[0]


        # ---------------- CUSTOMER PROFILE ----------------

        st.divider()
        st.subheader("Customer Profile Summary")

        if tenure <= 12:
            tenure_category = "New Customer"

        elif tenure <= 36:
            tenure_category = "Medium-term Customer"

        else:
            tenure_category = "Long-term Customer"

        profile_col1, profile_col2, profile_col3, profile_col4 = st.columns(4)

        with profile_col1:
            st.metric("Customer Tenure", f"{tenure} months")

        with profile_col2:
            st.metric("Customer Segment", tenure_category)

        with profile_col3:
            st.metric("Contract Type", contract)

        with profile_col4:
            st.metric("Monthly Charges", f"${monthly_charges:.2f}")

        st.write(f"**Internet Service:** {internet}")
        st.write(f"**Payment Method:** {payment}")


        # ---------------- PREDICTION RESULT ----------------

        st.divider()
        st.subheader("Prediction Result")

        st.metric(
            "Estimated Churn Probability",
            f"{churn_probability * 100:.2f}%"
        )


        # ---------------- RISK INDICATOR ----------------

        st.markdown("### Customer Risk Level")

        st.progress(float(churn_probability))

        if churn_probability >= 0.60:

            st.error(
                "HIGH RISK — Immediate attention recommended."
            )

            risk_level = "High"

        elif churn_probability >= 0.30:

            st.warning(
                "MODERATE RISK — Customer may need attention."
            )

            risk_level = "Moderate"

        else:

            st.success(
                "LOW RISK — Customer is less likely to churn."
            )

            risk_level = "Low"

        st.caption(
            "Risk bands are illustrative application thresholds, "
            "not independently validated business cut-offs."
        )


        if prediction == "Yes":

            st.error(
                "This customer is predicted to churn."
            )

        else:

            st.success(
                "This customer is predicted to stay."
            )

        st.caption(
            "This is a machine learning estimate, not a guarantee "
            "of future customer behaviour."
        )


        # ==================================================
        #             RETENTION RECOMMENDATIONS
        # ==================================================

        st.divider()
        st.subheader("Retention Recommendations")

        st.write(
            "Suggested actions based on this customer's "
            "service and billing profile."
        )

        recommendations = []

        if contract == "Month-to-month":

            recommendations.append(
                "Offer a one-year or two-year contract with "
                "a suitable loyalty benefit."
            )

        if tech_support == "No":

            recommendations.append(
                "Offer a tech support consultation or "
                "a suitable technical support package."
            )

        if online_security == "No" and internet != "No":

            recommendations.append(
                "Introduce available online security plans "
                "and explain their benefits."
            )

        if payment == "Electronic check":

            recommendations.append(
                "Encourage the customer to consider automatic "
                "bank transfer or credit card payments."
            )

        if monthly_charges > 70:

            recommendations.append(
                "Review the customer's current plan and "
                "check whether a more suitable package is available."
            )

        if tenure <= 12:

            recommendations.append(
                "Provide onboarding assistance and check whether "
                "the customer is satisfied with the service."
            )

        if recommendations:

            for i, recommendation in enumerate(recommendations, 1):

                st.markdown(
                    f"**{i}.** {recommendation}"
                )

        else:

            st.success(
                "No specific retention action was identified "
                "from the selected customer profile."
            )

        st.caption(
            "These are rule-based suggestions using customer "
            "attributes. They are not guaranteed churn-prevention actions."
        )


        # ==================================================
        #             SHAP EXPLAINABILITY
        # ==================================================

        st.divider()
        st.subheader("Why Is This Customer at Risk?")

        st.write(
            "The following features show how the trained model "
            "evaluated this individual customer's profile."
        )

        # Transform customer using the trained preprocessor

        customer_processed = model.named_steps[
            "preprocessor"
        ].transform(customer)

        # Calculate SHAP values

        customer_shap_values = explainer.shap_values(
            customer_processed
        )

        # Handle different SHAP output formats

        classes = list(model.named_steps["classifier"].classes_)
        positive_class_index = classes.index("Yes")

        if isinstance(customer_shap_values, list):
            customer_shap_values = customer_shap_values[positive_class_index]

        customer_shap_values = np.asarray(customer_shap_values)

        if customer_shap_values.ndim == 3:
            customer_shap_values = customer_shap_values[:, :, positive_class_index]

        if customer_shap_values.ndim == 2:
            customer_shap_values = customer_shap_values[0]

        customer_shap_values = customer_shap_values.reshape(-1)

        # Get transformed feature names

        feature_names = model.named_steps[
            "preprocessor"
        ].get_feature_names_out()

        # Validate feature dimensions

        if len(feature_names) != len(customer_shap_values):

            st.error(
                "SHAP feature count does not match the transformed "
                "feature count. Please check the model and explainer."
            )

            st.stop()

        # Ensure SHAP values are numeric and finite

        customer_shap_values = pd.to_numeric(
            pd.Series(customer_shap_values),
            errors="coerce"
        ).to_numpy(dtype=float)

        if not np.isfinite(customer_shap_values).all():

            st.error(
                "The SHAP calculation returned invalid values. "
                "Please check the model explanation output."
            )

            st.stop()

        # Create SHAP DataFrame

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Impact": customer_shap_values
        })

        # Calculate absolute impact

        shap_df["Absolute Impact"] = (
            shap_df["SHAP Impact"].abs()
        )

        # Select top 8 influencing features

        shap_df = shap_df.sort_values(
            "Absolute Impact",
            ascending=False
        ).head(8).copy()

        # Clean feature names

        shap_df["Feature"] = (
            shap_df["Feature"]
            .str.replace("cat__", "", regex=False)
            .str.replace("num__", "", regex=False)
            .str.replace("_", " ", regex=False)
        )

        # Determine direction of SHAP contribution

        shap_df["Direction"] = shap_df["SHAP Impact"].apply(
            lambda x: "Increases churn score" if x > 0
            else "Decreases churn score" if x < 0
            else "No contribution"
        )

    

        # Display SHAP explanation

        st.markdown("#### Top Influencing Features")

        st.dataframe(
            shap_df[
                ["Feature", "SHAP Impact", "Direction"]
            ].reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Positive SHAP values push the model's raw prediction "
            "score towards churn. Negative values push it away. "
            "These are model contributions, not percentage changes "
            "in churn probability or proof of causation."
        )


        # ==================================================
        #           CUSTOMER INTELLIGENCE CSV REPORT
        # ==================================================

        st.divider()
        st.subheader("Download Customer Intelligence Report")

        report_rows = []

        # 1. Prediction Summary

        summary = {
            "Estimated Churn Probability": (
                f"{churn_probability:.2%}"
            ),
            "Predicted Outcome": (
                "Likely to Churn"
                if prediction == "Yes"
                else "Likely to Stay"
            ),
            "Risk Level": risk_level
        }

        for field, value in summary.items():

            report_rows.append({
                "Section": "Prediction Summary",
                "Field": field,
                "Value": value
            })


        # 2. Customer Profile

        for column in customer.columns:

            report_rows.append({
                "Section": "Customer Profile",
                "Field": column,
                "Value": str(customer.iloc[0][column])
            })


        # 3. Retention Recommendations

        if recommendations:

            for i, recommendation in enumerate(
                recommendations,
                start=1
            ):

                report_rows.append({
                    "Section": "Retention Recommendations",
                    "Field": f"Recommendation {i}",
                    "Value": recommendation
                })

        else:

            report_rows.append({
                "Section": "Retention Recommendations",
                "Field": "Recommendation",
                "Value": "No specific action identified."
            })


        # 4. Individual SHAP Explanation

        for _, row in shap_df.iterrows():

            report_rows.append({
                "Section": "Why Is This Customer at Risk?",
                "Field": row["Feature"],
                "Value": (
                    f'{row["SHAP Impact"]:.6f} - '
                    f'{row["Direction"]}'
                )
            })


        # Create report DataFrame

        report_df = pd.DataFrame(report_rows)


        # Export CSV safely

        csv_buffer = StringIO()

        report_df.to_csv(
            csv_buffer,
            index=False,
            quoting=csv.QUOTE_ALL
        )

        csv_data = csv_buffer.getvalue().encode("utf-8-sig")


        # Download button

        st.download_button(
            label="Download Customer Intelligence Report",
            data=csv_data,
            file_name="retainiq_customer_intelligence_report.csv",
            mime="text/csv",
            use_container_width=True
        )


# ==================================================
#                   ABOUT PROJECT
# ==================================================

elif page == "About Project":

    st.markdown("""
    <div class="hero">
        <h1>About RetainIQ</h1>
        <p>An end-to-end customer churn intelligence project.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Technology Stack")

    st.write("""
    - Python and Pandas
    - NumPy and Scikit-learn
    - Machine Learning and SHAP
    - SQLite and SQL
    - Power BI
    - Streamlit
    """)

    st.markdown("### Project Objective")

    st.write("""
    RetainIQ aims to analyse customer behaviour, identify
    churn patterns, estimate customer churn risk and support
    data-driven retention decisions.
    """)
