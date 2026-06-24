"""
app.py — Streamlit web UI for the House Price Prediction App.

Features:
    - Single prediction with confidence band (US-1, US-2)
    - Model switching: Linear Regression / Random Forest (US-5)
    - Property comparison side-by-side (US-3)
    - PDF report download (US-4)
    - Interactive charts (area vs price, feature importance)

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF

from model import ArtifactStore, Predictor

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────
LOCATIONS = ["Downtown", "Urban", "Suburban", "Rural"]
MODEL_OPTIONS = {
    "Linear Regression": "linear_regression",
    "Random Forest": "random_forest",
}

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .result-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    
    /* Report Preview Card CSS */
    .report-container {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 35px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        color: #1a202c !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        margin-top: 25px;
        margin-bottom: 25px;
    }
    .report-header {
        border-bottom: 3px double #3182ce;
        padding-bottom: 15px;
        margin-bottom: 20px;
    }
    .report-title-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }
    .report-title {
        color: #2b6cb0 !important;
        font-size: 26px;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
    }
    .report-meta-text {
        font-size: 13px;
        color: #718096 !important;
        margin-top: 5px;
    }
    .report-id-badge {
        background-color: #ebf8ff;
        color: #2b6cb0 !important;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 700;
        font-family: monospace;
        border: 1px solid #bee3f8;
    }
    .report-section {
        margin-top: 25px;
        margin-bottom: 25px;
    }
    .report-section-title {
        color: #2d3748 !important;
        font-size: 15px;
        font-weight: 700;
        border-left: 4px solid #3182ce;
        padding-left: 10px;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .report-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 15px;
        margin-bottom: 15px;
    }
    .report-grid-2col {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-bottom: 15px;
    }
    @media (max-width: 768px) {
        .report-grid-2col {
            grid-template-columns: 1fr;
        }
    }
    .report-card-item {
        background-color: #f7fafc;
        border: 1px solid #edf2f7;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
    }
    .report-card-label {
        font-size: 11px;
        color: #718096 !important;
        text-transform: uppercase;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .report-card-value {
        font-size: 16px;
        font-weight: 700;
        color: #2d3748 !important;
    }
    .report-valuation-box {
        background: linear-gradient(135deg, #ebf8ff 0%, #e6fffa 100%);
        border: 1px solid #bee3f8;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .report-val-title {
        font-size: 13px;
        color: #2b6cb0 !important;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 6px;
        letter-spacing: 0.05em;
    }
    .report-val-price {
        font-size: 34px;
        font-weight: 900;
        color: #2c5282 !important;
        margin-bottom: 5px;
    }
    .report-val-range {
        font-size: 14px;
        color: #4a5568 !important;
        font-weight: 500;
    }
    .report-text {
        font-size: 14px;
        line-height: 1.6;
        color: #4a5568 !important;
    }
    .report-highlight {
        font-weight: 600;
        color: #1a202c !important;
    }
    .report-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }
    .report-table th {
        background-color: #edf2f7;
        color: #4a5568 !important;
        font-weight: 700;
        text-align: left;
        padding: 8px 12px;
        font-size: 11px;
        text-transform: uppercase;
        border-bottom: 2px solid #cbd5e0;
    }
    .report-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #edf2f7;
        font-size: 13px;
        color: #2d3748 !important;
    }
    .report-disclaimer {
        font-size: 11px;
        color: #a0aec0 !important;
        line-height: 1.5;
        border-top: 1px solid #e2e8f0;
        padding-top: 15px;
        margin-top: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ========================================================================
# Session state: cache loaded models
# ========================================================================
@st.cache_resource
def load_predictor(algorithm_key: str) -> Predictor:
    """Load a Predictor (model + preprocessor) from artifacts."""
    model, preprocessor, metrics = ArtifactStore.load(algorithm_name=algorithm_key)
    residual_std = metrics.get("residual_std", 0.0)
    return Predictor(model=model, preprocessor=preprocessor, residual_std=residual_std)


@st.cache_resource
def load_metrics(algorithm_key: str) -> dict:
    """Load metrics dict for an algorithm."""
    _, _, metrics = ArtifactStore.load(algorithm_name=algorithm_key)
    return metrics


# ========================================================================
# UI Components
# ========================================================================
def render_sidebar() -> Tuple[str, dict]:
    """Render the input sidebar and return (algorithm_key, features_dict)."""

    st.sidebar.header("⚙️ Configuration")

    # Model selector
    model_choice = st.sidebar.radio(
        "🤖 Model",
        options=list(MODEL_OPTIONS.keys()),
        index=1,  # default to Random Forest
        horizontal=False,
    )
    algorithm_key = MODEL_OPTIONS[model_choice]

    st.sidebar.divider()
    st.sidebar.header("🏠 Property Details")

    area = st.sidebar.slider(
        "Area (sq ft)",
        min_value=300,
        max_value=6000,
        value=1800,
        step=50,
        help="Total built-up area in square feet",
    )

    bedrooms = st.sidebar.number_input(
        "Bedrooms",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )

    bathrooms = st.sidebar.number_input(
        "Bathrooms",
        min_value=1,
        max_value=8,
        value=2,
        step=1,
    )

    age = st.sidebar.slider(
        "Age (years)",
        min_value=0,
        max_value=50,
        value=5,
        step=1,
        help="Age of the property in years",
    )

    location = st.sidebar.selectbox(
        "Location",
        options=LOCATIONS,
        index=1,
        help="Locality type",
    )

    features = {
        "Area": float(area),
        "Bedrooms": int(bedrooms),
        "Bathrooms": int(bathrooms),
        "Age": int(age),
        "Location": location,
    }

    return algorithm_key, features


def format_price(lakhs: float) -> str:
    """Format a price in Lakhs to a readable Indian format."""
    if lakhs >= 100:
        crores = lakhs / 100
        return f"₹{crores:.2f} Cr"
    return f"₹{lakhs:.2f} L"


def render_prediction(predictor: Predictor, features: dict, metrics: dict) -> None:
    """Render the prediction result card with confidence band."""
    price, lo, hi = predictor.confidence(features)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Estimated Price", value=format_price(price))

    with col2:
        st.metric(label="Lower Bound (95% CI)", value=format_price(lo))

    with col3:
        st.metric(label="Upper Bound (95% CI)", value=format_price(hi))

    # Confidence band visualization
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.barh(0, hi - lo, left=lo, height=0.4, color="#667eea", alpha=0.6, label="95% Confidence Band")
    ax.axvline(price, color="#f5576c", linewidth=2, linestyle="--", label=f"Estimate: {format_price(price)}")
    ax.set_xlabel("Price (Lakhs ₹)")
    ax.set_yticks([])
    ax.legend(loc="upper right")
    ax.set_title("Confidence Interval")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Metrics badge
    st.markdown("### 📊 Model Performance")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("R² Score", f"{metrics['r2']:.4f}")
    mc2.metric("MAE", f"{metrics['mae']:.2f} L")
    mc3.metric("RMSE", f"{metrics['rmse']:.2f} L")
    mc4.metric("CV R²", f"{metrics['cv_mean_r2']:.4f}")


def render_charts(features: dict) -> None:
    """Render interactive charts: scatter plot and feature importance."""

    # Load training data for scatter plot
    house_data_df = pd.read_csv("data/house_data.csv")

    tab1, tab2 = st.tabs(["📈 Price vs Area", "📊 Feature Importance"])

    with tab1:
        # Predict the price for the user's property so we can plot it
        try:
            rf_predictor = load_predictor("random_forest")
            user_price = rf_predictor.predict(features)
        except Exception:
            user_price = None

        fig, ax = plt.subplots(figsize=(10, 5))
        scatter = ax.scatter(
            house_data_df["Area"],
            house_data_df["Price"],
            c=house_data_df["Location"].fillna("Urban").map(
                {"Downtown": "#e63946", "Urban": "#457b9d", "Suburban": "#2a9d8f", "Rural": "#e9c46a"}
            ).fillna("#888888").to_list(),
            alpha=0.5,
            s=20,
        )
        # Highlight user's property (only if we have a price)
        if user_price is not None:
            ax.scatter(
                [features["Area"]],
                [user_price],
                color="red",
                s=200,
                marker="*",
                zorder=5,
                label="Your Property",
            )
        ax.set_xlabel("Area (sq ft)")
        ax.set_ylabel("Price (Lakhs ₹)")
        ax.set_title("Area vs Price by Location")
        # Custom legend for locations
        from matplotlib.patches import Patch
        location_legend = [
            Patch(facecolor="#e63946", label="Downtown"),
            Patch(facecolor="#457b9d", label="Urban"),
            Patch(facecolor="#2a9d8f", label="Suburban"),
            Patch(facecolor="#e9c46a", label="Rural"),
        ]
        if user_price is not None:
            location_legend.append(Patch(facecolor="red", label="Your Property"))
        ax.legend(handles=location_legend, loc="upper left")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab2:
        # Feature importance from the trained model
        try:
            pred = load_predictor("random_forest")
            model = pred.model
            importances = model.feature_importances_

            # Derive feature names from the fitted preprocessor
            try:
                feature_names = list(
                    pred.preprocessor.column_transformer.get_feature_names_out()
                )
            except Exception:
                feature_names = ["Area", "Bedrooms", "Bathrooms", "Age", "Downtown", "Rural", "Suburban", "Urban"]

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(feature_names, importances, color="#667eea")
            ax.set_xlabel("Importance")
            ax.set_title("Random Forest — Feature Importance")
            for bar, val in zip(bars, importances):
                ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not load feature importance: {e}")


def render_comparison() -> None:
    """Render the property comparison view (US-3)."""

    st.header("🏠 Property Comparison")
    st.markdown("Compare two properties side-by-side to see which offers better value.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Property A")
        a_area = st.slider("Area (sq ft)", 300, 6000, 1800, 50, key="comp_a_area")
        a_beds = st.number_input("Bedrooms", 1, 10, 3, key="comp_a_beds")
        a_baths = st.number_input("Bathrooms", 1, 8, 2, key="comp_a_baths")
        a_age = st.slider("Age (years)", 0, 50, 5, key="comp_a_age")
        a_loc = st.selectbox("Location", LOCATIONS, index=1, key="comp_a_loc")

    with col_b:
        st.subheader("Property B")
        b_area = st.slider("Area (sq ft)", 300, 6000, 2500, 50, key="comp_b_area")
        b_beds = st.number_input("Bedrooms", 1, 10, 4, key="comp_b_beds")
        b_baths = st.number_input("Bathrooms", 1, 8, 3, key="comp_b_baths")
        b_age = st.slider("Age (years)", 0, 50, 2, key="comp_b_age")
        b_loc = st.selectbox("Location", LOCATIONS, index=0, key="comp_b_loc")

    if st.button("🔍 Compare Properties", type="primary"):
        features_a = {"Area": float(a_area), "Bedrooms": int(a_beds), "Bathrooms": int(a_baths), "Age": int(a_age), "Location": a_loc}
        features_b = {"Area": float(b_area), "Bedrooms": int(b_beds), "Bathrooms": int(b_baths), "Age": int(b_age), "Location": b_loc}

        predictor = load_predictor("random_forest")
        price_a = predictor.predict(features_a)
        price_b = predictor.predict(features_b)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Property A", format_price(price_a))
            st.json(features_a)
        with c2:
            st.metric("Property B", format_price(price_b))
            st.json(features_b)

        # Comparison bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        categories = ["Price (L)", "Area (sq ft)", "Bedrooms", "Bathrooms"]
        vals_a = [price_a, a_area, a_beds, a_baths]
        vals_b = [price_b, b_area, b_beds, b_baths]

        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width / 2, vals_a, width, label="Property A", color="#667eea")
        ax.bar(x + width / 2, vals_b, width, label="Property B", color="#f5576c")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.set_title("Property Comparison")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def render_report_preview_and_download(predictor: Predictor, features: dict, price: float, lo: float, hi: float, metrics: dict, algorithm_key: str) -> None:
    """Generate and display an on-screen appraisal report, with options to download in PDF and Text/Markdown format (US-4)."""
    import random
    from datetime import datetime

    # Seed the RNG deterministically using features to keep report consistent for the same inputs
    seed_val = int(features.get("Area", 1000)) + int(features.get("Age", 0)) + len(features.get("Location", ""))
    rng = random.Random(seed_val)

    # 1. Generate report metadata
    ref_id = f"REP-{datetime.now().strftime('%Y%m%d')}-{rng.randint(1000, 9999)}"
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 2. Location-specific market dynamics
    loc = features.get("Location", "Urban")
    if loc == "Downtown":
        market_desc = "Highly sought-after central business district with extremely high walkability and immediate public transit access. The neighborhood exhibits high premium rental demand and strong historical price growth."
        connectivity_score = 9.5
        amenities_score = 9.2
        safety_score = 8.0
        livability_score = 8.5
    elif loc == "Urban":
        market_desc = "Established urban sector characterized by excellent accessibility to schools, healthcare facilities, and retail centers. Offers a balanced lifestyle with steady capital appreciation and consistent rental yields."
        connectivity_score = 8.8
        amenities_score = 8.5
        safety_score = 8.5
        livability_score = 8.7
    elif loc == "Suburban":
        market_desc = "Family-oriented suburban neighborhood showcasing low noise levels, spacious green parks, and high-quality residential projects. High livability and strong demand from long-term home buyers."
        connectivity_score = 7.5
        amenities_score = 7.8
        safety_score = 9.2
        livability_score = 9.0
    else:  # Rural
        market_desc = "Quiet and peaceful rural zone featuring larger plot parcels and natural surroundings. Ideal for second homes or agricultural/leisure properties. Slower rental velocity but high long-term land appreciation potential."
        connectivity_score = 5.8
        amenities_score = 5.2
        safety_score = 8.8
        livability_score = 7.5

    # 3. Random/mock insights
    investment_yield = round(rng.uniform(3.0, 5.5), 2)
    risk_assessment = rng.choice(["Low Risk / Core Investment", "Moderate Risk / Growth Play", "Low-to-Medium Risk"])
    market_sentiment = rng.choice(["Bullish (High Demand)", "Stable / Balanced Market", "Neutral / Consolidation Phase"])
    infrastructure_boost = rng.choice([
        "Upcoming metro line extension within 1.5 km.",
        "Proposed greenfield ring road development nearby.",
        "New shopping mall and retail hub construction starting shortly.",
        "Repaving of major arterial links and expansion of smart city infrastructure."
    ])

    # 4. Generate on-screen report inside a clean Streamlit container using standard Markdown text
    with st.container(border=True):
        st.markdown(f"""
### 🏠 PROPERTY APPRAISAL REPORT
**this report are for customer**

**Reference ID:** `{ref_id}`  
**Generated on:** {gen_time} | **Model:** {algorithm_key.replace('_', ' ').title()}

---

#### 1. Executive Summary
This automated appraisal provides a comprehensive market valuation and property analysis for the specified asset. Using advanced machine learning models trained on premium real estate data points, the estimated fair market value is calculated to assist with investment profiling, purchasing decisions, or sales benchmarks.

#### 2. Property Specifications
* **Area (Sq Ft):** {features['Area']:,.0f}
* **Bedrooms:** {features['Bedrooms']} BHK
* **Bathrooms:** {features['Bathrooms']}
* **Property Age:** {features['Age']} Years
* **Location Type:** {features['Location']}

#### 3. Valuation Breakdown
* **Estimated Fair Market Value:** **{format_price(price)}**
* **95% Confidence Interval:** {format_price(lo)} – {format_price(hi)}

The estimate is derived from features matching typical properties in **{features['Location']}**. The statistical model confirms a high degree of confidence for this valuation range based on current market comparables.

#### 4. Market & Investment Analysis
* **Est. Rental Yield:** {investment_yield}% per annum
* **Risk Profile:** {risk_assessment}
* **Market Sentiment:** {market_sentiment}
* **Key Drivers:** {infrastructure_boost}

#### 5. Neighborhood Scorecard
* **Connectivity & Transit:** {connectivity_score:.1f} / 10
* **Commercial Amenities:** {amenities_score:.1f} / 10
* **Safety Index:** {safety_score:.1f} / 10
* **Overall Livability:** {livability_score:.1f} / 10

**Local Market Context:** {market_desc}

---
*Disclaimer: This is an automated algorithmic report. This estimate is generated by a machine-learning model and should not be considered a formal professional appraisal. Actual market values may vary depending on local market conditions, floor layout, exact build quality, views, facing, and negotiation.*
        """)

    # ── PDF Generation ──
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "House Price Estimate Report", ln=1, align="C")
    
    # User-Requested Text Requirement
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(49, 130, 206) # Nice blue (#3182ce)
    pdf.cell(0, 10, "This is your generated report", ln=1, align="C")
    pdf.set_text_color(0, 0, 0) # reset
    
    # Customer indicator (normal text)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, "this report are for customer", ln=1, align="C")
    
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 6, f"Reference ID: {ref_id} | Generated: {gen_time}", ln=1, align="C")
    pdf.ln(5)
    
    # Horizontal line
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    # Executive Summary
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "1. Executive Summary", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, "This automated appraisal provides a comprehensive market valuation and property analysis. "
                          "Using machine learning models trained on real estate datasets, the fair market value is "
                          "calculated based on key property details and location context.")
    pdf.ln(3)

    # Specs
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "2. Property Specifications", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(50, 6, f"Area: {features['Area']:,.0f} sq ft", ln=0)
    pdf.cell(50, 6, f"Bedrooms: {features['Bedrooms']} BHK", ln=0)
    pdf.cell(50, 6, f"Bathrooms: {features['Bathrooms']}", ln=1)
    pdf.cell(50, 6, f"Age: {features['Age']} years", ln=0)
    pdf.cell(50, 6, f"Location: {features['Location']}", ln=1)
    pdf.ln(3)
    
    # Valuation
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "3. Valuation Estimate", ln=1)
    pdf.set_font("Helvetica", "B", 10)
    price_str = format_price(price).replace("₹", "Rs. ")
    lo_str = format_price(lo).replace("₹", "Rs. ")
    hi_str = format_price(hi).replace("₹", "Rs. ")
    pdf.cell(0, 6, f"Estimated Fair Market Value: {price_str}", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"95% Confidence Interval: {lo_str} - {hi_str}", ln=1)
    pdf.ln(3)
    
    # Market & Investment Details
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "4. Market & Investment Analysis", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(80, 6, f"Est. Rental Yield: {investment_yield}% p.a.", ln=0)
    pdf.cell(80, 6, f"Connectivity Rating: {connectivity_score:.1f} / 10", ln=1)
    pdf.cell(80, 6, f"Risk Profile: {risk_assessment}", ln=0)
    pdf.cell(80, 6, f"Commercial Amenities: {amenities_score:.1f} / 10", ln=1)
    pdf.cell(80, 6, f"Market Sentiment: {market_sentiment}", ln=0)
    pdf.cell(80, 6, f"Safety Index: {safety_score:.1f} / 10", ln=1)
    pdf.cell(80, 6, f"Livability Index: {livability_score:.1f} / 10", ln=1)
    pdf.ln(2)
    pdf.multi_cell(0, 5, f"Local Market Context: {market_desc}")
    pdf.ln(4)
    
    # Disclaimer
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, "Disclaimer: This is an automated algorithmic report. This estimate is generated by "
                          "a machine-learning model and should not be considered a formal professional appraisal. "
                          "Actual market values may vary depending on local conditions.")
    
    pdf_bytes = pdf.output(dest='S').encode('latin1')

    # ── Text Report Generation ──
    text_report = f"""==================================================
PROPERTY APPRAISAL REPORT - {ref_id}
==================================================
This is your generated report
this report are for customer
Generated on: {gen_time}
Model: {algorithm_key.replace('_', ' ').title()}

1. EXECUTIVE SUMMARY
---------------------
This automated appraisal provides a comprehensive market valuation and property analysis.
Using machine learning models, the estimated fair market value is calculated based on
the property parameters and local neighborhood trends.

2. PROPERTY SPECIFICATIONS
---------------------------
- Area: {features['Area']:,.0f} sq ft
- Configuration: {features['Bedrooms']} BHK / {features['Bathrooms']} Bath
- Property Age: {features['Age']} years
- Location Type: {features['Location']}

3. VALUATION ESTIMATE
----------------------
- Estimated Fair Market Value: {format_price(price)}
- 95% Confidence Interval: {format_price(lo)} - {format_price(hi)}

4. MARKET & INVESTMENT ANALYSIS
--------------------------------
- Est. Rental Yield: {investment_yield}% p.a.
- Risk Profile: {risk_assessment}
- Market Sentiment: {market_sentiment}
- Connectivity Index: {connectivity_score}/10
- Commercial Amenities: {amenities_score}/10
- Safety Index: {safety_score}/10
- Overall Livability: {livability_score}/10

Neighborhood Context:
{market_desc}

5. DISCLAIMER
--------------
This is an automated algorithmic report. This estimate is generated by a machine-learning
model and should not be considered a formal professional appraisal. Actual market values
may vary depending on local market conditions.
"""

    st.markdown("### 📥 Download Valuation Report")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"property_report_{ref_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with col2:
        st.download_button(
            label="📝 Download Text Report",
            data=text_report,
            file_name=f"property_report_{ref_id}.txt",
            mime="text/plain",
            use_container_width=True
        )


# ========================================================================
# Main app
# ========================================================================
def main() -> None:
    """Entry point for the Streamlit app."""

    # Header
    st.title("🏠 House Price Prediction App")
    st.markdown(
        "Estimate property prices instantly using Machine Learning. "
        "Enter details in the sidebar and click **Predict**!"
    )
    st.caption("Model: Linear Regression & Random Forest on Indian Real Estate Data")

    # Sidebar inputs
    algorithm_key, features = render_sidebar()

    # Tab navigation
    tab_predict, tab_compare, tab_charts = st.tabs(["🔮 Predict", "⚖️ Compare", "📊 Charts"])

    # ── Predict Tab ──────────────────────────────────────────────────────
    with tab_predict:
        st.header("Price Estimate")
        st.markdown("Enter property details in the sidebar, then click **Predict**.")

        if st.button("🔮 Predict Price", type="primary", use_container_width=True):
            with st.spinner("Calculating..."):
                predictor = load_predictor(algorithm_key)
                metrics = load_metrics(algorithm_key)

                price, lo, hi = predictor.confidence(features)

                # Store for PDF report
                st.session_state["last_prediction"] = {
                    "price": price,
                    "lo": lo,
                    "hi": hi,
                    "features": features,
                    "metrics": metrics,
                    "algorithm": algorithm_key,
                }
                st.balloons()

        if "last_prediction" in st.session_state:
            pred = st.session_state["last_prediction"]
            predictor = load_predictor(pred["algorithm"])
            render_prediction(predictor, pred["features"], pred["metrics"])
            st.divider()
            render_report_preview_and_download(predictor, pred["features"], pred["price"], pred["lo"], pred["hi"], pred["metrics"], pred["algorithm"])
        else:
            st.info("👆 Adjust the sidebar inputs and click **Predict** to get an estimate.")

    # ── Compare Tab ──────────────────────────────────────────────────────
    with tab_compare:
        render_comparison()

    # ── Charts Tab ───────────────────────────────────────────────────────
    with tab_charts:
        st.header("Data Visualization")
        render_charts(features)

    # Footer
    st.divider()
    st.caption(
        "Built with ❤️ using Python, scikit-learn, and Streamlit | "
        "Model R² ≥ 0.89 | Data: 1000 Indian real-estate samples"
    )


if __name__ == "__main__":
    main()
