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
            c=house_data_df["Location"].map(
                {"Downtown": "#e63946", "Urban": "#457b9d", "Suburban": "#2a9d8f", "Rural": "#e9c46a"}
            ),
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


def render_pdf_report(predictor: Predictor, features: dict, price: float, lo: float, hi: float, metrics: dict) -> None:
    """Generate and offer a PDF report for download (US-4)."""

    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "House Price Estimate Report", new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)

    # Property details
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Property Details", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for key, val in features.items():
        pdf.cell(0, 8, f"  {key}: {val}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Price estimate
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Price Estimate", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"  Estimated Price: {format_price(price)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"  Lower Bound (95% CI): {format_price(lo)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"  Upper Bound (95% CI): {format_price(hi)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Model metrics
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Model Performance", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"  R2 Score: {metrics['r2']:.4f}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"  MAE: {metrics['mae']:.2f} Lakh", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"  RMSE: {metrics['rmse']:.2f} Lakh", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # Disclaimer
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0,
        5,
        "Disclaimer: This estimate is generated by a machine-learning model and should not be "
        "considered a professional appraisal. Actual prices may vary based on factors not captured "
        "in this model (e.g., floor, facing, amenities, market conditions).",
    )

    pdf_bytes = pdf.output()
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name=f"price_estimate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
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
                render_prediction(predictor, features, metrics)

                st.divider()
                st.subheader("📄 Export Report")
                render_pdf_report(predictor, features, price, lo, hi, metrics)
        else:
            # Show placeholder if no prediction yet
            st.info("👆 Adjust the sidebar inputs and click **Predict** to get an estimate.")

            # Show cached result if available
            if "last_prediction" in st.session_state:
                pred = st.session_state["last_prediction"]
                st.subheader("Last Prediction")
                st.metric("Estimated Price", format_price(pred["price"]))
                st.metric("Range", f"{format_price(pred['lo'])} — {format_price(pred['hi'])}")

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
