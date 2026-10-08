import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.data_preprocessing import clean_columns, engineer_features, load_raw_data
from src.predict import load_artifacts, predict_single, top_factors

st.set_page_config(page_title="Manufacturing Failure Prediction Dashboard", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_data
def get_raw_data():
    df = load_raw_data()
    df = clean_columns(df)
    df = engineer_features(df)
    return df


model, metrics = get_artifacts()
raw_df = get_raw_data()

FEATURE_LABELS = {
    "air_temp_k": "Air Temperature (K)",
    "process_temp_k": "Process Temperature (K)",
    "rotational_speed_rpm": "Rotational Speed (rpm)",
    "torque_nm": "Torque (Nm)",
    "tool_wear_min": "Tool Wear (min)",
    "temp_diff_k": "Temperature Difference (K)",
    "power_w": "Mechanical Power (W)",
    "type_H": "Type: High",
    "type_L": "Type: Low",
    "type_M": "Type: Medium",
}
RISK_COLORS = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}

st.sidebar.title("Manufacturing Failure Prediction")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Data Analysis", "Model Performance", "Failure Prediction", "Business Insights"],
)


def page_overview():
    st.title("Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset size", f"{metrics['dataset_size']:,} machines")
    col2.metric("Failure rate", f"{metrics['failure_rate']*100:.2f}%")
    col3.metric("Model", metrics["chosen_model"].replace("_", " ").title())

    col4, col5, col6, col7 = st.columns(4)
    tm = metrics["test_metrics"]
    col4.metric("Recall", f"{tm['recall']*100:.1f}%")
    col5.metric("Precision", f"{tm['precision']*100:.1f}%")
    col6.metric("F1", f"{tm['f1']*100:.1f}%")
    col7.metric("ROC-AUC", f"{tm['roc_auc']*100:.1f}%")

    st.markdown(
        """
        This dashboard predicts whether a manufacturing machine is likely to fail, using
        operating sensor data (temperature, rotational speed, torque, tool wear) from the
        **AI4I 2020 Predictive Maintenance Dataset**. It is a portfolio project demonstrating
        an end-to-end classification workflow, not a production monitoring system.
        """
    )


def page_data_analysis():
    st.title("Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Failure Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="machine_failure", data=raw_df, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Failure", "Failure"])
        ax.set_ylabel("Count")
        ax.set_xlabel("")
        st.pyplot(fig)

    with col2:
        st.subheader("Failures by Machine Type")
        fig, ax = plt.subplots()
        sns.countplot(x="type", hue="machine_failure", data=raw_df, ax=ax)
        ax.set_xlabel("Machine Type")
        st.pyplot(fig)

    st.subheader("Feature Distributions by Failure Status")
    numeric_cols = ["air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]
    selected = st.selectbox("Select a feature", numeric_cols, format_func=lambda c: FEATURE_LABELS.get(c, c))
    fig, ax = plt.subplots()
    sns.histplot(data=raw_df, x=selected, hue="machine_failure", kde=True, ax=ax, stat="density", common_norm=False)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    corr_cols = numeric_cols + ["temp_diff_k", "power_w", "machine_failure"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(raw_df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)


def page_model_performance():
    st.title("Model Performance")
    st.caption(
        f"Final model: **{metrics['chosen_model'].replace('_', ' ').title()}**, "
        f"evaluated once on a held-out test set."
    )

    tm = metrics["test_metrics"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{tm['accuracy']*100:.1f}%")
    col2.metric("Precision", f"{tm['precision']*100:.1f}%")
    col3.metric("Recall", f"{tm['recall']*100:.1f}%")
    col4.metric("F1", f"{tm['f1']*100:.1f}%")
    col5.metric("ROC-AUC", f"{tm['roc_auc']*100:.1f}%")

    cm = metrics["test_confusion_matrix"]
    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.subheader("Confusion Matrix (Test Set)")
        matrix = np.array(
            [
                [cm["true_negative"], cm["false_positive"]],
                [cm["false_negative"], cm["true_positive"]],
            ]
        )
        fig, ax = plt.subplots()
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Pred No Failure", "Pred Failure"],
            yticklabels=["Actual No Failure", "Actual Failure"],
            ax=ax,
        )
        st.pyplot(fig)

    with col_roc:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots()
        ax.plot(metrics["roc_curve"]["fpr"], metrics["roc_curve"]["tpr"], label=f"ROC-AUC = {tm['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

    st.info(
        f"We prioritize **recall** over raw accuracy: missing a real failure (a false negative) "
        f"risks unplanned downtime, while a false alarm (false positive) only costs an unnecessary "
        f"inspection. Of {cm['false_negative'] + cm['true_positive']} real failures in the test set, "
        f"the model caught {cm['true_positive']} and missed {cm['false_negative']}."
    )


def page_failure_prediction():
    st.title("Failure Prediction")
    st.write("Enter machine operating conditions to estimate failure risk.")

    col1, col2, col3 = st.columns(3)
    with col1:
        air_temp = st.slider("Air Temperature (K)", 295.0, 305.0, 300.0, 0.1)
        process_temp = st.slider("Process Temperature (K)", 305.0, 314.0, 310.0, 0.1)
    with col2:
        rpm = st.slider("Rotational Speed (rpm)", 1150, 2900, 1550, 10)
        torque = st.slider("Torque (Nm)", 3.0, 77.0, 40.0, 0.5)
    with col3:
        tool_wear = st.slider("Tool Wear (min)", 0, 260, 108, 1)
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"])

    if st.button("Predict Failure Risk", type="primary"):
        raw_inputs = dict(
            air_temp_k=air_temp,
            process_temp_k=process_temp,
            rotational_speed_rpm=rpm,
            torque_nm=torque,
            tool_wear_min=tool_wear,
            machine_type=machine_type,
        )
        result = predict_single(model, **raw_inputs)
        factors = top_factors(model, raw_inputs)

        prob_pct = result["probability"] * 100
        risk = result["risk_category"]

        st.markdown(f"### {RISK_COLORS[risk]} {risk} RISK — {prob_pct:.1f}% failure probability")
        if risk == "LOW":
            st.success("This machine's operating conditions look typical for normal operation.")
        elif risk == "MEDIUM":
            st.warning("This machine shows some elevated risk factors — consider monitoring.")
        else:
            st.error("This machine shows a high failure probability — consider inspection.")

        st.subheader("Main Contributing Factors")
        factor_df = pd.DataFrame(factors, columns=["feature", "impact"])
        factor_df["feature"] = factor_df["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
        factor_df["direction"] = np.where(factor_df["impact"] >= 0, "Increases risk", "Decreases risk")
        fig, ax = plt.subplots()
        colors = ["#C44E52" if v >= 0 else "#4C72B0" for v in factor_df["impact"]]
        ax.barh(factor_df["feature"], factor_df["impact"], color=colors)
        ax.set_xlabel("Impact on failure probability")
        ax.invert_yaxis()
        st.pyplot(fig)
        for _, row in factor_df.iterrows():
            st.write(f"- **{row['feature']}** — {row['direction']}")


def page_business_insights():
    st.title("Business Insights")

    st.subheader("What the Model Found")
    importance_df = pd.DataFrame(metrics["global_importance"][:6], columns=["feature", "importance"])
    importance_df["feature"] = importance_df["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
    fig, ax = plt.subplots()
    ax.barh(importance_df["feature"], importance_df["importance"], color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (average impact on prediction)")
    st.pyplot(fig)

    st.markdown(
        """
        **Patterns the model found**: tool wear, rotational speed, mechanical power, and torque are
        the strongest drivers of predicted failure — consistent with the physical failure modes in
        this dataset (overstrain and power failures tend to occur at high torque / low speed combinations,
        while tool-wear failures accumulate over time).

        **What this model can tell us**: which operating conditions are statistically associated with
        a higher failure probability in this historical dataset, and a ranked risk estimate for new
        readings that resemble it.

        **What this model cannot tell us**: it cannot prove *why* a machine will fail mechanically, it
        has not seen real plant data or sensor drift, and it should not be the sole basis for a maintenance
        decision — it is a triage/prioritization signal, not a diagnosis.

        **Preventive-maintenance use case**: flag HIGH-risk machines for inspection before failure,
        prioritize maintenance scheduling by risk score rather than fixed time intervals, and use the
        top contributing factors to tell technicians what to check first.
        """
    )


PAGES = {
    "Overview": page_overview,
    "Data Analysis": page_data_analysis,
    "Model Performance": page_model_performance,
    "Failure Prediction": page_failure_prediction,
    "Business Insights": page_business_insights,
}

PAGES[page]()
