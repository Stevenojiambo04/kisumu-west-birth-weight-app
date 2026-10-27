from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Kisumu West Birth Weight Analysis",
    page_icon="👶",
    layout="wide",
)

st.title("The Effect of Gravidity and Parity on Birth Weight")
st.subheader("Kisumu West Sub-County Maternity Data Analysis")

st.markdown(
    """
    This group capstone examines the relationship between gravidity,
    parity and infant birth weight using the available maternity records.
    The application provides descriptive analysis and machine-learning
    predictions for educational and analytical purposes.
    """
)

# --------------------------------------------------
# LOAD THE PROCESSED DATASET
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed_maternity_data.csv"

if not DATA_PATH.is_file():
    st.error(
        "Dataset not found. Expected file: "
        "data/processed_maternity_data.csv"
    )
    st.info(
        "Check that processed_maternity_data.csv is committed "
        "inside the data folder of the connected GitHub repository."
    )
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)
except Exception as exc:
    st.error(f"The dataset could not be read: {exc}")
    st.stop()

# --------------------------------------------------
# VALIDATE AND PREPARE DATA
# --------------------------------------------------

required_columns = [
    "Gravidae",
    "Parity_First",
    "Parity_Second",
    "Birth_weight",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(
        "The processed dataset is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.info(
        "Export df_filtered from the latest group notebook and "
        "upload the resulting CSV to the repository's data folder."
    )
    st.stop()

for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=required_columns).copy()

if df.empty:
    st.error("No usable records remain after data preparation.")
    st.stop()

# Create the low-birth-weight target using the notebook threshold.
df["Low_Birth_Weight"] = (
    df["Birth_weight"] < 2500
).astype(int)

FEATURES = [
    "Gravidae",
    "Parity_First",
    "Parity_Second",
]

# --------------------------------------------------
# TRAIN THE DECISION TREE REGRESSION MODEL
# --------------------------------------------------

X_reg = df[FEATURES]
y_reg = df["Birth_weight"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.2,
        random_state=42,
    )
)

reg_model = DecisionTreeRegressor(
    max_depth=4,
    random_state=42,
)

reg_model.fit(X_train_reg, y_train_reg)

reg_predictions = reg_model.predict(X_test_reg)

mae = mean_absolute_error(y_test_reg, reg_predictions)
rmse = np.sqrt(
    mean_squared_error(y_test_reg, reg_predictions)
)
r2 = r2_score(y_test_reg, reg_predictions)

# --------------------------------------------------
# TRAIN LOGISTIC REGRESSION CLASSIFIER
# --------------------------------------------------

classification_df = df[
    FEATURES + ["Low_Birth_Weight"]
].dropna().copy()

class_counts = classification_df["Low_Birth_Weight"].value_counts()

classifier = None
classification_results = None
classification_error = None

# Stratified train/test splitting requires at least two
# records in each class and enough records for both splits.
if len(class_counts) == 2 and class_counts.min() >= 2:
    X_cls = classification_df[FEATURES]
    y_cls = classification_df["Low_Birth_Weight"]

    try:
        X_train_cls, X_test_cls, y_train_cls, y_test_cls = (
            train_test_split(
                X_cls,
                y_cls,
                test_size=0.2,
                random_state=42,
                stratify=y_cls,
            )
        )

        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        )

        classifier.fit(X_train_cls, y_train_cls)

        cls_predictions = classifier.predict(X_test_cls)
        cls_probabilities = classifier.predict_proba(
            X_test_cls
        )[:, 1]

        classification_results = {
            "accuracy": accuracy_score(
                y_test_cls, cls_predictions
            ),
            "precision": precision_score(
                y_test_cls,
                cls_predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test_cls,
                cls_predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                y_test_cls,
                cls_predictions,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_test_cls, cls_predictions
            ),
            "report": classification_report(
                y_test_cls,
                cls_predictions,
                target_names=[
                    "Not Low Birth Weight",
                    "Low Birth Weight",
                ],
                zero_division=0,
                output_dict=True,
            ),
            "roc_auc": (
                roc_auc_score(y_test_cls, cls_probabilities)
                if y_test_cls.nunique() == 2
                else None
            ),
        }

    except ValueError as exc:
        classification_error = str(exc)
else:
    classification_error = (
        "The dataset does not contain enough examples "
        "from both birth-weight classes to train the classifier."
    )

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "Overview",
        "Data Analysis",
        "Birth Weight Prediction",
        "Low Birth Weight Classification",
        "Model Evaluation",
    ],
)

st.sidebar.caption(
    "Kisumu West Sub-County group capstone project"
)

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":
    st.header("Project Overview")

    total_records = len(df)
    low_weight_count = int(df["Low_Birth_Weight"].sum())
    low_weight_rate = (
        low_weight_count / total_records * 100
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Usable Infant Records", f"{total_records:,}")
    col2.metric(
        "Average Birth Weight",
        f"{df['Birth_weight'].mean():,.1f} g",
    )
    col3.metric(
        "Low Birth Weight",
        f"{low_weight_rate:.2f}%",
    )

    st.markdown("### Dataset preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Variables used by the models")

    st.write(
        "- **Gravidae:** number of pregnancies recorded."
    )
    st.write(
        "- **Parity_First:** first numeric component extracted "
        "from the notebook's Parity field."
    )
    st.write(
        "- **Parity_Second:** second numeric component extracted "
        "from the notebook's Parity field."
    )
    st.write(
        "- **Birth_weight:** infant birth weight in grams."
    )

    st.caption(
        "Parity_First and Parity_Second retain the meanings "
        "of the two components as encoded in the source dataset."
    )

# --------------------------------------------------
# DATA ANALYSIS
# --------------------------------------------------

elif page == "Data Analysis":
    st.header("Exploratory Data Analysis")

    st.markdown("### Birth weight distribution")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        df["Birth_weight"],
        bins=30,
        edgecolor="black",
    )
    ax.set_xlabel("Birth Weight (grams)")
    ax.set_ylabel("Number of Infants")
    ax.set_title("Distribution of Infant Birth Weight")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Gravidity and birth weight")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        df["Gravidae"],
        df["Birth_weight"],
        alpha=0.5,
    )
    ax.set_xlabel("Gravidae")
    ax.set_ylabel("Birth Weight (grams)")
    ax.set_title("Gravidity and Birth Weight")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Parity components and birth weight")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        df["Parity_First"],
        df["Birth_weight"],
        alpha=0.5,
    )
    ax.set_xlabel("Parity_First")
    ax.set_ylabel("Birth Weight (grams)")
    ax.set_title("Parity_First and Birth Weight")
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        df["Parity_Second"],
        df["Birth_weight"],
        alpha=0.5,
    )
    ax.set_xlabel("Parity_Second")
    ax.set_ylabel("Birth Weight (grams)")
    ax.set_title("Parity_Second and Birth Weight")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Birth weight category")

    category_counts = (
        df["Low_Birth_Weight"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        ["Not Low Birth Weight", "Low Birth Weight"],
        category_counts.values,
    )
    ax.set_ylabel("Number of Infants")
    ax.set_title("Low and Normal Birth Weight Categories")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Correlation matrix")

    corr = df[
        FEATURES + ["Birth_weight"]
    ].corr(numeric_only=True)

    st.dataframe(
        corr.round(3),
        use_container_width=True,
    )

# --------------------------------------------------
# BIRTH WEIGHT PREDICTION
# --------------------------------------------------

elif page == "Birth Weight Prediction":
    st.header("Predict Infant Birth Weight")

    st.write(
        "Enter the three predictor values used by the "
        "Decision Tree model in the group notebook."
    )

    with st.form("birth_weight_form"):
        gravidae = st.number_input(
            "Gravidae",
            min_value=0,
            max_value=30,
            value=2,
            step=1,
        )

        parity_first = st.number_input(
            "Parity_First",
            min_value=0,
            max_value=30,
            value=1,
            step=1,
        )

        parity_second = st.number_input(
            "Parity_Second",
            min_value=0,
            max_value=30,
            value=0,
            step=1,
        )

        submitted = st.form_submit_button(
            "Predict Birth Weight"
        )

    if submitted:
        input_data = pd.DataFrame(
            [[gravidae, parity_first, parity_second]],
            columns=FEATURES,
        )

        predicted_weight = float(
            reg_model.predict(input_data)[0]
        )

        st.success(
            f"Predicted birth weight: "
            f"{predicted_weight:,.1f} grams"
        )

        if predicted_weight < 2500:
            st.warning(
                "The prediction is below 2,500 grams, "
                "the low-birth-weight threshold used in this project."
            )
        else:
            st.info(
                "The prediction is at least 2,500 grams, "
                "the threshold used in this project."
            )

        st.caption(
            "This is a model estimate, not a clinical assessment. "
            "It must not be used on its own to make medical decisions."
        )

# --------------------------------------------------
# LOW BIRTH WEIGHT CLASSIFICATION
# --------------------------------------------------

elif page == "Low Birth Weight Classification":
    st.header("Classify Low Birth Weight")

    st.write(
        "Logistic Regression estimates whether the infant's "
        "birth weight falls below 2,500 grams."
    )

    if classifier is None:
        st.warning(
            "Classification is unavailable: "
            + str(classification_error)
        )
    else:
        with st.form("low_weight_form"):
            gravidae_cls = st.number_input(
                "Gravidae",
                min_value=0,
                max_value=30,
                value=2,
                step=1,
                key="gravidae_cls",
            )

            parity_first_cls = st.number_input(
                "Parity_First",
                min_value=0,
                max_value=30,
                value=1,
                step=1,
                key="parity_first_cls",
            )

            parity_second_cls = st.number_input(
                "Parity_Second",
                min_value=0,
                max_value=30,
                value=0,
                step=1,
                key="parity_second_cls",
            )

            classify = st.form_submit_button(
                "Classify Birth Weight"
            )

        if classify:
            input_data = pd.DataFrame(
                [[
                    gravidae_cls,
                    parity_first_cls,
                    parity_second_cls,
                ]],
                columns=FEATURES,
            )

            probability = float(
                classifier.predict_proba(input_data)[0, 1]
            )

            predicted_class = int(
                classifier.predict(input_data)[0]
            )

            st.metric(
                "Estimated probability of low birth weight",
                f"{probability:.1%}",
            )

            if predicted_class == 1:
                st.warning(
                    "Model classification: Low Birth Weight "
                    "(below 2,500 grams)."
                )
            else:
                st.success(
                    "Model classification: Not Low Birth Weight "
                    "(2,500 grams or above)."
                )

            st.caption(
                "This classification is a statistical estimate, "
                "not a diagnosis or a substitute for clinical care."
            )

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

elif page == "Model Evaluation":
    st.header("Model Evaluation")

    st.subheader("Decision Tree Regression")

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{mae:,.2f} g")
    c2.metric("RMSE", f"{rmse:,.2f} g")
    c3.metric("R² Score", f"{r2:.4f}")

    st.write(
        "MAE and RMSE measure prediction error in grams. "
        "R² indicates how much variation is explained by the model "
        "on the held-out test data."
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        y_test_reg,
        reg_predictions,
        alpha=0.6,
    )

    low = min(
        float(y_test_reg.min()),
        float(reg_predictions.min()),
    )
    high = max(
        float(y_test_reg.max()),
        float(reg_predictions.max()),
    )

    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual Birth Weight (grams)")
    ax.set_ylabel("Predicted Birth Weight (grams)")
    ax.set_title("Actual vs Predicted Birth Weight")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Logistic Regression Classification")

    if classification_results is None:
        st.warning(
            "Classification evaluation is unavailable: "
            + str(classification_error)
        )
    else:
        result = classification_results

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Accuracy", f"{result['accuracy']:.1%}")
        c2.metric("Precision", f"{result['precision']:.1%}")
        c3.metric("Recall", f"{result['recall']:.1%}")
        c4.metric("F1 Score", f"{result['f1']:.3f}")

        if result["roc_auc"] is not None:
            st.metric("ROC-AUC", f"{result['roc_auc']:.3f}")

        st.write("Confusion matrix:")
        st.dataframe(
            pd.DataFrame(
                result["confusion_matrix"],
                index=[
                    "Actual: Not Low",
                    "Actual: Low",
                ],
                columns=[
                    "Predicted: Not Low",
                    "Predicted: Low",
                ],
            ),
            use_container_width=True,
        )

        st.write("Classification report:")
        st.dataframe(
            pd.DataFrame(result["report"]).T.round(3),
            use_container_width=True,
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Group Capstone Project | Diploma in Data Science and Analytics | "
    "Tropical Institute of Community Health and Development (TICH)"
)

st.caption(
    "Findings describe the available maternity records and "
    "do not independently establish causation."
)
