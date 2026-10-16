
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Kisumu West Birth Weight Analysis",
    page_icon="👶",
    layout="wide"
)

st.title("The Effect of Gravidity and Parity on Birth Weight")
st.subheader("Kisumu West Sub-County Maternity Data Analysis")

st.write(
    "This application explores the relationship between gravidity, "
    "previous births, and infant birth weight."
)
DATA_PATH = Path(
    "data/Filtered maternity Register - Filtered maternity Register (1).csv"
)
if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Please upload the maternity CSV file "
        "into the data folder in this GitHub repository."
    )
    st.stop()

df = pd.read_csv(DATA_PATH)

st.sidebar.header("Data filters")

if "Facility Name (Kisumu West)" in df.columns:
    facilities = sorted(
        df["Facility Name (Kisumu West)"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_facilities = st.sidebar.multiselect(
        "Select facility",
        facilities,
        default=facilities
    )

    df = df[
        df["Facility Name (Kisumu West)"]
        .astype(str)
        .isin(selected_facilities)
    ]

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Records displayed", f"{len(df):,}")

if "Gravidae" in df.columns:
    col2.metric(
        "Gravidity records",
        f"{df['Gravidae'].notna().sum():,}"
    )
else:
    col2.metric("Gravidity records", "Not available")

weight_cols = [
    col for col in [
        "Birth weight 1",
        "Birth weight 2",
        "Birth weight 3"
    ]
    if col in df.columns
]

if weight_cols:
    weights = pd.to_numeric(
        pd.Series(df[weight_cols].values.flatten()),
        errors="coerce"
    ).dropna()

    col3.metric("Recorded infant weights", f"{len(weights):,}")
else:
    weights = pd.Series(dtype=float)
    col3.metric("Recorded infant weights", "Not available")

st.divider()
st.header("Explore the Data")

if "Gravidae" in df.columns and weight_cols:
    plot_df = df.copy()

    plot_df["Gravidae"] = pd.to_numeric(
        plot_df["Gravidae"], errors="coerce"
    )

    for col in weight_cols:
        plot_df[col] = pd.to_numeric(
            plot_df[col], errors="coerce"
        )

    plot_df["Birth_weight"] = plot_df[weight_cols].mean(axis=1)

    plot_df = plot_df.dropna(
        subset=["Gravidae", "Birth_weight"]
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Gravidity and Birth Weight")

        fig, ax = plt.subplots()
        ax.scatter(
            plot_df["Gravidae"],
            plot_df["Birth_weight"],
            alpha=0.5
        )
        ax.set_xlabel("Gravidity")
        ax.set_ylabel("Birth weight (as recorded)")
        ax.set_title("Gravidity vs Birth Weight")
        st.pyplot(fig)
        plt.close(fig)

    if "Babies born" in plot_df.columns:
        with right:
            st.subheader("Babies Born and Birth Weight")

            plot_df["Babies born"] = pd.to_numeric(
                plot_df["Babies born"], errors="coerce"
            )

            fig, ax = plt.subplots()
            ax.scatter(
                plot_df["Babies born"],
                plot_df["Birth_weight"],
                alpha=0.5
            )
            ax.set_xlabel("Babies born")
            ax.set_ylabel("Birth weight (as recorded)")
            ax.set_title("Babies Born vs Birth Weight")
            st.pyplot(fig)
            plt.close(fig)
else:
    st.warning(
        "The expected gravidity or birth-weight columns were not found. "
        "Please check the CSV column names."
    )

st.header("Birth Weight Summary")

if not weights.empty:
    st.dataframe(
        weights.describe().to_frame("Birth weight summary")
    )

st.caption(
    "Academic capstone demonstration. Findings describe the available "
    "records and do not independently establish causation."
)


st.divider()

st.header("Model Information")

st.write(
    "The following machine-learning evaluation results were obtained "
    "during the group's capstone analysis. Live model predictions are "
    "not currently integrated into this application."
)

st.subheader("1. Decision Tree Regression")

st.write(
    "This model estimates infant birth weight using gravidity and "
    "the number of babies born."
)

col1, col2, col3 = st.columns(3)

col1.metric("Mean Absolute Error (MAE)", "339.53 g")
col2.metric("Root Mean Squared Error (RMSE)", "431.62 g")
col3.metric("R² Score", "0.0621")

st.caption(
    "The R² score indicates that the model explains only a small "
    "proportion of the variation in birth weight."
)

st.subheader("2. Logistic Regression")

st.write(
    "This model classifies infants into low birth weight "
    "(below 2,500 g) and non-low birth weight categories."
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "67.17%")
col2.metric("Precision", "9.19%")
col3.metric("Recall", "60.34%")
col4.metric("F1 Score", "15.95%")

st.caption(
    "The low precision and F1 score indicate important limitations. "
    "The results should not be used as a substitute for clinical "
    "assessment."
)

st.divider()
st.header("Birth Weight Prediction")

st.write(
    "Enter gravidity and babies born to obtain an experimental "
    "birth-weight estimate from a Decision Tree model."
)

st.warning(
    "Academic demonstration only. The models have limited predictive "
    "performance and must not be used for clinical decisions."
)

# Prepare modelling data using the same predictors as the group notebook
model_data = df[
    ["Gravidae", "Babies born"] + weight_cols
].copy()

model_data["Gravidae"] = pd.to_numeric(
    model_data["Gravidae"], errors="coerce"
)
model_data["Babies born"] = pd.to_numeric(
    model_data["Babies born"], errors="coerce"
)

# Match the notebook's long-format birth-weight structure
model_data = model_data.melt(
    id_vars=["Gravidae", "Babies born"],
    value_vars=weight_cols,
    value_name="Birth_weight",
    var_name="Baby_number"
)

model_data["Birth_weight"] = pd.to_numeric(
    model_data["Birth_weight"], errors="coerce"
)

model_data = model_data.dropna(
    subset=["Gravidae", "Babies born", "Birth_weight"]
)

# Keep plausible positive weights for this app-level demonstration
model_data = model_data[model_data["Birth_weight"] > 0]

if len(model_data) < 20:
    st.error("There are not enough complete records to train the model.")
else:
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    model_data["Low_Birth_Weight"] = (
        model_data["Birth_weight"] < 2500
    ).astype(int)

    X = model_data[["Gravidae", "Babies born"]]
    y_weight = model_data["Birth_weight"]
    y_low = model_data["Low_Birth_Weight"]

    # Train the regression model using the notebook's tree settings
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_weight, test_size=0.2, random_state=42
    )

    birth_weight_model = DecisionTreeRegressor(
        max_depth=4, random_state=42
    )
    birth_weight_model.fit(X_train, y_train)

    # Train a low-birth-weight classifier when both classes are present
    logistic_model = None

    if y_low.nunique() == 2 and y_low.value_counts().min() >= 2:
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            X, y_low, test_size=0.2, random_state=42, stratify=y_low
        )

        logistic_model = LogisticRegression(
            class_weight="balanced", random_state=42, max_iter=1000
        )
        logistic_model.fit(Xc_train, yc_train)

    st.subheader("Enter the values")

    min_gravidity = max(1, int(model_data["Gravidae"].min()))
    max_gravidity = max(min_gravidity, int(model_data["Gravidae"].max()))
    min_babies = max(0, int(model_data["Babies born"].min()))
    max_babies = max(min_babies, int(model_data["Babies born"].max()))

    with st.form("birth_weight_prediction_form"):
        gravidity_input = st.number_input(
            "Gravidity (number of pregnancies)",
            min_value=min_gravidity,
            max_value=max_gravidity,
            value=min_gravidity,
            step=1
        )

        babies_input = st.number_input(
            "Babies born (recorded previous births)",
            min_value=min_babies,
            max_value=max_babies,
            value=min_babies,
            step=1
        )

        predict_clicked = st.form_submit_button("Predict")

    if predict_clicked:
        input_data = pd.DataFrame({
            "Gravidae": [gravidity_input],
            "Babies born": [babies_input]
        })

        predicted_weight = birth_weight_model.predict(input_data)[0]

        st.subheader("Prediction result")
        st.metric("Estimated birth weight", f"{predicted_weight:,.0f} g")

        if logistic_model is not None:
            predicted_class = logistic_model.predict(input_data)[0]

            if predicted_class == 1:
                st.write(
                    "The experimental classifier flags this input as "
                    "low birth weight (below 2,500 g)."
                )
            else:
                st.write(
                    "The experimental classifier flags this input as "
                    "not low birth weight (2,500 g or above)."
                )
        else:
            st.info("The classifier could not be trained on the available data.")

        st.caption(
            "These are experimental model outputs, not medical advice. "
            "Actual birth weight must be measured, and clinical decisions "
            "must be made by qualified health professionals."
        )
