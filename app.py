
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Kisumu West Birth Weight Analysis",
    page_icon="👶",
    layout="wide"
)

st.title("The Effect of Gravidity and Parity on Birth Weight of Infants")
st.subheader("Kisumu West Sub-County, Kenya")

st.write(
    "This group capstone application explores the relationship between "
    "gravidity, parity, and infant birth weight using maternity register data. "
    "The findings describe available records and do not independently "
    "establish causation."
)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

DATA_PATH = (
    Path(__file__).parent
    / "data"
    / "processed_maternity_data.csv"
)


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Check that the maternity CSV file is "
        "inside the data folder in this GitHub repository."
    )
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except Exception as error:
    st.error(f"The dataset could not be loaded: {error}")
    st.stop()

if df.empty:
    st.error("The dataset contains no records.")
    st.stop()


# --------------------------------------------------
# 3. PREPARE AVAILABLE NUMERIC COLUMNS
# --------------------------------------------------

numeric_columns = [
    "Gravidae",
    "Parity_First",
    "Parity_Second",
    "Babies born",
    "Birth_weight",
    "Birth weight 1",
    "Birth weight 2",
    "Birth weight 3"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

st.success("The maternity dataset was loaded successfully.")


# --------------------------------------------------
# 4. DATA FILTERS
# --------------------------------------------------

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

if df.empty:
    st.warning("No records match the selected facility filters.")
    st.stop()


# --------------------------------------------------
# 5. DATASET OVERVIEW
# --------------------------------------------------

st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Records displayed", f"{len(df):,}")

if "Gravidae" in df.columns:
    col2.metric(
        "Gravidity records",
        f"{df['Gravidae'].notna().sum():,}"
    )
else:
    col2.metric("Gravidity records", "Unavailable")

weight_cols = [
    column
    for column in ["Birth weight 1", "Birth weight 2", "Birth weight 3"]
    if column in df.columns
]

if weight_cols:
    weights = pd.to_numeric(
        pd.Series(df[weight_cols].values.flatten()),
        errors="coerce"
    ).dropna()

    col3.metric("Recorded infant weights", f"{len(weights):,}")
else:
    weights = pd.Series(dtype=float)
    col3.metric("Recorded infant weights", "Unavailable")

with st.expander("View dataset columns and sample records"):
    st.write("Columns:", ", ".join(df.columns.astype(str)))
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("View numerical summary"):
    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )


# --------------------------------------------------
# 6. EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

st.header("2. Exploratory Data Analysis")

if weight_cols and "Gravidae" in df.columns:

    plot_df = df.copy()

    plot_df["Gravidae"] = pd.to_numeric(
        plot_df["Gravidae"],
        errors="coerce"
    )

    for column in weight_cols:
        plot_df[column] = pd.to_numeric(
            plot_df[column],
            errors="coerce"
        )

    # Convert the recorded infant weight columns into one column
    plot_df = plot_df.melt(
        id_vars=["Gravidae"],
        value_vars=weight_cols,
        value_name="Birth_weight",
        var_name="Baby_number"
    )

    plot_df["Birth_weight"] = pd.to_numeric(
        plot_df["Birth_weight"],
        errors="coerce"
    )

    plot_df = plot_df.dropna(
        subset=["Gravidae", "Birth_weight"]
    )

    plot_df = plot_df[plot_df["Birth_weight"] > 0]

    # Birth-weight distribution
    if not plot_df.empty:
        fig, ax = plt.subplots()

        ax.hist(
            plot_df["Birth_weight"],
            bins=25,
            edgecolor="black"
        )

        ax.set_title("Distribution of Infant Birth Weight")
        ax.set_xlabel("Birth weight (grams)")
        ax.set_ylabel("Number of recorded infants")

        st.pyplot(fig)
        plt.close(fig)

        # Gravidity versus birth weight
        fig, ax = plt.subplots()

        ax.scatter(
            plot_df["Gravidae"],
            plot_df["Birth_weight"],
            alpha=0.5
        )

        ax.set_title("Gravidity and Birth Weight")
        ax.set_xlabel("Gravidae (number of pregnancies)")
        ax.set_ylabel("Birth weight (grams)")

        st.pyplot(fig)
        plt.close(fig)

    # Babies born versus birth weight, if available
    if "Babies born" in df.columns:
        babies_df = df.copy()

        babies_df["Babies born"] = pd.to_numeric(
            babies_df["Babies born"],
            errors="coerce"
        )

        babies_df["Birth_weight"] = babies_df[weight_cols].mean(axis=1)

        babies_df = babies_df.dropna(
            subset=["Babies born", "Birth_weight"]
        )

        if not babies_df.empty:
            fig, ax = plt.subplots()

            ax.scatter(
                babies_df["Babies born"],
                babies_df["Birth_weight"],
                alpha=0.5
            )

            ax.set_title("Babies Born and Birth Weight")
            ax.set_xlabel("Babies born")
            ax.set_ylabel("Average recorded birth weight (grams)")

            st.pyplot(fig)
            plt.close(fig)

else:
    st.warning(
        "The expected gravidity or birth-weight columns were not found. "
        "Check the CSV column names."
    )


# --------------------------------------------------
# 7. BIRTH-WEIGHT SUMMARY
# --------------------------------------------------

st.header("3. Birth Weight Summary")

if not weights.empty:
    st.dataframe(
        weights.describe().to_frame("Birth weight summary"),
        use_container_width=True
    )

    valid_weights = weights[weights > 0]
    low_count = int((valid_weights < 2500).sum())
    normal_count = int((valid_weights >= 2500).sum())
    total_count = low_count + normal_count

    a, b, c = st.columns(3)

    a.metric("Low birth weight (<2500 g)", f"{low_count:,}")
    b.metric("2500 g or above", f"{normal_count:,}")

    c.metric(
        "Low birth-weight percentage",
        f"{100 * low_count / total_count:.2f}%"
        if total_count else "Unavailable"
    )

    fig, ax = plt.subplots()

    ax.bar(
        ["Low birth weight", "2500 g or above"],
        [low_count, normal_count]
    )

    ax.set_title("Low Birth Weight Classification")
    ax.set_ylabel("Number of recorded infants")

    st.pyplot(fig)
    plt.close(fig)

st.caption(
    "Birth weight below 2,500 grams is classified as low birth weight. "
    "These summaries describe the available records."
)


# --------------------------------------------------
# 8. MODEL EVALUATION INFORMATION
# --------------------------------------------------

st.header("4. Model Evaluation Information")

st.subheader("Decision Tree Regression")

st.write(
    "The group's earlier regression analysis reported the following "
    "evaluation results. These figures are reported results, not a "
    "guarantee of performance on every dataset or new input."
)

m1, m2, m3 = st.columns(3)

m1.metric("MAE", "339.53 g")
m2.metric("RMSE", "431.62 g")
m3.metric("R² Score", "0.0621")

st.caption(
    "The R² score indicates that the model explained only a small "
    "proportion of the variation in birth weight in that evaluation."
)

st.subheader("Logistic Regression")

st.write(
    "The group's earlier low-birth-weight classification analysis "
    "reported these evaluation results."
)

l1, l2, l3, l4 = st.columns(4)

l1.metric("Accuracy", "67.17%")
l2.metric("Precision", "9.19%")
l3.metric("Recall", "60.34%")
l4.metric("F1 Score", "15.95%")

st.caption(
    "The classification results have limitations and must not be "
    "used to make clinical decisions."
)


# --------------------------------------------------
# 9. LIVE BIRTH-WEIGHT PREDICTION
# --------------------------------------------------

st.header("5. Birth Weight Prediction")

st.write(
    "This section trains a Decision Tree regression model from the "
    "available dataset and accepts the three predictor values used "
    "in the latest group notebook."
)

st.warning(
    "For academic demonstration only. Predictions are estimates, "
    "not medical advice or a substitute for clinical assessment."
)

regression_features = [
    "Gravidae",
    "Parity_First",
    "Parity_Second"
]

missing_features = [
    column
    for column in regression_features
    if column not in df.columns
]

if missing_features:
    st.error(
        "Live prediction is unavailable because the dataset is missing "
        "these notebook predictors: "
        + ", ".join(missing_features)
        + ". Export the correct processed dataset from the latest "
        "group notebook and place it in the data folder. The dashboard "
        "and charts above can still be viewed."
    )

elif not weight_cols:
    st.warning(
        "Live prediction requires at least one birth-weight column."
    )

else:
    # Prepare the data using the three notebook predictors
    model_data = df[regression_features + weight_cols].copy()

    for column in regression_features + weight_cols:
        model_data[column] = pd.to_numeric(
            model_data[column],
            errors="coerce"
        )

    # Convert the available birth-weight columns to long format
    model_data = model_data.melt(
        id_vars=regression_features,
        value_vars=weight_cols,
        value_name="Birth_weight",
        var_name="Baby_number"
    )

    model_data["Birth_weight"] = pd.to_numeric(
        model_data["Birth_weight"],
        errors="coerce"
    )

    model_data = model_data.dropna(
        subset=regression_features + ["Birth_weight"]
    )

    model_data = model_data[model_data["Birth_weight"] > 0]

    if len(model_data) < 20:
        st.warning(
            "There are not enough complete records to train the "
            "prediction model."
        )

    else:
        X = model_data[regression_features]
        y = model_data["Birth_weight"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        reg_model = DecisionTreeRegressor(
            max_depth=4,
            random_state=42
        )

        reg_model.fit(X_train, y_train)

        # Display evaluation on this app's available data
        test_predictions = reg_model.predict(X_test)

        e1, e2, e3 = st.columns(3)

        e1.metric(
            "App MAE",
            f"{mean_absolute_error(y_test, test_predictions):.2f} g"
        )

        e2.metric(
            "App RMSE",
            f"{mean_squared_error(y_test, test_predictions) ** 0.5:.2f} g"
        )

        e3.metric(
            "App R²",
            f"{r2_score(y_test, test_predictions):.4f}"
        )

        st.subheader("Enter the prediction values")

        with st.form("birth_weight_form"):

            gravidae = st.number_input(
                "Gravidae (number of pregnancies)",
                min_value=0,
                max_value=30,
                value=1,
                step=1
            )

            parity_first = st.number_input(
                "Parity first value",
                min_value=0,
                max_value=30,
                value=0,
                step=1
            )

            parity_second = st.number_input(
                "Parity second value",
                min_value=0,
                max_value=30,
                value=0,
                step=1
            )

            submitted = st.form_submit_button(
                "Estimate birth weight"
            )

        if submitted:
            input_data = pd.DataFrame([{
                "Gravidae": gravidae,
                "Parity_First": parity_first,
                "Parity_Second": parity_second
            }])

            prediction = reg_model.predict(input_data)[0]

            st.metric(
                "Estimated birth weight",
                f"{prediction:,.0f} grams"
            )

            st.caption(
                "This is an experimental model estimate. Actual infant "
                "birth weight must be measured, and clinical decisions "
                "must be made by qualified health professionals."
            )


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Group capstone project | TICH School of Data Science and Analytics | 2026"
)
