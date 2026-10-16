# kisumu-west-birth-weight-app
Streamlit deployment app for the Kisumu West gravidity, parity and infant birth weight capstone project

st.header("Model Information")

st.write(
    "This section presents the machine-learning models evaluated "
    "during the group capstone analysis. The results shown are from "
    "our completed analysis; predictions are not yet integrated "
    "into this application."
)

st.subheader("1. Decision Tree Regression")

st.write(
    "Purpose: Estimate infant birth weight using gravidity and "
    "the number of babies born."
)

col1, col2, col3 = st.columns(3)

col1.metric("MAE", "339.53 g")
col2.metric("RMSE", "431.62 g")
col3.metric("R² Score", "0.0621")

st.caption(
    "The low R² score indicates that the model explains only a "
    "small proportion of the variation in birth weight."
)

st.subheader("2. Logistic Regression")

st.write(
    "Purpose: Classify infants as having low birth weight "
    "(below 2,500 g) or not."
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "67.17%")
col2.metric("Precision", "9.19%")
col3.metric("Recall", "60.34%")
col4.metric("F1 Score", "15.95%")


st.divider()
st.header("Low Birth Weight Analysis")

st.write(
    "Low birth weight (LBW) is defined as a birth weight below "
    "2,500 grams. This section summarises the recorded infant "
    "weights currently available in the selected data."
)

# Use positive recorded weights for this summary
valid_weights = weights[weights > 0]

low_weights = valid_weights[valid_weights < 2500]
other_weights = valid_weights[valid_weights >= 2500]

if not valid_weights.empty:
    total_valid = len(valid_weights)
    lbw_count = len(low_weights)
    other_count = len(other_weights)

    lbw_percentage = lbw_count / total_valid * 100
    other_percentage = other_count / total_valid * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("Valid recorded weights", f"{total_valid:,}")
    col2.metric("Low birth weight (<2,500 g)", f"{lbw_count:,}")
    col3.metric("Low birth weight (%)", f"{lbw_percentage:.2f}%")

    chart_data = pd.DataFrame({
        "Category": ["Low birth weight", "2,500 g or above"],
        "Number of infants": [lbw_count, other_count]
    })

    fig, ax = plt.subplots()
    ax.bar(chart_data["Category"], chart_data["Number of infants"])
    ax.set_title("Distribution of Recorded Birth Weights")
    ax.set_ylabel("Number of recorded infant weights")
    ax.tick_params(axis="x", rotation=10)

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "This chart is calculated from positive recorded weights in "
        "the current dataset. Its counts may differ from the cleaned "
        "analysis used for the group's final model results."
    )
else:
    st.warning("No positive recorded birth weights are available.")
st.caption(
    "The low precision and F1 score indicate that this classifier "
    "has important limitations. These results should be interpreted "
    "carefully, especially when considering healthcare applications."
)
