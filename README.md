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

st.caption(
    "The low precision and F1 score indicate that this classifier "
    "has important limitations. These results should be interpreted "
    "carefully, especially when considering healthcare applications."
)
