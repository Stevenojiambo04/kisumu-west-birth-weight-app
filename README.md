The Effect of Gravidity and Parity on Birth Weight of Infants in Kisumu West Sub-County

Project Overview

This group capstone project examines the relationship between gravidity, the number of babies previously born as recorded in the available maternity data, and infant birth weight in Kisumu West Sub-County, Kenya.

The project applies data cleaning, exploratory data analysis, visualization, and machine-learning techniques to investigate patterns in the maternity records.

Project Objectives

Explore the relationship between gravidity, babies born, and infant birth weight.

Prepare and analyze maternity-register data.

Develop and evaluate machine-learning models for birth-weight estimation and low-birth-weight classification.

Present findings through an interactive web application.

Technologies Used

Python

Pandas

Matplotlib

Scikit-learn

Streamlit

Google Colab

GitHub

Repository Contents

app.py — Streamlit application source code.

requirements.txt — Python packages required by the application.

README.md — Project documentation and usage instructions.

Kisumu_West_Birth_Weight_Analysis.ipynb — Analysis notebook.

data/ — Maternity-register dataset used by the application.

Machine-Learning Models

Decision Tree Regression: Estimates infant birth weight using gravidity and the recorded number of babies born. The notebook's reported evaluation results were MAE of approximately 339.53 g, RMSE of approximately 431.62 g, and R² of approximately 0.0621.

Logistic Regression: Classifies birth weight as low birth weight (below 2,500 g) or not low birth weight. The notebook's reported results were accuracy of approximately 67.17%, precision of 9.19%, recall of 60.34%, and F1 score of 15.95%.

These metrics describe the evaluated notebook models. Results from the application's prediction interface may differ because the app prepares its modelling data separately.

Run the Application Locally

Install Python.

Clone or download this repository.

Open a terminal in the project folder.

Install the dependencies:

pip install -r requirements.txt

Start the application:

streamlit run app.py

Open the local address displayed in the terminal.

Ensure the dataset is present at the path expected by app.py.

Deployed Application

https://kisumu-west-birth-weight-app-xk24emclnmmrx87dosetrr.streamlit.app/

Responsible Use and Limitations

This project is an academic demonstration. The models have limited predictive performance and must not be used to diagnose, treat, or make clinical decisions about mothers or infants. Actual infant birth weight should be measured, and healthcare decisions must be made by qualified professionals.

The findings describe patterns in the available records and do not independently establish causation. Data privacy and responsible interpretation should be maintained when handling maternity information.

Academic Project

Group capstone project submitted to the Tropical Institute of Community Health and Development (TICH), School of Data Science and Analytics, toward the Diploma in Data Science.
