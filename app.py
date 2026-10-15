
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
    "data/Filtered maternity Register - Filtered maternity Register.csv"
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
