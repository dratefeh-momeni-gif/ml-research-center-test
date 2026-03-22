"""
Exploratory Data Analysis for Research Center Clustering Task

Author: Dr.Atefeh Momeni
Description:
This script performs initial data inspection for the research center dataset.
The goal is to understand data structure, check for missing values,
and identify candidate features for clustering.
"""

import pandas as pd


# -----------------------------
# Load dataset
# -----------------------------

DATA_PATH = "research_centers.csv"

df = pd.read_csv(DATA_PATH)


# -----------------------------
# Basic shape
# -----------------------------

print("Dataset shape:")
print(df.shape)


# -----------------------------
# Columns
# -----------------------------

print("\nColumns:")
print(df.columns.tolist())


# -----------------------------
# Info
# -----------------------------

print("\nInfo:")
df.info()


# -----------------------------
# Missing values
# -----------------------------

print("\nMissing values per column:")
print(df.isnull().sum())


# -----------------------------
# Duplicate rows
# -----------------------------

print("\nDuplicate rows:")
print(df.duplicated().sum())


# -----------------------------
# Describe numeric features
# -----------------------------

print("\nDescribe:")
print(df.describe())


# -----------------------------
# Unique values for categorical
# -----------------------------

if "city" in df.columns:
    print("\nUnique cities:")
    print(df["city"].unique())