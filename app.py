import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Load the data
genes = pd.read_csv("sfari_genes.csv")

# Drop unnecessary columns
columns_to_drop = ['status', 'chromosome', 'number-of-reports', 'gene-name', 'ensembl-id', 'gene-score', 'genetic-category']
genes = genes.drop(columns=columns_to_drop)

# Encode gene symbols as dummy variables
genes_encoded = pd.get_dummies(genes, columns=['gene-symbol'])

# Features (X) excluding the 'syndromic' column
X = genes_encoded.drop(columns='syndromic')

# Labels (y)
y = genes_encoded['syndromic']

# Convert to binary classification (1 for syndromic, 0 for non-syndromic)
y_binary = (y == 1).astype(int)

# Resample the dataset using SMOTETomek
smotetomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smotetomek.fit_resample(X, y_binary)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize the classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier()
}

# Create a Streamlit app
st.title("Autism Gene Classifier")

# Get user input for a gene symbol
gene_symbol = st.text_input("Enter a gene symbol: ")

# Check if the gene symbol exists in the data
if gene_symbol in genes['gene-symbol'].values:
    # Extract the corresponding row from the dataframe
    gene_info = genes[genes['gene-symbol'].astype(str) == gene_symbol]

    # Check if the gene is syndromic or not
    if gene_info['syndromic'].values[0] == 1:
        st.write(f"The gene {gene_symbol} is associated with autism.")
    else:
        st.write(f"The gene {gene_symbol} is not associated with autism.")
else:
    st.write("The gene symbol does not exist in the data.")
