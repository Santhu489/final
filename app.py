# app.py

import streamlit as st
import pandas as pd
import pickle

def preprocess_input(gene_info):
    # Drop unnecessary columns
    columns_to_drop = ['status', 'chromosome', 'number-of-reports', 'gene-name', 'ensembl-id', 'gene-score', 'genetic-category']
    gene_info = gene_info.drop(columns=columns_to_drop)

    return gene_info

# Load the data
genes = pd.read_csv("sfari_genes.csv")

# Get user input for a gene symbol
gene_symbol = st.text_input("Enter a gene symbol:")

if st.button("Classify Gene"):
    if gene_symbol in genes['gene-symbol'].values:
        # Extract the corresponding row from the dataframe
        gene_info = genes[genes['gene-symbol'] == gene_symbol]

        # Preprocess the input
        gene_info_processed = preprocess_input(gene_info)

        # Load the models
        svm_model = pickle.load(open("SVM_model.pkl", "rb"))
        rf_model = pickle.load(open("Random_Forest_model.pkl", "rb"))

        # Make predictions using each model
        svm_prediction = svm_model.predict(gene_info_processed)
        rf_prediction = rf_model.predict(gene_info_processed)

        # Return the majority vote as the final prediction
        syndromic_status = int((svm_prediction + rf_prediction) >= 1)

        st.subheader("Classification Result")
        if syndromic_status == 1:
            st.write(f"The gene {gene_symbol} is associated with autism.")
        else:
            st.write(f"The gene {gene_symbol} is not associated with autism.")
    else:
        st.write("The gene symbol does not exist in the data.")
