# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler
import pickle

def preprocess_data(genes):
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

    # Resample the dataset
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y_binary)

    return X_resampled, y_resampled

def train_and_save_models(X, y):
    classifiers = {
        'SVM': SVC(),
        'Random Forest': RandomForestClassifier()
    }

    for clf_name, clf in classifiers.items():
        # Train the classifier
        clf.fit(X, y)

        # Evaluate the classifier
        y_pred = clf.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        print(f"\nResults for {clf_name} on original data:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        # Classification Report
        report = classification_report(y, y_pred)
        print(report)

        # Save the trained model to disk
        filename = f"{clf_name}_model.pkl"
        with open(filename, "wb") as f:
            pickle.dump(clf, f)

    return classifiers

# main.py
import streamlit as st
import pandas as pd
import pickle

def load_model(model_name):
    """
    Load a trained model from disk.
    """
    with open(f"{model_name}.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def predict(gene_info, classifiers):
    """
    Make a prediction for a given gene using the loaded models.
    """
    # Make predictions using each model
    predictions = []
    for clf_name, clf in classifiers.items():
        prediction = clf.predict(gene_info)
        predictions.append(prediction[0])

    # Return the majority vote as the final prediction
    return max(set(predictions), key=predictions.count)

# Streamlit app
st.title('Autism Genes Classifier')
st.markdown("A simple web app to classify genes as syndromic or non-syndromic.")

# Load the data
genes = pd.read_csv("sfari_genes.csv")

# Train and save the classifiers
X_resampled, y_resampled = preprocess_data(genes)
classifiers = train_and_save_models(X_resampled, y_resampled)

# Get user input for a gene symbol
gene_symbol = st.text_input("Enter a gene symbol:")

if st.button("Classify Gene"):
    if gene_symbol in genes['gene-symbol'].values:
        # Extract the corresponding row from the dataframe
        gene_info = genes[genes['gene-symbol'] == gene_symbol]

        # Classify the gene as syndromic or non-syndromic
        syndromic_status = predict(gene_info, classifiers)

        if syndromic_status == 1:
            st.subheader("Classification Result")
            st.write(f"The gene {gene_symbol} is associated with autism.")
        else:
            st.subheader("Classification Result")
            st.write(f"The gene {gene_symbol} is not associated with autism.")
    else:
        st.write("The gene symbol does not exist in the data.")
