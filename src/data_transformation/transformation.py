import os
import sys
import pandas as pd

# ------------------ SETUP ------------------
# Make paths robust by defining them relative to the project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utility.exception import CustomException
from src.utility.logger import setup_logging

# Setup logging
logger = setup_logging("data_transformation")

# Define path for the input CSV file
prepared_csv_file = os.path.join(PROJECT_ROOT, "data", "prepared", "prepared_heart_data_latest.csv")

# Define path for the output CSV file
transformed_data_dir = os.path.join(PROJECT_ROOT, "data", "transformed")
os.makedirs(transformed_data_dir, exist_ok=True)
transformed_csv_file = os.path.join(transformed_data_dir, "transformed_heart_data.csv")

def perform_feature_engineering(df):
    """
    Performs feature engineering for the Heart Disease dataset.
    Creates derived features relevant to cardiac health.
    """
    logger.info("Starting feature engineering...")
    df_transformed = df.copy()

    # 1. Rate Pressure Product (RPP)
    # RPP = Heart Rate * Systolic Blood Pressure. It is a measure of the stress put on the cardiac muscle.
    # Note: Features are scaled, but the interaction term is still valuable for non-linear models.
    if 'thalach' in df_transformed.columns and 'trestbps' in df_transformed.columns:
        df_transformed['rate_pressure_product'] = df_transformed['thalach'] * df_transformed['trestbps']
        logger.info("Created feature: rate_pressure_product")

    # 2. Age Groups
    # Binning age into quartiles to capture non-linear risk factors associated with aging.
    if 'age' in df_transformed.columns:
        try:
            df_transformed['age_group'] = pd.qcut(
                df_transformed['age'], 
                q=4, 
                labels=['age_q1', 'age_q2', 'age_q3', 'age_q4'], 
                duplicates='drop'
            )
            # One-hot encode the new categorical feature
            df_transformed = pd.get_dummies(df_transformed, columns=['age_group'], prefix='', prefix_sep='')
            logger.info("Created features from age groups.")
        except Exception as e:
            logger.warning(f"Could not create age groups. Error: {e}")

    # 3. High Risk Flag
    # Flag patients with high ST depression (oldpeak) and high number of vessels colored (ca)
    # Using 75th percentile as threshold for 'high' since data is scaled.
    if 'oldpeak' in df_transformed.columns and 'ca' in df_transformed.columns:
        high_oldpeak = df_transformed['oldpeak'].quantile(0.75)
        high_ca = df_transformed['ca'].quantile(0.75)
        
        df_transformed['is_high_risk'] = (
            (df_transformed['oldpeak'] > high_oldpeak) & 
            (df_transformed['ca'] > high_ca)
        ).astype(int)
        logger.info("Created feature: is_high_risk")

    # 4. Metabolic Indicator (Interaction)
    # Interaction between Cholesterol and Fasting Blood Sugar (if fbs_1.0 exists)
    if 'chol' in df_transformed.columns and 'fbs_1.0' in df_transformed.columns:
        df_transformed['chol_fbs_interaction'] = df_transformed['chol'] * df_transformed['fbs_1.0']
        logger.info("Created feature: chol_fbs_interaction")

    logger.info("Feature engineering completed.")
    return df_transformed

def main():
    """
    Main function to run the data transformation and storage pipeline.
    """
    logger.info("Starting Data Transformation and Storage Pipeline...")

    try:
        logger.info(f"Loading prepared data from {prepared_csv_file}...")
        if not os.path.exists(prepared_csv_file):
            raise FileNotFoundError(f"The prepared data file was not found at {prepared_csv_file}. Please run the data preparation script first.")
        df_prepared = pd.read_csv(prepared_csv_file)
        logger.info("Prepared data loaded successfully.")

        df_transformed = perform_feature_engineering(df_prepared)
        
        logger.info(f"Saving transformed data to {transformed_csv_file}...")
        df_transformed.to_csv(transformed_csv_file, index=False)
        logger.info("Transformed data saved successfully.")

        print(f"\nDATA TRANSFORMATION COMPLETED SUCCESSFULLY. Saved to: {transformed_csv_file}")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        print(f"\nError: {e}")
        sys.exit(1)
    except CustomException as e:
        logger.error(f"A pipeline error occurred: {e}")
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
