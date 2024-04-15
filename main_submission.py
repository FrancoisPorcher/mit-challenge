"""
    This is the main script that will create the predictions on test data and save 
    a predictions file.
"""

import warnings
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import utils
import argparse


def main_submission(path_to_dataset, path_to_model):

    # INPUT/OUTPUT PATHS WITHIN THE DOCKER CONTAINER
    TRAINED_MODEL_DIR = Path(path_to_model)
    TEST_DATA_DIR = Path(path_to_dataset)
    # TEST_PREDS_FP = Path('/submission/submission.csv')

    precision_thresh_ew = 0.1
    precision_thresh_ns = 0.1
    recall_thresh_ew = 0.11
    recall_thresh_ns = 0.26

    # Rest of configuration, specific to this submission
    feature_cols = [
        "Eccentricity",
        "Semimajor Axis (m)",
        "Inclination (deg)",
        "RAAN (deg)",
        "Argument of Periapsis (deg)",
        "True Anomaly (deg)",
        "Latitude (deg)",
        "Longitude (deg)",
        "Altitude (m)",
        "X (m)",
        "Y (m)",
        "Z (m)",
        "Vx (m/s)",
        "Vy (m/s)",
        "Vz (m/s)"
    ]

    lag_steps = 6

    test_data, updated_feature_cols = utils.tabularize_data(
        TEST_DATA_DIR, feature_cols, lag_steps=lag_steps, add_heurestic=True)

    # Load the trained models (don't use the utils module, use pickle)
    model_EW = pickle.load(open(TRAINED_MODEL_DIR / 'model_EW.pkl', 'rb'))
    model_NS = pickle.load(open(TRAINED_MODEL_DIR / 'model_NS.pkl', 'rb'))
    model_NS_preprocess = pickle.load(
        open(TRAINED_MODEL_DIR / 'model_NS_preprocess.pkl', 'rb'))
    le_EW = pickle.load(open(TRAINED_MODEL_DIR / 'le_EW.pkl', 'rb'))
    le_NS = pickle.load(open(TRAINED_MODEL_DIR / 'le_NS.pkl', 'rb'))

    added_proba_feature_NS = pd.DataFrame(model_NS_preprocess.predict_proba(
        test_data[updated_feature_cols])).add_prefix('proba_feature_NS_')
    added_proba_feature_NS.index = test_data.index
    test_data = pd.concat([test_data, added_proba_feature_NS], axis=1)

    # Make predictions on the test data for EW
    test_data['Predicted_EW'] = le_EW.inverse_transform(
        utils.dual_threshold_prediction(
            model_EW, test_data[model_EW.feature_names_], precision_thresh_ew, recall_thresh_ew)
    )

    added_proba_feature_EW = pd.DataFrame(model_EW.predict_proba(
        test_data[updated_feature_cols+list(added_proba_feature_NS.columns)])).add_prefix('proba_feature_EW_')
    added_proba_feature_EW.index = test_data.index
    test_data = pd.concat([test_data, added_proba_feature_EW], axis=1)

    # Make predictions on the test data for NS
    test_data['Predicted_NS'] = le_NS.inverse_transform(
        utils.dual_threshold_prediction(
            model_NS, test_data[model_NS.feature_names_], precision_thresh_ns, recall_thresh_ns)
    )

    test_data['Predicted_EW'] = test_data['Predicted_EW'].mask(
        test_data['Predicted_EW'] == 'Nothing').ffill()
    test_data['Predicted_NS'] = test_data['Predicted_NS'].mask(
        test_data['Predicted_NS'] == 'Nothing').ffill()

    # Print the first few rows of the test data with predictions for both EW and NS
    test_results = utils.convert_classifier_output(test_data)
    test_results.loc[test_results.TimeIndex == 0, 'Node'] = 'SS'

    # Save the test results to a csv file to be submitted to the challenge
    test_results.to_csv(
        TEST_DATA_DIR/'../test_labels_evaluation.csv', index=False)
    print("Saved predictions to: {}".format(
        TEST_DATA_DIR/'../test_labels_evaluation.csv'))


if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Create argument parser
    parser = argparse.ArgumentParser(description='Run training on the data')

    # Add arguments
    parser.add_argument('--path_to_dataset', type=str,
                        help='Path to dataset')
    parser.add_argument('--path_to_model', type=str,
                        help='Path to model')

    # Parse arguments

    args = parser.parse_args()
    main_submission(args.path_to_dataset, args.path_to_model)
