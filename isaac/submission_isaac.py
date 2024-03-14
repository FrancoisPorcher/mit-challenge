"""
    This is the main script that will create the predictions on test data and save 
    a predictions file.
"""
import time
from pathlib import Path
import pickle

import utils_isaac as utils


start_time = time.time()

# INPUT/OUTPUT PATHS WITHIN THE DOCKER CONTAINER
TRAINED_MODEL_DIR = Path('/trained_model/')
TEST_DATA_DIR = Path('/dataset/test/')
TEST_PREDS_FP = Path('/submission/submission.csv')


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
le_EW = pickle.load(open(TRAINED_MODEL_DIR / 'le_EW.pkl', 'rb'))
le_NS = pickle.load(open(TRAINED_MODEL_DIR / 'le_NS.pkl', 'rb'))

# Make predictions on the test data for EW
test_data['Predicted_EW'] = le_EW.inverse_transform(
    model_EW.predict(test_data[model_EW.feature_names_])
)

# Make predictions on the test data for NS
test_data['Predicted_NS'] = le_NS.inverse_transform(
    model_NS.predict(test_data[model_NS.feature_names_])
)

test_data['Predicted_EW'] = test_data['Predicted_EW'].mask(
    test_data['Predicted_EW'] == 'Nothing').ffill()
test_data['Predicted_NS'] = test_data['Predicted_NS'].mask(
    test_data['Predicted_NS'] == 'Nothing').ffill()

# Print the first few rows of the test data with predictions for both EW and NS
test_results = utils.convert_classifier_output(test_data)

# Save the test results to a csv file to be submitted to the challenge
test_results.to_csv(TEST_PREDS_FP, index=False)
print("Saved predictions to: {}".format(TEST_PREDS_FP))
time.sleep(360)  # TEMPORARY FIX TO OVERCOME EVALAI BUG
