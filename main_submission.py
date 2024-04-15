"""
    This is the main script that will create the predictions on test data and save 
    a predictions file.
"""
import time
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import utils_isaac as utils


start_time = time.time()

# INPUT/OUTPUT PATHS WITHIN THE DOCKER CONTAINER
TRAINED_MODEL_DIR = Path('/trained_model/')
TEST_DATA_DIR = Path('/dataset/test/')
TEST_PREDS_FP = Path('/submission/submission.csv')


def do_prediction(model, data, thresh, thresh_add):
    def top_k(x, k):
        ind = np.argpartition(x, -1*k)[-1*k:]
        return ind[np.argsort(x[ind])]

    def top_k_values(x, k):
        ind = np.argpartition(x, -1*k)[-1*k:]
        return x[ind][np.argsort(x[ind])]

    pred_proba = pd.DataFrame(model.predict(
        data, prediction_type='Probability'))
    pred = pred_proba.idxmax(1)
    print('Num of ex to cut', sum(pred_proba.max(1) < thresh))
    nothing_index = pred.value_counts().index[0]
    pred.loc[pred_proba.max(1) < thresh] = nothing_index

    top_proba = pd.DataFrame(np.apply_along_axis(
        lambda x: top_k(x, 2), 1, pred_proba.to_numpy()))
    top_proba_values = pd.DataFrame(np.apply_along_axis(
        lambda x: top_k_values(x, 2), 1, pred_proba.to_numpy()))
    # print(top_proba)
    # print(pred_proba*100)
    # print(top_proba_values*100)
    compt = 0
    for i in range(len(top_proba)):
        if top_proba.iloc[i, 1] == nothing_index:
            if top_proba_values.iloc[i, 0] > thresh_add:
                compt += 1
                pred.iloc[i] = top_proba.iloc[i, 0]
    print(compt)
    pred = pred.to_numpy().reshape(-1, 1)
    return pred


threshold_ew = 0.1
threshold_ns = 0.1
tresh_add_ew = 0.11
tresh_add_ns = 0.26

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
    do_prediction(
        model_EW, test_data[model_EW.feature_names_], threshold_ew, tresh_add_ew)
)

added_proba_feature_EW = pd.DataFrame(model_EW.predict_proba(
    test_data[updated_feature_cols+list(added_proba_feature_NS.columns)])).add_prefix('proba_feature_EW_')
added_proba_feature_EW.index = test_data.index
test_data = pd.concat([test_data, added_proba_feature_EW], axis=1)


# Make predictions on the test data for NS
test_data['Predicted_NS'] = le_NS.inverse_transform(
    do_prediction(
        model_NS, test_data[model_NS.feature_names_], threshold_ns, tresh_add_ns)
)

test_data['Predicted_EW'] = test_data['Predicted_EW'].mask(
    test_data['Predicted_EW'] == 'Nothing').ffill()
test_data['Predicted_NS'] = test_data['Predicted_NS'].mask(
    test_data['Predicted_NS'] == 'Nothing').ffill()

# Print the first few rows of the test data with predictions for both EW and NS
test_results = utils.convert_classifier_output(test_data)
test_results.loc[test_results.TimeIndex == 0, 'Node'] = 'SS'

# Save the test results to a csv file to be submitted to the challenge
test_results.to_csv(TEST_PREDS_FP, index=False)
print("Saved predictions to: {}".format(TEST_PREDS_FP))
time.sleep(360)  # TEMPORARY FIX TO OVERCOME EVALAI BUG
