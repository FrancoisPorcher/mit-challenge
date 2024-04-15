
import utils
import pickle
import sys
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from fastcore.basics import Path, AttrDict
from catboost import CatBoostClassifier


sys.path.append('../')

config = AttrDict(
    challenge_data_dir=Path(
        r'C:\Users\isaac\Documents\Challenge_Francois\splid-devkit\dataset'),
    valid_ratio=0.00001,
    lag_steps=6,
    tolerance=6,  # Default evaluation tolerance
    n_estimators=100
)

# Define the list of feature columns
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

# Define the directory paths
train_data_dir = config.challenge_data_dir / "train"

# Load the ground truth data
ground_truth = pd.read_csv(config.challenge_data_dir / 'train_labels.csv')

# # Apply the function to the ground truth data
data, updated_feature_cols = utils.tabularize_data(train_data_dir,
                                                   feature_cols,
                                                   ground_truth,
                                                   lag_steps=config.lag_steps,
                                                   add_heurestic=True,
                                                   nb_of_ex=20)

data['EW'] = data['EW'].fillna('Nothing')
data['NS'] = data['NS'].fillna('Nothing')

updated_feature_cols = list(data.columns)
updated_feature_cols.remove('TimeIndex')
updated_feature_cols.remove('Timestamp')
updated_feature_cols.remove('ObjectID')
updated_feature_cols.remove('EW')
updated_feature_cols.remove('NS')
updated_feature_cols.remove('EW_baseline_heuristic')
updated_feature_cols.remove('NS_baseline_heuristic')
updated_feature_cols.remove('EW_baseline_heuristic_ffill')
updated_feature_cols.remove('NS_baseline_heuristic_ffill')

# Create a validation set without mixing the ObjectIDs


print('Number of objects in the training set:',
      len(data['ObjectID'].unique()))


# Convert categorical data to numerical data
le_EW = LabelEncoder()
le_NS = LabelEncoder()

# Encode the 'EW' and 'NS' columns
data['EW_encoded'] = le_EW.fit_transform(data['EW'])
data['NS_encoded'] = le_NS.fit_transform(data['NS'])

# Define the Random Forest model for NS
model_NS_preprocess = CatBoostClassifier(
    n_estimators=config.n_estimators, random_state=42)
# Fit the model to the training data for NS
model_NS_preprocess.fit(
    data[updated_feature_cols], data['NS_encoded'])

added_proba_feature_NS = pd.DataFrame(model_NS_preprocess.predict_proba(
    data[model_NS_preprocess.feature_names_])).add_prefix('proba_feature_NS_')
added_proba_feature_NS.index = data.index
data = pd.concat([data, added_proba_feature_NS], axis=1)

# Define the Random Forest model for EW
model_EW = CatBoostClassifier(
    n_estimators=config.n_estimators, random_state=42)
# Fit the model to the training data for EW
model_EW.fit(data[updated_feature_cols +
             list(added_proba_feature_NS.columns)], data['EW_encoded'])

added_proba_feature_EW = pd.DataFrame(model_EW.predict_proba(
    data[model_EW.feature_names_])).add_prefix('proba_feature_EW_')
added_proba_feature_EW.index = data.index
data = pd.concat([data, added_proba_feature_EW], axis=1)


# Define the Random Forest model for NS
model_NS = CatBoostClassifier(
    n_estimators=config.n_estimators, random_state=42)
# Fit the model to the training data for NS
model_NS.fit(data[updated_feature_cols +
             list(added_proba_feature_EW.columns)], data['NS_encoded'])

Path('trained_model').mkdir(exist_ok=True)
pickle.dump(model_EW, open('trained_model/model_EW.pkl', 'wb'))
pickle.dump(model_NS, open('trained_model/model_NS.pkl', 'wb'))
pickle.dump(model_NS_preprocess, open(
    'trained_model/model_NS_preprocess.pkl', 'wb'))
pickle.dump(le_EW, open('trained_model/le_EW.pkl', 'wb'))
pickle.dump(le_NS, open('trained_model/le_NS.pkl', 'wb'))
