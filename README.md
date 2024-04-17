# A simple model and advanced strategies are all you need

Representing Team QR_IS, we achieved fourth place in Phase I of the competition https://eval.ai/web/challenges/challenge-page/2164/overview.
In this repository, we provide all the code needed to reproduce results.
## Runing code
### Prerequisites
A python environement that contains these packages:
- pandas ==  2.2.1
- numpy == 1.26.4
- fastcore ==  1.5.29 
- scikit-learn==1.3.2
- matplotlib == 3.8.3
- catboost == 1.2.3


### Training
The python file main_train.py is able to construct the model aimed at node detection.
It will read all csv files containing satellite trajectories and the ground truth csv and will train the model.
One argument need to be provided:

- `--path_to_dataset`: Directory containing the training data and train_labels.csv.

Example: python main_train.py --path_to_dataset C:\Users\isaac\Documents\Challenge_Francois\splid-devkit\dataset

### Test
The python file main_submission.py is able to read a trained model and use it on new data.
Two arguments need to be provided:

- `--path_to_dataset`: Directory containing the training data and train_labels.csv.
- `--path_to_model`: Directory containing the models created during train

Example: python main_submission.py --path_to_dataset C:\Users\isaac\Documents\Challenge_Francois\splid-devkit\dataset\test --path_to_model C:\Users\isaac\Documents\Challenge_Francois\mit-challenge\trained_model_for_submission

You can use the trained model that trained_model_for_submission folder

## Presentation of the code

If we follow the description that we provide in the technical report, let focus on four steps that are necessary to obtain a trained model.
1. Features engineering: Done in the python file utils.py with the tabularize_data function
2. Model Separation and Recombination Strategy: 
    - Gradient Boosting model initialization is done in the python file main_train.py
    - Model 1, Model 2 and Model 3 are sequentially trained in the python file main_train.py
3. Model stacking: Add Heuristic model is done in the python file utils.py within the tabularize_data function with add_baseline_heuristic
4. Model prediction strategy: Dual Threshold mechanism is done using dual_threshold_prediction function in utils.py 

### Features engineering

During this step, we create a lot of features that will have a high importance to have a performing model. Going from the raw data $r_i$ (where $i$ is the timestamp index), the feature $f_i$ is created using a transformation selected using intuition, visualization, experimentation and statistics. Raw data can be one of the 15 measures provided for all trajectory. 
Here is an exhaustive list of the 15 measures: 
- Eccentricity
- Semimajor Axis (m)
- Inclination (deg)
- RAAN (deg)
- Argument of Periapsis (deg)
- True Anomaly (deg)
- Latitude (deg)
- Longitude (deg)
- Altitude (m)
- X (m)
- Y (m)
- Z (m)
- Vx (m/s)
- Vy (m/s)
- Vz (m/s) 

Here is an exhaustive list of the trajectory-based feature *(257 features)*:
1. Lag: $f_i = r_{i-j}$ for $j \in {-6,-5,-4,-3,-2,-1,1,2,3,4,5,6}$ for all 15 measures ($15 \times 12 = 180$ *features*)
2. Difference: $f_i = r_i - r_{i-j}$ for $j \in {-2,-1,1,2}$ for Eccentricity, Semimajor Axis (m), Inclination (deg), RAAN (deg),Argument of Periapsis (deg), True Anomaly (deg),  Latitude (deg), Longitude (deg), Altitude (m) ($9 \times 4 = 36$ *features*)
3. Percent change: $f_i = (r_i - r_{i-j})/r_{i-j}$ for $j \in {-1,1}$ for Vx (m/s), Vy (m/s), Vz (m/s) ($3 \times 2 = 6$ *features*)
4. Rolling average: $f_i = average(r_{i+6},..., r_{i-6})$ for Eccentricity,  Semimajor Axis (m), Inclination (deg),  RAAN (deg), Argument of Periapsis (deg), True Anomaly (deg), Altitude (m), Vx (m/s),  Vy (m/s), Vz (m/s) *(10 features)*
5. Rolling average: $f_i = std(r_{i+6},..., r_{i-6})$ for Eccentricity,  Semimajor Axis (m), Inclination (deg),  RAAN (deg), Argument of Periapsis (deg), True Anomaly (deg), Altitude (m), Vx (m/s),  Vy (m/s), Vz (m/s) *(10 features)*
6. Envelope:  $f_i = max(r_{i+18},..., r_{i-18}) - min(r_{i+18},..., r_{i-18})$ for Latitude (deg), Longitude (deg),  Altitude (m) *(3 features)*
7. Envelope difference: for the 3 enveloppe features, compute the difference transformation ($3\times 4 = 12$ *features*)

Here is an exhaustive list of the saltelitte characterized feature *(16 features)*:
1. Average on all the trajectory for  features: Eccentricity, Semimajor Axis (m), Inclination (deg), RAAN (deg), Argument of Periapsis (deg), True Anomaly (deg), Altitude (m)
2. Std on all the trajectory for  features: X (m), Y (m), Z (m), Vx (m/s),  Vy (m/s), Vz (m/s), Eccentricity, Altitude (m), Inclination (deg)