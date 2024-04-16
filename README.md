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

## Presentation of the code

If we follow the description that we provide in the technical report, let focus on four steps that are necessary to obtain a trained model.
1. Features engineering: Done in the python file utils.py with the tabularize_data function
2. Model Separation and Recombination Strategy: 
    - Gradient Boosting model initialization is done in the python file main_train.py
    - Model 1, Model 2 and Model 3 are sequentially trained in the python file main_train.py
3. Model stacking: Add Heuristic model is done in the python file utils.py within the tabularize_data function with add_baseline_heuristic
4. Model prediction strategy: Dual Threshold mechanism is done using dual_threshold_prediction function in utils.py 

### Features engineering

During this step, we create a lot of features that will have a high importance


1. Envelope: "Latitude (deg)", "Longitude (deg)",  "Altitude (m)"

$$
x = \frac{{-b \pm \sqrt{{b^2 - 4ac}}}}{{2a}}
$$