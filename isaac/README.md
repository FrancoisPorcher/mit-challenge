# Satellite Pattern-of-Life Identification Challenge Starter Toolkit

## ML baseline
The ML Model train notebook `ml_baseline.ipynb` provides a foundational model using a Random Forest Classifier 
aimed at node detection. This notebook acts as a quick start guide and establishes a 
performance baseline for the challenge. It uses a predefined list of features for 
training the model.

**Configuration Parameters**:  
- `challenge_dir`: Directory containing the challenge data.
- `valid_ratio`: Proportion of the dataset to be used for validation.
- `lag_steps`: Number of lag steps for the model.

Once you run the notebook and the model is trained, you can follow the following steps to build and test your Docker submission:
- Build docker image for submission: `docker build -t splid-submission .`

- Test submission docker on a toy test dataset:
```
docker run -v [[TOY_TEST_DATASET_DIR]]:/dataset -v $(pwd)/submission:/submission splid-submission`
```

> Note: If your submission needs the use of GPUs to generate
the predictions, add the flag `--gpus all` to the `docker run` instruction.
