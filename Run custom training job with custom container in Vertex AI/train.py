# Import libraries
import os, joblib, logging, argparse
import pandas as pd
import numpy as np
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Setup logging and parser
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()

# Input Arguments
parser.add_argument(
    '--data_gcs_path',
    help = 'Dataset file on Google Cloud Storage',
    type = str
)

parser.add_argument(
    '--model_dir',
    help = 'Directory to output model artifacts',
    type = str,
    default = os.environ['AIP_MODEL_DIR'] if 'AIP_MODEL_DIR' in os.environ else ""
)

# Parse arguments
args = parser.parse_args()
arguments = args.__dict__

# Get dataset from GCS
# data_gcs_path = 'gs://datasets-c4ds/healthcare-dataset-stroke-data.csv'
data_gcs_path = arguments['data_gcs_path']
df = pd.read_csv(data_gcs_path)
logging.info("reading gs data: {}".format(data_gcs_path))

# Save categorical column names
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'] 

# One hot encode categorical columns
df = pd.get_dummies(df, columns = categorical_cols)

# Replace NaNs with mean
df = df.fillna(df.mean())

# Separate features and labels
X, y = df.drop(columns=['id', 'stroke']), df['stroke'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Train a decision tree model
print('Training a LightGBM model...')
model = lgb.LGBMClassifier().fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
logging.info('Accuracy: {}'.format(acc))

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
logging.info('AUC: {}'.format(auc))

# Define model name
artifact_filename = 'model.joblib'

# Save model artifact to local filesystem (doesn't persist)
local_path = artifact_filename
joblib.dump(model, local_path)

# Upload model artifact to Cloud Storage
model_directory = arguments['model_dir']
if model_directory == "":
    print("Training is run locally - skipping model saving to GCS.")
else:
    storage_path = os.path.join(model_directory, artifact_filename)
    blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
    blob.upload_from_filename(local_path)
    logging.info("model exported to : {}".format(storage_path))
