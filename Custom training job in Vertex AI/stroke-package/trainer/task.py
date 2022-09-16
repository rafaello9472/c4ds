# Import libraries
import os, joblib, logging, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Setup logging and parser
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()


# Input Arguments
parser.add_argument(
    '--data_gcs_path',
    help = 'Dataset file on Google Cloud Storage',
    type = str
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
print('Training a decision tree model...')
model = DecisionTreeClassifier().fit(X_train, y_train)

# # calculate accuracy
# y_hat = model.predict(X_test)
# acc = np.average(y_hat == y_test)
# print('Accuracy:', acc)
# # run.log('Accuracy', np.float(acc))

# # calculate AUC
# y_scores = model.predict_proba(X_test)
# auc = roc_auc_score(y_test,y_scores[:,1])
# print('AUC: ' + str(auc))
# # run.log('AUC', np.float(auc))

# # plot ROC curve
# fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
# fig = plt.figure(figsize=(6, 4))
# # Plot the diagonal 50% line
# plt.plot([0, 1], [0, 1], 'k--')
# # Plot the FPR and TPR achieved by our model
# plt.plot(fpr, tpr)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# # run.log_image(name = "ROC", plot = fig)
# plt.show()

artifact_filename = 'model.joblib'

# Save model artifact to local filesystem (doesn't persist)
local_path = artifact_filename
joblib.dump(model, local_path)

# Upload model artifact to Cloud Storage
model_directory = os.environ['AIP_MODEL_DIR']
storage_path = os.path.join(model_directory, artifact_filename)
blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
blob.upload_from_filename(local_path)
logging.info("model exported to : {}".format(storage_path))