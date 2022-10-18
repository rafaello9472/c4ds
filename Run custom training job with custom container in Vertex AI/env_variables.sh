export PROJECT_ID=$(gcloud config list project --format "value(core.project)") 
export REPO_NAME=quickstart-docker-repo 
export IMAGE_NAME=stroke_model 
export IMAGE_TAG=latest 
export IMAGE_URI=europe-west1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}