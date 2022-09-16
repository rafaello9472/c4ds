# 1. Base image
FROM python:slim-buster
WORKDIR /root

# 2. Copy files
COPY c4ds/Custom training job in Vertex AI/Custom container/task.py /root/task.py
COPY c4ds/Custom training job in Vertex AI/Custom container/requirements.txt /root/requirements.txt

# 3. Install dependencies
RUN pip install -r /root/requirements.txt


# 4. Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "task.py"]
