# 1. Base image
FROM python:3.9-buster

# 2. Specify directory where all subsequent instructions are run
WORKDIR /root

# 3. Copy files
COPY train.py /root/train.py
COPY requirements.txt /root/requirements.txt

# 4. Install dependencies
RUN pip install -r /root/requirements.txt

# 5. Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "train.py"]
