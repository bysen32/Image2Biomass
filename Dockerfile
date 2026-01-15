FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y git libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt
COPY . /app

WORKDIR /app

# CMD ["python", "train.py"]