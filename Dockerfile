FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y git

RUN pip install -r requirements.txt

COPY . /app

WORKDIR /app

CMD ["python", "train.py"]