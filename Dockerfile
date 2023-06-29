FROM python:3.8

RUN apt-get update && apt-get install -y libsndfile1

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
