FROM python:3.9

WORKDIR /ignis_face_api

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./app ./app
COPY ./models ./models

CMD [ "python", "./app/main.py" ]