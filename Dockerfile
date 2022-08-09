FROM python:3.9

WORKDIR /ignis_face_api

COPY requirements.txt .

RUN apt-get update && \
    apt-get install libgl1-mesa-glx -y 
RUN pip install -r requirements.txt

COPY ./app ./app
COPY ./models ./models

CMD [ "python", "./app/main.py" ]