FROM python:3.8

WORKDIR /app

COPY ./requirements.txt /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt update
RUN apt install -y libgl-dev

COPY . /app

ENTRYPOINT ["streamlit", "run", "app.py"]
EXPOSE 8501
