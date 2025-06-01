FROM python:3.8

WORKDIR /app

COPY ./requirements.txt /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt update
RUN apt install -y libgl-dev

COPY ./sd-v1-5-inpainting.ckpt /app/sd-v1-5-inpainting.ckpt
COPY ./sam2/ /app/sam2
COPY ./runway-stable-diffusion-inpainting /app/runway-stable-diffusion-inpainting

COPY . /app

ENTRYPOINT ["streamlit", "run", "app.py"]
EXPOSE 8501
