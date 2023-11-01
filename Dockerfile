# Use an official Python runtime as a parent image
FROM python:3.9.7

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./src /app

RUN pip install -r requirements.txt
RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN chmod +x run.sh

ENTRYPOINT ["./run.sh"]
