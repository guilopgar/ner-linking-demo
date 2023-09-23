FROM python:3.8.17-buster

# Set the working directory in the container to /app
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app


# Add the current directory contents into the container at /app (only add what you need)
RUN pip install  -r requirements.txt

COPY ./src /app/src
COPY ./models /app/models
COPY ./scripts /app/scripts
COPY ./app.py /app

COPY ./src/utils /app/utils


CMD ["python" , "app.py"] 

EXPOSE 5000

# Run script.py when the container launches
#CMD ["python", "src/distemist/scripts/LoadData.py"]
#CMD ["python", "src/distemist/scripts/Inference-NER_Transformers_Test-Load_Weights-CPU.py"]