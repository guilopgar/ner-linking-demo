FROM python:3.8.17-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app (only add what you need)
ADD ./src /app/src
ADD ./requirements.txt /app

ADD ./src/utils /app/utils


RUN pip install  -r requirements.txt


EXPOSE 82


# Run script.py when the container launches
#CMD ["python", "src/distemist/scripts/LoadData.py"]
#CMD ["python", "src/distemist/scripts/Inference-NER_Transformers_Test-Load_Weights-CPU.py"]