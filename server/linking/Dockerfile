FROM python:3.8.2


COPY ./requirements.txt /app/requirements.txt
WORKDIR /app

RUN pip install -r requirements.txt

COPY ./scripts /app/scripts
COPY ./src /app/src
COPY ./models /app/models
COPY ./app.py /app
COPY ./tasks.py /app


# Install Supervisor
RUN apt-get update && apt-get install -y redis-server
RUN apt-get update && apt-get install -y supervisor

# COPY Supervisor configuration file
COPY ./supervisord.conf /etc/supervisor/conf.d/supervisord.conf

CMD ["/usr/bin/supervisord"]

EXPOSE 8080
