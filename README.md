# ner-linking-demo
Demo app that performs both the detection and normalization of different clinical entities found in an input text. The detection step is carried out following a named entity recognition (NER) approach, while the normalization phase is performed following an entity linking (EL) strategy.

# Starting the Application

1. **Starting the Containers**:
   Navigate to the root directory and execute the following command to build and run the containers for the two models (NER and linking models):
docker-compose up

2. **Starting the Backend**:
Open a new terminal window and run the backend using the provided shell script:
./start_backend.sh


3. **Starting the Frontend**:
In another terminal window, execute the following command to start the frontend:
./start_frontend.sh


   

