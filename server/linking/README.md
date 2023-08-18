Most of the code implemented in [scripts](https://github.com/Lokmane-ELO/docker-deepspanorm/tree/main/scripts) and [src](https://github.com/Lokmane-ELO/docker-deepspanorm/tree/main/src) folders was reused from the [deepspanorm](https://github.com/luisgasco/deepspanorm/tree/main) original repository.

### Setup and Execution Instructions
## 1. Restore MongoDB Data
  - Unzip the bioMnorm.zip file to a directory:
    
        unzip database/bioMnorm.zip -d /path_to_directory_where_you_want_to_unzip/
  - Restore the MongoDB data using the mongorestore command:

        mongorestore -d bioMnorm /path_to_directory_where_you_unzipped/bioMnorm
  
## 2. Build and Run the Docker Container
   - Navigate to the directory containing the Dockerfile (root directory):

         cd /path_to_directory_containing_dockerfile/

   - Build the Docker image (replace *your_image_name* with a suitable name for your Docker image):
    
         docker build -t <your_image_name> .
     - The size of the image is ∼6GB, but it could be reduced by eliminating unused packages specified in the [requirements](https://github.com/Lokmane-ELO/docker-deepspanorm/blob/main/requirements.txt) file (which was reused from [here](https://github.com/luisgasco/deepspanorm/blob/main/requirements.txt))

   - Run the Docker container:

         docker run -it --gpus all --network host <your_image_name>
## 3. Send a POST Request
  Now, you can send a POST request to the container. If executed successfully, you'll receive the candidate codes in response:
  
    curl -X POST -H "Content-Type: application/json" -d '{"project_id": "testing_set_cantemist", "gazetteer":"enfermedad-distemist", "k_candidates":"3"}' http://localhost:8080/generate_candidates_by_project

  - Optionally, to check the logs of the previous execution, firstly, you need to access to the running container:

        docker exec -it <your_running_container_id> bash

  - Finally, you can inspect the logs by executing the following command:

        cat /var/log/celery.log



