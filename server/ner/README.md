Most of the code implemented in [scripts](https://github.com/guilopgar/ner-linking-demo/tree/main/server/ner/scripts) and [src](https://github.com/guilopgar/ner-linking-demo/tree/main/server/ner/src) folders was reused from the [icb-transformers](https://github.com/guilopgar/icb-transformers/tree/main) original repository.

### Setup and Execution Instructions
## 1. Detecting the mentions
*TODO: this command should be executed everytime the client sends a POST request to the server*
   - We need to provide the script the path of the shared data folder (and optionally other arguments):

         python scripts/mentions_detection.py -d ../data
