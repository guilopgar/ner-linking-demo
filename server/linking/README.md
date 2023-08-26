Most of the code implemented in [scripts](https://github.com/guilopgar/ner-linking-demo/tree/main/server/linking/scripts) and [src](https://github.com/guilopgar/ner-linking-demo/tree/main/server/linking/src) folders was reused from the [deepspanorm](https://github.com/luisgasco/deepspanorm/tree/main) original repository.

### Setup and Execution Instructions
## 1. Processing the gazetteers
*TODO: these two commands should be executed at the beginning, when building the image*

  - For each clinical entity type to be normalized, a gazetteer needs to be loaded, pre-processed and saved. For instance, to process the gazetteer corresponding to both the "disease" and "procedure" types, the following commands must be executed:
    
        python scripts/gazetteer_creation.py -g disease
        python scripts/gazetteer_creation.py -g procedure
  
  
## 2. Normalizing the mentions
*TODO: this command should be executed everytime the client sends a POST request to the server*
   - We need to provide the script the path of the table containing the mentions to be normalized, as well as the number of candidate codes to be predicted by the model:

         python scripts/mentions_normalization.py -p ../data/mentions.tsv -k 3
