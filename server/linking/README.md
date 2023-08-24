Most of the code implemented in [scripts](https://github.com/guilopgar/ner-linking-demo/tree/main/server/linking/scripts) and [src](https://github.com/guilopgar/ner-linking-demo/tree/main/server/linking/src) folders was reused from the [deepspanorm](https://github.com/luisgasco/deepspanorm/tree/main) original repository.

### Setup and Execution Instructions
## 1. Processing the gazetteers
  - For each clinical entity type to be normalized, a gazetteer needs to be loaded, pre-processed and saved. For instance, to process the gazetteer corresponding to the "enfermedad" type, the following command must be executed:
    
        python scripts/gazetteer_creation.py -g enfermedad
  
  
## 2. Normalizing the mentions
   - We need to provide the script the path of the table containing the mentions to be normalized, as well as the number of candidate codes to be predicted by the model:

         python scripts/mentions_normalization.py -p ../data/example_annotations.tsv -k 3
