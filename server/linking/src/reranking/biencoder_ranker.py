import os, sys, torch
import numpy as np 
import pandas as pd
import os, sys
from sklearn.model_selection import train_test_split
from sentence_transformers import models, SentenceTransformer, util
from sentence_transformers import SentencesDataset
from torch.utils.data import DataLoader
from sentence_transformers.losses import TripletLoss
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.readers import TripletReader

import datetime

general_path = os.getcwd().split("deepspanorm")[0]+"deepspanorm/"
sys.path.append(general_path+'src/')
from utils.data_structures import predictions

class BiEncoderReranker:
    def __init__(self, model_name:str, model_type = "mask", max_seq_length:int = 256) -> None:
        # When reranking I always use a model trained before (or the CLS version), i don't need the indexer.
        self.load_lm(model_type, model_name,max_seq_length)
            
                
    def load_lm(self, model_type, model_name, max_seq_length ):
        if model_type == "mask":
            word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            self.lm_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            print("Use .fit() method to train your own sentente-transformer with TripletLoss")
        elif model_type == "st":
            self.lm_model = SentenceTransformer(model_name)
        
        # Check if GPU is available and use it
        if torch.cuda.is_available():
            self.lm_model = self.lm_model.to(torch.device("cuda"))
        print(self.lm_model.device)
        
    def train(self, triplet_file_name, model_path,tmp_folder, batch_size, epochs, 
                optimizer_parameters = {"lr":1e-5}, weight_decay = 0.01, save_best_model  = True,
                test_size=0.3):
        """
        data_reader needs to have the same path than temp_folder
        """
        ## Get Data
        # Read dataframe. 
        file_df = pd.read_csv(triplet_file_name, sep="\t", header=None)
        # Stratify based on column 0 (the anchor)
        train_df, test_df = train_test_split(file_df, test_size=test_size, stratify=file_df[[0]])
        ct = datetime.datetime.now()
        # Save training and val temp files.
        temp_train_path = tmp_folder+"livingner_bienc_train_df_"+str(ct.toordinal())+".tsv"
        temp_test_path = tmp_folder+"livingner_bienc_valid_df_"+str(ct.toordinal())+".tsv"
        train_df.to_csv(temp_train_path,sep="\t",index=False,header =None)
        test_df.to_csv(temp_test_path,sep="\t",index=False,header =None)
        # Create data_reader
        data_reader = TripletReader(tmp_folder)
        # Read training and validation triplets from those files
        train_triplets = data_reader.get_examples(temp_train_path)
        test_triplets = data_reader.get_examples(temp_test_path)
        # Continue the process
        train_data = SentencesDataset(train_triplets, model=self.lm_model)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=1)
        train_loss = TripletLoss(model=self.lm_model)
        warmup_steps = int(len(train_triplets) / batch_size * epochs * 0.1)
        
        # Train model
        self.lm_model.fit(train_objectives=[(train_dataloader, train_loss)],
                           evaluator=TripletEvaluator.from_input_examples(test_triplets),
                           epochs=epochs,
                           optimizer_params = optimizer_parameters,
                           weight_decay = weight_decay,
                           evaluation_steps=int(len(train_triplets) / batch_size),
                           warmup_steps=warmup_steps,
                           save_best_model = save_best_model,
                           output_path=model_path
                           )
        
        
    def rerank_candidates(self, entity_texts, candidates, sim_measure = "cos"):
        # Candidates follows the datastructure generate at candidate_generataion step
        can_codes, _, can_texts = candidates # Ignore previous scores
        
        # output lists
        reranked_codes = list()
        reranked_texts = list()
        reranked_scores = list()
        
        # Iterate over entity
        for candidates_codes, candidates_texts, entidad in zip(can_codes, can_texts, entity_texts):
            candidates_texts_embeddings = self.lm_model.encode( sentences = candidates_texts)
            entity_embedding = self.lm_model.encode( sentences = entidad)
            # Medir similitud dos maneras diferentes.
            if sim_measure == "cos":
                candidates_sim_scores = util.cos_sim(entity_embedding, candidates_texts_embeddings)
            elif sim_measure == "dot":
                candidates_sim_scores = util.dot_score(entity_embedding, candidates_texts_embeddings)

            # Order results based on candidates_sim_scores
            candidates_codes = np.array(candidates_codes)
            candidates_texts = np.array(candidates_texts)
            candidates_scores = np.array(candidates_sim_scores)
            
            inds = (-candidates_scores).argsort()
            candidates_codes_sorted = candidates_codes[inds]
            candidates_texts_sorted = candidates_texts[inds]
            candidates_scores_sorted = -np.sort(-candidates_scores)[::-1]
            
            reranked_codes.append(candidates_codes_sorted.tolist())
            reranked_texts.append(candidates_texts_sorted.tolist())
            reranked_scores.append(candidates_scores_sorted.tolist())
            # Preparamos datos para ser guardados en variable predictions:
        self.candidates = predictions(reranked_codes, reranked_scores, reranked_texts)
    
        return reranked_texts