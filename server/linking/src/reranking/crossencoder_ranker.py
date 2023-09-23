import os, sys, torch
import numpy as np 
import pandas as pd
import os, sys
from sklearn.model_selection import train_test_split
from sentence_transformers import models, SentenceTransformer, util
from sentence_transformers import SentencesDataset
from collections import OrderedDict
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.losses import TripletLoss
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator, CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample

general_path = os.getcwd().split("deepspanorm")[0]+"deepspanorm/"
sys.path.append(general_path+'src/')
from utils.data_structures import predictions

class CrossEncoderReranker:
    def __init__(self, model_name:str, model_type = "mask", max_seq_length:int = 256) -> None:
        # When reranking I always use a model trained before (or the CLS version), i don't need the indexer.
        self.load_lm(model_type, model_name,max_seq_length)
            
                
    def load_lm(self, model_type, model_name, max_seq_length ):
        if model_type == "mask":
            self.lm_model = CrossEncoder(model_name, max_length=max_seq_length, num_labels = 1)
            print("Use .fit() method to train your own sentente-transformer with TripletLoss")
        elif model_type == "st":
            self.lm_model = CrossEncoder(model_name)
        # Crossencoder uses GPU automaticalle if available 
    
    def read_triplets(self, path_to_triplets):
        # Read triplets
        triplets_df = pd.read_csv(path_to_triplets,sep="\t",header=None,skiprows=1)
        triplets_df.columns = ["anchor","positive","negative"]
        return triplets_df

    def prepare_triplets(self, triplets_df):
        # Prepare triplets in format for cross_encoder:
        # Positive
        positive_triplets = triplets_df[["anchor","positive"]].drop_duplicates().reset_index(drop=True)
        positive_triplets["label"] = 1
        positive_triplets.columns = ["anchor","descriptor","label"]
        # Negative
        negative_triplets = triplets_df[["anchor","negative"]].drop_duplicates().reset_index(drop=True)
        negative_triplets["label"] = 0
        negative_triplets.columns = ["anchor","descriptor","label"]
        # Join both.
        triplets_df_prepared = pd.concat([positive_triplets,negative_triplets])
        # PRepare input examples
        triplets_samples = list()
        for sentence1, sentence2, label_id in zip(triplets_df_prepared.anchor, triplets_df_prepared.descriptor, triplets_df_prepared.label):
            triplets_samples.append(InputExample(texts = [sentence1,sentence2], label = label_id))
            
        return triplets_samples

    def transform_triplets_rankingeval(self, df_triplets):
        # Create output dictionary
        dev_samples_dict = dict()

        # Iterate over each row of dataframe
        for sample,idx in zip(df_triplets,range(0,len(df_triplets))):
            # Check if text exists in the dictionary
            if sample.texts[0] in dev_samples_dict:
                # If exist and label is 1, include in the positive key, else in "negative"
                if sample.label == 1:
                    dev_samples_dict[sample.texts[0]]["positive"].add(sample.texts[1])
                elif sample.label == 0:
                    dev_samples_dict[sample.texts[0]]["negative"].add(sample.texts[1])
            else:
                # If the anchor does not exist, create a new dictionary field depending on label.
                if sample.label == 1:
                    dev_samples_dict[sample.texts[0]] = {
                                "query": sample.texts[0],
                                "positive":  set([sample.texts[1]]),
                                "negative": set(),
                            }
                elif sample.label == 0:
                    dev_samples_dict[sample.texts[0]] = {
                                "query": sample.texts[0],
                                "positive": set(),
                                "negative": set([sample.texts[1]]) ,
                            }
        # Transform to the appropiate output format needed for CERankingEvaluator
        dev_dict_list =  [{"query":dev_samples_dict[t]["query"],
            "positive":dev_samples_dict[t]["positive"],
            "negative":dev_samples_dict[t]["negative"]} for t in dev_samples_dict]

        return dev_dict_list

    def train(self, triplet_file_name, model_path, batch_size, epochs, evaluator_type = "BinaryClassificationEvaluator", 
            optimizer_parameters = {"lr":1e-5}, weight_decay = 0.01, evaluation_steps = 10000, save_best_model=True, test_size = 0.3):
        # Read triplets:
        df_triplets = self.read_triplets(triplet_file_name)
        # Divide triplets_samples with stratification based on anchor
        train_samples, dev_samples = train_test_split(df_triplets, test_size=test_size, stratify=df_triplets[['anchor']])
        # prepare triplets to train the crossencoder
        self.train_samples = self.prepare_triplets(train_samples)
        self.dev_samples = self.prepare_triplets(dev_samples)
        
        #We wrap train_samples, which is a list ot InputExample, in a pytorch DataLoader
        train_dataloader = DataLoader(self.train_samples, shuffle=True, batch_size=batch_size)
        
        # Evaluator selection
        if evaluator_type == "BinaryClassificationEvaluator":
            evaluator = CEBinaryClassificationEvaluator.from_input_examples(self.dev_samples, name='dev-binaryclass')
        elif evaluator_type == "RerankingEvaluator":
            self.dev_samples_dict = self.transform_triplets_rankingeval(self.dev_samples)
            evaluator = CERerankingEvaluator(self.dev_samples_dict, name='dev-ranking')

        warmup_steps = int(len(self.train_samples) / batch_size * epochs * 0.1)

        # Revise parameteres here: https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_cross-encoder_kd.py
        self.lm_model.fit(train_dataloader=train_dataloader,
                  evaluator=evaluator,
                  epochs=epochs,
                  optimizer_params = optimizer_parameters,
                  weight_decay= weight_decay,
                  evaluation_steps=evaluation_steps,
                  warmup_steps=warmup_steps,
                  save_best_model = save_best_model,
                  output_path=model_path)

    def rerank_candidates(self, entity_texts, candidates, k=200):
        # Candidates follows the datastructure generate at candidate_generataion step
        can_codes, _, can_texts = candidates # Ignore previous scores

        # output lists
        self.codes_candidates = list()
        reranked_codes = list()
        reranked_texts = list()
        reranked_scores = list()

        # Prepare prediction inputs [entity, candidate_i]
        for entidad, candidatos, codigos_candidatos in zip(entity_texts, can_texts, can_codes):
            # Iterating over query mentions
            inner_list = [[entidad, candidato] for candidato in candidatos]

            scores = self.lm_model.predict(inner_list)
            results = [{'input': inp, 'score': score} for inp, score in zip(inner_list, scores)]
            results = sorted(results, key=lambda x: x['score'], reverse=True)

            temp_dict = dict(zip(candidatos, codigos_candidatos))
            # Guille: temp_dict seems to be redundant and inefficient,
            # the mapping (if needed) could be provided as a :param
            temp_df = pd.DataFrame(results)
            temp_df["entity"] = temp_df.input.apply(lambda x: x[0])
            temp_df["candidate"] = temp_df.input.apply(lambda x: x[1])
            temp_df["code"] = temp_df.candidate.map(temp_dict)

            arr_codes = temp_df.code.to_list()
            arr_texts = temp_df.candidate.to_list()
            arr_scores = temp_df.score.to_list()
            assert len(arr_codes) == len(arr_texts) == len(arr_scores)

            reranked_codes.append(arr_codes)
            reranked_texts.append(arr_texts)
            reranked_scores.append(arr_scores)

            # Save the set of unique codes and the corresponding texts
            dict_code_texts = OrderedDict()
            for i in range(len(arr_codes)):
                code = arr_codes[i]
                text = arr_texts[i]
                if code in dict_code_texts:
                    dict_code_texts[code].append(text)
                else:
                    dict_code_texts[code] = [text]

            # Sort the texts associated to each code
            for code in dict_code_texts:
                texts = dict_code_texts[code]
                # Duplicated texts associated with the same code
                # are not expected
                assert len(texts) == len(set(texts))
                dict_code_texts[code] = sorted(texts)

            self.codes_candidates.append(dict_code_texts)

        # Preparamos datos para ser guardados en variable predictions:
        self.candidates = predictions(reranked_codes, reranked_scores, reranked_texts)

        return reranked_texts
