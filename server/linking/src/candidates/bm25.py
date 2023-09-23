import pickle,os, sys
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import rank_bm25
from nltk.tokenize.treebank import TreebankWordDetokenizer

general_path = os.getcwd().split("deepspanorm")[0]+"deepspanorm/"
sys.path.append(general_path+'src/')
from utils.data_preparation import check_path_ending
from utils.data_structures import predictions
from multiprocessing import Pool, cpu_count



class BM25Candidates:
    def __init__(self, bm25_type = "BM25Okapi", txt_column_name= "term",code_column_name = "code", norm_scores = True,
                 load: bool = False, model_path:str = None, vocab:pd.DataFrame = None) -> None:
        # Build detokenizer
        self.detokenizer = TreebankWordDetokenizer()
        self.norm_scores = norm_scores
        if load:
            self.bm25_index, self.tokenized_terms,  self.text2id = pickle.load(open(model_path, 'rb'))
        else:
            if vocab is not None:
                self.fit_BM25(vocab, bm25_type, txt_column_name)
                self.generate_text2id(vocab,txt_column_name,code_column_name)
                
    def fit_BM25(self, vocab: pd.DataFrame, bm25_type:str, txt_column_name):
        self.tokenized_terms = [word_tokenize(term, language='spanish') for term in vocab[txt_column_name].to_list()]
        if bm25_type == "BM25Okapi":
            self.bm25_index = rank_bm25.BM25Okapi(self.tokenized_terms)
        
    def save(self, model_path):
        # Store data (serialize)
        pickle.dump([self.bm25_index, self.tokenized_terms, self.text2id], open(model_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        print("Model saved in "+model_path)
        
    def generate_text2id(self, vocab: pd.DataFrame, txt_column_name, code_column_name):
        self.text2id = pd.Series(vocab[code_column_name].values, index = vocab[txt_column_name]).to_dict()
    
    
    def multiprocess_get_top_n(self, args):
        mention, tokenized_terms, k = args
        # Get similarity scores bigger value, more similar
        scores = self.bm25_index.get_scores(mention)
        # Sorte results
        scores_top_n = sorted(scores,reverse=True)[:k]
        # If norm_scores, normalize results between 0 and 1.
        if self.norm_scores == True:
            sum_lst = sum(scores_top_n)
            sum_lst = sum_lst if sum_lst >0 else 1  # Avoid division by 0
            scores_top_n = [x / sum_lst for x in scores_top_n]
            
        # Get index of k-top scores
        top_n = np.argsort(scores)[::-1][:k]
        # Get concepts related to those indexes.
        predicted_concepts = [tokenized_terms[i] for i in top_n]
        # Detokenize and return 
        predicted_concepts_j = [[self.detokenizer.detokenize(entidad)] for entidad in predicted_concepts]
        return (predicted_concepts_j, scores_top_n)
    
    def get_candidates(self, entity_texts: list, k:int, multiproc:bool) -> np.array:
        if self.bm25_index is None:
            raise Exception("The index was not initialized")
        if type(entity_texts) == str:
            print("Warning: Please, introduce the mention as a one-element list to improve performance")
            entity_texts = [entity_texts]
        # Tokenizamos elementos
        tokenized_entities = [word_tokenize(term) for term in entity_texts]
        # Get candidates per entity
        if multiproc:
            # Prepare for multiproc
            valores = zip(tokenized_entities, [self.tokenized_terms]*len(tokenized_entities), [k]*len(tokenized_entities) )
            pool = Pool(len(os.sched_getaffinity(0)))
            
            predicted_concepts_j, scores_j = zip(*pool.map(self.multiprocess_get_top_n, iterable = valores, chunksize = 10))
            pool.close()
            # Flatten the inner list
            predicted_concepts_j = [sum(_list, []) for _list in predicted_concepts_j ]
        else:
            # Get candidates per entity
            predicted_concepts = [self.bm25_index.get_top_n(entity, self.tokenized_terms, n=k) for entity in tokenized_entities]
            # Join tokens!
            predicted_concepts_j = [[self.detokenizer.detokenize(concept) for concept in entidad] for entidad in predicted_concepts ]

        # Get codes
        # Instead of text2id llama a función que haga algo en un except si no encuentra el key (un try-catch)
        predicted_codes = [[self.text2id[k] for k in concepts] for concepts in predicted_concepts_j]
        
        # Preparamos datos para ser guardados en variable predictions:
        self.candidates = predictions(predicted_codes, list(scores_j), predicted_concepts_j)
        return predicted_concepts_j