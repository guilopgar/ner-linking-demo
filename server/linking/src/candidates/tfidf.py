from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import pandas as pd
import pickle,os, sys

general_path = os.getcwd().split("deepspanorm")[0]+"deepspanorm/"
sys.path.append(general_path+'src/')
from utils.data_structures import predictions

class TfidfCandidates:
    def __init__(self, txt_column_name= "term",code_column_name = "code",
                 load: bool = False, model_path:str = None, vocab:pd.DataFrame = None, 
                 ngram_range = (1,1), tfidf_level = "word",max_features = 8000, 
                 index_type = "standard" ) -> None:
        # Include some variables in the "self" to share with other functions
        self.code_column_name = code_column_name
        self.txt_column_name = txt_column_name
        if load:
            self.tfidf_index, self.vocab,  self.tfidf_vectorizer = pickle.load(open(model_path, 'rb'))
        else:
            self.vocab = vocab
            if vocab is not None:
                if index_type =="standard":
                    self.fit_tfidf(vocab, txt_column_name, ngram_range, tfidf_level, max_features)                    
                
    def fit_tfidf(self, vocab: pd.DataFrame, txt_column_name, ngram_range, tfidf_level, max_features):
        # Intializating the tfIdf model
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True, stop_words=None, token_pattern=r"(?u)\b\w\w+\b", 
            ngram_range=ngram_range, strip_accents="unicode", analyzer = tfidf_level, 
            max_features=max_features
        )
                # Fit the TfIdf model
        self.tfidf_vectorizer.fit(vocab[txt_column_name])
        # Transform the TfIdf model
        self.tfidf_index =self.tfidf_vectorizer.transform(vocab[txt_column_name])

        
    def save(self, model_path):
        # Store data (serialize)
        pickle.dump([self.tfidf_index, self.vocab, self.tfidf_vectorizer], open(model_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        print("Model saved in "+model_path)
        

    def get_candidates(self, entity_texts: list, k:int, type_retrieval:str = "cosine", multiproc:bool = True) -> np.array:
        # Check that tfidf_index exists.
        if self.tfidf_vectorizer is None:
            raise Exception("The index was not initialized")
        if type(entity_texts) == str:
            print("Warning: Please, introduce the mention as a one-element list to improve performance")
            entity_texts = [entity_texts]
            
        # Transform queries 
        queries = self.tfidf_vectorizer.transform(entity_texts)

        # Cosine similarity between queries and concepts
        if type_retrieval == "cosine":
            # Calculate similarity
            cosine_similarities = cosine_similarity(queries, self.tfidf_index)
            # Order similarities 
            related_concept_indices = cosine_similarities.argsort() # Return indexes ordered
            # Filter only the top-k per list of candidates
            recovered_indexes = [predicted_indexes[-k:] for predicted_indexes in related_concept_indices]
            # Reverse inner list to get the correct order:
            recovered_indexes = [i[::-1] for i in recovered_indexes ]           
            # Get the codes of the k concepts recovered.
            codes_recovered = [self.vocab[self.code_column_name][list_indexes].to_list()  for list_indexes in recovered_indexes]
            # Get the similarity scores of the k concepts recovered
            similarity_recovered = [similarity_index[best_index].tolist() for best_index, similarity_index in zip(recovered_indexes,cosine_similarities)]

            # Get the descriptions textss of the k concepts recovered
            texts_recovered = [self.vocab[self.txt_column_name][list_indexes].to_list() for list_indexes in recovered_indexes]

            # Preparamos datos para ser guardados en variable predictions:
            self.candidates = predictions(codes_recovered, similarity_recovered, texts_recovered)
        
        return texts_recovered
    