import pickle
import os
import sys
import torch
from sentence_transformers import SentenceTransformer, models
from collections import OrderedDict
import numpy as np
import pandas as pd
import faiss

from typing import Dict

general_path = os.getcwd().split("deepspanorm")[0] + "deepspanorm/"
sys.path.append(general_path+'src/')
from utils.data_structures import predictions


class FaissLmCandidates:
    def __init__(
            self, model_name: str, arr_emb_path: str, model_type: str = "mask",
            faiss_type: str = "FlatL2", max_seq_length: int = 256,
            load: str = "No", model_path: str = None, vocab: Dict = None,
            k: int = 200
    ) -> None:
        if load == "lm":
            # Use this option when save(option="lm").
            # It loads language_model data, but generate a new faiss
            # Self faiss_type:
            self.faiss_type = faiss_type
            self.lm_model = pickle.load(open(model_path, 'rb'))[0]

        # TODO: deal with other load modes:
        # https://github.com/luisgasco/deepspanorm/blob/main/src/candidates/faisslm.py#L17

        else:
            # Use this option when save has not been used
            self.load_lm(model_type, model_name, max_seq_length)
            # Self faiss_type:
            self.faiss_type = faiss_type

        if vocab is not None:
            self.k = k
            self.text2code = vocab
            self.max_n_texts = self.calculate_max_n_texts()
            self.arr_text = sorted(self.text2code.keys())
            self.arr_text_id = list(range(len(self.arr_text)))
            self.arr_emb_path = arr_emb_path
            self.fit_faiss()

    def calculate_max_n_texts(self) -> int:
        """
        Calculate the maximum number of texts needed to produce 200 codes,
        i.e. the sum of the number of texts associated with the 200 codes with
        more synonyms.
        """
        code2text = {}
        for text in self.text2code:
            code = self.text2code[text]
            if code not in code2text:
                code2text[code] = [text]
            else:
                code2text[code].append(text)
        # Eliminate duplicated text for a single code
        for code in code2text:
            code2text[code] = set(code2text[code])

        return int(pd.Series([
            len(code2text[code]) for code in code2text
        ]).sort_values(ascending=False).values[:self.k].sum())

    def load_lm(self, model_type, model_name, max_seq_length):
        if model_type == "mask":
            word_embedding_model = models.Transformer(
                model_name, max_seq_length=max_seq_length
            )
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension()
            )
            self.lm_model = SentenceTransformer(
                modules=[word_embedding_model, pooling_model]
            )

        elif model_type == "cls":
            word_embedding_model = models.Transformer(
                model_name, max_seq_length=max_seq_length
            )
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode="cls"
            )
            self.lm_model = SentenceTransformer(
                modules=[word_embedding_model, pooling_model]
            )

        elif model_type == "st":
            self.lm_model = SentenceTransformer(model_name)

        else:
            raise Exception(
                "Only mask, cls and st model types are available right now"
            )

        # Check if GPU is available and use it
        if torch.cuda.is_available():
            self.lm_model = self.lm_model.to(torch.device("cuda"))
        print(self.lm_model.device)

    def fit_faiss(self):
        # Guille: modify to load a matrix of embeddings if exists
        if os.path.exists(self.arr_emb_path):
            print("Gazetteer embeddings matrix already exists!")
            embeddings = np.load(self.arr_emb_path)
        else:
            print("Gazetteer embeddings matrix does not exist, creating it...")
            # First, encode texts
            embeddings = self.lm_model.encode(
                self.arr_text, show_progress_bar=True
            )
            # Transform to float32
            embeddings = np.array(
                [embedding for embedding in embeddings]
            ).astype("float32")
            # Save embeddings matrix
            np.save(file=self.arr_emb_path, arr=embeddings)

        # Instantiate the index
        if self.faiss_type == "FlatL2":
            # Create index
            faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
            # Pass the index to IndexIDMap
            faiss_index = faiss.IndexIDMap(faiss_index)

        elif self.faiss_type == "FlatIP":
            # Generate a flat index
            faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
            # Pass the index to IndexIDMap
            faiss_index = faiss.IndexIDMap(faiss_index)
            # Normalize vectors before introducing them,
            # then the dot-product will be the same as cosine distance
            faiss.normalize_L2(embeddings)

        else:
            print(
                "There is no option to create a FAISS index of type " +
                str(self.faiss_type)
            )
            sys.exit()

        # Add vectors and their IDs (integers: [0, len(self.arr_texts) - 1])
        faiss_index.add_with_ids(
            embeddings,
            np.array(self.arr_text_id)
        )
        self.faiss_index = faiss_index

        print(
            f"Number of vectors in the Faiss index: {self.faiss_index.ntotal}"
        )

    def save(self, option, model_path):
        # TODO: re-implement save method (if needed)
        if option == "all":
            pickle.dump(
                [
                    self.faiss_index, self.text2code,
                    self.lm_model, self.faiss_type
                ],
                open(model_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL
            )
        elif option == "faiss":
            pickle.dump(
                [
                    self.faiss_index, self.text2code, self.faiss_type
                ],
                open(model_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL
            )
        elif option == "lm":
            pickle.dump(
                [
                    self.lm_model
                ],
                open(model_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL
            )
        print("Model saved in " + model_path)

    def get_candidates(
            self, entity_texts: list, multiproc: bool = False
    ) -> np.array:
        if self.faiss_index is None:
            raise Exception("The index was not initialized")
        if type(entity_texts) == str:
            print("Warning: Please, introduce the mention as a one-element" +
                  "list to improve performance")
            entity_texts = [entity_texts]

        # Codificamos elementos
        encoded_entities = self.lm_model.encode(
            entity_texts, show_progress_bar=True
        )

        if self.faiss_type == "FlatIP":
            faiss.normalize_L2(encoded_entities)

        # Buscamos elementos en el índice (integers)
        recovered_entities = [
            self.faiss_index.search(
                np.array([entity]), k=self.max_n_texts
            ) for entity in encoded_entities
        ]

        # Include text and code for each recovered entity
        texts_ids_recovered = [I.tolist()[0] for D, I in recovered_entities]
        # texts_ids_recovered shape: [n_queries, self.max_n_texts]
        similarity_recovered = [D.tolist()[0] for D, I in recovered_entities]

        # For each query, calculate the number of texts that produce
        # self.k distinct codes, and produce the codes_candidates
        # List of OrderedDict
        self.codes_candidates, texts_recovered = [], []
        for elem_texts_ids_recovered in texts_ids_recovered:
            # Iterate over queries
            dict_code_texts = OrderedDict()
            i = 0
            n_codes = 0
            while n_codes < self.k:
                # i < len(elem_texts_ids_recovered[i]),
                # since max_n_texts were retrieved
                text_id = elem_texts_ids_recovered[i]
                text = self.arr_text[text_id]
                code = self.text2code[text]
                if code in dict_code_texts:
                    dict_code_texts[code].append(text)
                else:
                    dict_code_texts[code] = [text]
                    n_codes += 1
                i += 1
            # Sort the texts associated to each code
            for code in dict_code_texts:
                texts = dict_code_texts[code]
                # Duplicated texts associated with the same code
                # are not expected
                assert len(texts) == len(set(texts))
                dict_code_texts[code] = sorted(texts)

            self.codes_candidates.append(dict_code_texts)
            texts_recovered.append([
                self.arr_text[elem_texts_ids_recovered[j]] for j in range(i)
            ])

        codes_recovered = [
            [self.text2code[text] for text in elem_texts_recovered]
            for elem_texts_recovered in texts_recovered
        ]

        assert len(codes_recovered) == len(similarity_recovered)
        similarity_recovered = [
            similarity_recovered[i][:len(codes_recovered[i])]
            for i in range(len(codes_recovered))
        ]

        # Preparamos datos para ser guardados en variable predictions:
        self.candidates = predictions(
            codes_recovered, similarity_recovered, texts_recovered
        )

        return texts_recovered
