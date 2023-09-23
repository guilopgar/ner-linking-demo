import pickle
import os
import sys
import torch
from sentence_transformers import SentenceTransformer, models
import numpy as np
import faiss

from typing import Dict, List

general_path = os.getcwd().split("deepspanorm")[0] + "deepspanorm/"
sys.path.append(general_path+'src/')
from utils.data_structures_term_avg import predictions


class FaissLmCandidates:
    def __init__(
            self, model_name: str, arr_text: List, model_type: str = "mask",
            faiss_type: str = "FlatL2", max_seq_length: int = 256,
            load: str = "No", model_path: str = None, vocab: Dict = None,
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
            self.arr_text = arr_text
            self.code2int = vocab  # keys: int
            self.arr_code = sorted([
                code for code in self.code2int.keys()
            ])
            self.fit_faiss()

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
        # First, encode terms
        embeddings = self.lm_model.encode(
            self.arr_text, show_progress_bar=True
        )
        # Transform to float32
        embeddings = np.array(
            [embedding for embedding in embeddings]
        ).astype("float32")

        # Group terms and average them
        arr_embeddings_code = []
        for code in self.arr_code:
            arr_embeddings_code.append(
                np.mean(
                    embeddings[self.code2int[code]],
                    axis=0
                )
            )
        embeddings = np.array(arr_embeddings_code)

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

        # Add vectors and their IDs (codes)
        faiss_index.add_with_ids(
            embeddings,
            np.array(self.arr_code)
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
            self, entity_texts: list, k: int, multiproc: bool = False
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

        # Buscamos elementos en el índice (codes)
        recovered_entities = [
            self.faiss_index.search(
                np.array([entity]), k=k
            ) for entity in encoded_entities
        ]

        # Include code for each recovered entity
        codes_recovered = [I.tolist()[0] for D, I in recovered_entities]
        # texts_ids_recovered shape: [n_queries, k]
        similarity_recovered = [D.tolist()[0] for D, I in recovered_entities]

        # Preparamos datos para ser guardados en variable predictions:
        self.candidates = predictions(
            codes_recovered, similarity_recovered
        )

        return codes_recovered
