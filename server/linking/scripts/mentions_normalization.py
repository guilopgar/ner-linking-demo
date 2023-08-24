"""
Script to perform Entity Linking (EL) on a set of previously detected mentions
"""

import sys
import os
import pickle
import time
import argparse
import pandas as pd


# Constant variables
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "spanish_sapbert_15_parents_1epoch"
ENTITY_COL = "label"
OFFSET_COLS = ["off0", "off1"]
SPAN_COL = "span"
CODE_COL = "code"


# Import modules
sys.path.append(
    os.path.join(ROOT_PATH, "src")
)
from candidates import faisslm


# Input arguments management
parser = argparse.ArgumentParser()
# TODO: modify according to the shared folder
parser.add_argument(
    "-p",
    "--preds",
    type=str,
    action='store',
    required=True,
    help="Path of the table containing the detected " +
    "mentions to be normalized"
)
parser.add_argument(
    "-k",
    "--kcand",
    type=int,
    action='store',
    default=10,
    help="The number of candidate codes to be retrieved for each mention, " +
         "e.g. 20"
)

args = parser.parse_args()
preds_path = args.preds
k_cand = args.kcand

# Gazetteer directory
gaz_dir = os.path.join(
    ROOT_PATH,
    "data",
    "gazetteer"
)


# Model path
if "spanish_sapbert" in MODEL_NAME:
    # Transformer model
    models_path = os.path.join(
        ROOT_PATH,
        "models",
        "spanish_sapbert_models"
    )
    transformer_model_name = '_'.join(MODEL_NAME.split('_')[1:])
    transformer_model_path = os.path.join(
        models_path,
        transformer_model_name
    )
    transformer_model_type = "cls"
    transformer_max_seq_len = None

print("Model path:", transformer_model_path)


# 1. Mentions data loading

# TODO: modify according to the shared folder
df_ner = pd.read_csv(preds_path, sep='\t', header=0)
# df_ner expected columns: label, off0, off1, span, code


# 2. Mentions normalization

if df_ner.shape[0] == 0:
    print("No mentions to be normalized were found!")
    df_preds = df_ner.copy()
    df_preds[CODE_COL] = []

else:
    arr_df_preds = []
    for entity_type in sorted(set(df_ner[ENTITY_COL])):
        print("\nNormalizing", entity_type, "entity...")
        df_ner_entity = df_ner[df_ner[ENTITY_COL] == entity_type].copy()
        print("Number of mentions to be normalized:", df_ner_entity.shape[0])
        entity_type = entity_type.lower()

        # 2.1. Loading gazetteer
        gaz_path = os.path.join(
            gaz_dir,
            entity_type + "_dict_term_code.pkl"
        )
        if not os.path.exists(gaz_path):
            raise Exception(
                "Gazetteer not found for the entity: " + entity_type
            )

        with open(gaz_path, "rb") as f:
            dict_term_code = pickle.load(f)

        print("Number of terms in the gazetteer:", len(dict_term_code))

        # 2.2. Fitting embedding model
        emb_file_path = os.path.join(
            gaz_dir,
            MODEL_NAME,
            entity_type + "_term_embeddings.npy"
        )
        if not os.path.exists(emb_file_path):
            raise Exception(
                "Embeddings not found for the entity: " + entity_type
            )

        start_time = time.time()
        faisslm_biencoder = faisslm.FaissLmCandidates(
            model_name=transformer_model_path,
            arr_emb_path=emb_file_path,
            model_type=transformer_model_type,
            max_seq_length=transformer_max_seq_len,
            faiss_type="FlatIP",
            vocab=dict_term_code,
            k=k_cand
        )
        end_time = time.time()
        print(
            "Execution time of fitting embedding model (mins):",
            str(round((end_time - start_time)/60, 2))
        )

        # 2.3. Getting candidates
        start_time = time.time()
        faisslm_candidates = faisslm_biencoder.get_candidates(
            entity_texts=df_ner_entity[SPAN_COL].to_list()
        )
        end_time = time.time()
        print(
            "Execution time of getting candidates (mins):",
            str(round((end_time - start_time)/60, 2))
        )
        # k codes
        df_ner_entity[CODE_COL] = [
            [int(code) for code in dict_code_texts] for dict_code_texts in
            faisslm_biencoder.codes_candidates
        ]

        arr_df_preds.append(df_ner_entity)

    # Create final predictions table
    df_preds = pd.concat(
        arr_df_preds
    ).sort_values(
        OFFSET_COLS, ascending=True
    )
    assert ~df_preds[OFFSET_COLS].duplicated().any()


# 3. Save normalized data

# TODO: modify according to the shared folder
df_preds.to_csv(preds_path, sep='\t', index=False, header=True)
