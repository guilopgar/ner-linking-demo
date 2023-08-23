"""
Script to load, process and save a gazetteer.
The gazetteer is expected to be located in server/linking/data/gazetteer
folder.
The name of the gazetteer (specified by the user) should correspond to a
clinical entity type.
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


# Import modules
sys.path.append(
    os.path.join(ROOT_PATH, "src")
)
from candidates import faisslm
from utils import gazetteer_pre_process

# Input arguments management
parser = argparse.ArgumentParser()
parser.add_argument(
    "-g",
    "--gaz",
    type=str,
    action='store',
    required=True,
    help="Name of the gazetteer to be used. It should correspond to a " +
    "clinical entity type, e.g. enfermedad"
)

args = parser.parse_args()
gaz_name = args.gaz

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


# 1. Data loading

# 1.1. Dictionary loading
gaz_path = os.path.join(
    gaz_dir,
    gaz_name + ".tsv"
)
out_gaz_path = os.path.join(
    gaz_dir,
    gaz_name + "_dict_term_code.pkl"
)
if os.path.exists(out_gaz_path):
    with open(out_gaz_path, "rb") as f:
        print("Term-code dictionary already exists!")
        dict_term_code = pickle.load(f)
else:
    print("Term-code dictionary does not exist, creating it...")
    if not os.path.isfile(gaz_path):
        raise Exception("The specified gazetteer does not exist: " + gaz_name)

    dict_term_code = gazetteer_pre_process.read_gazetteer_to_dict(
        df_gaz=pd.read_csv(gaz_path, sep='\t', header=0, dtype={'code': 'str'})
    )
    with open(out_gaz_path, "wb") as f:
        pickle.dump(dict_term_code, f)

print("Number of terms in the gazetteer:", len(dict_term_code))


# 2. Generate & save embeddings

# 2.1. Fit embedding model
emb_dir_path = os.path.join(
    gaz_dir,
    MODEL_NAME
)
if not os.path.exists(emb_dir_path):
    os.makedirs(emb_dir_path)

emb_file_path = os.path.join(
    emb_dir_path,
    gaz_name + "_term_embeddings.npy"
)

start_time = time.time()
faisslm_biencoder = faisslm.FaissLmCandidates(
    model_name=transformer_model_path,
    arr_emb_path=emb_file_path,
    model_type=transformer_model_type,
    max_seq_length=transformer_max_seq_len,
    faiss_type="FlatIP",
    vocab=dict_term_code,
    k=200  # not used when generating embeddings
)
end_time = time.time()
print(
    "Execution time of fitting embedding model (mins):",
    str(round((end_time - start_time)/60, 2))
)
