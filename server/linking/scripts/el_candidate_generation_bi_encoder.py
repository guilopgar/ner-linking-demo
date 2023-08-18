"""
Script to perform Entity Linking (EL) using the Spanish SapBERT model
"""

import sys
import os
import pickle
import time
import argparse
import pandas as pd
from pymongo import MongoClient


# Constant variables
ROOT_PATH = "/app/"
MODEL_NAME = "spanish_sapbert_15_parents_1epoch"
DB_NAME = "bioMnorm"


# Import modules
sys.path.append("/app/src")
from candidates import faisslm
from utils import gazetteer_pre_process


# MongoDB variables
client = MongoClient('mongodb://localhost:27017/')
db = client[DB_NAME]


# Input arguments management
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--project_id",
    type=str,
    action='store',
    required=True,
    help="Project name to retrieve documents, e.g. testing_set_cantemist"
)
parser.add_argument(
    "-g",
    "--gaz",
    type=str,
    action='store',
    default="enfermedad-distemist",
    help="Name of the gazetteer to be used, e.g. procedimiento-medprocner"
)
parser.add_argument(
    "-k",
    "--kcand",
    type=int,
    action='store',
    default=30,
    help="The number of candidate codes to be retrieved for each mention," +
         "e.g. 20"
)

args = parser.parse_args()
gaz_name = args.gaz
k_cand = args.kcand


# Fetch project id from arguments
project_name = args.project_id

# Fetch documents based on project id
proyectos_collection = db['proyectos']
documents = list(proyectos_collection.find({'name': project_name}))
arr_mention_id = [
    doc_id for sublist in documents for doc_id in sublist['documents']
]


# Output data path
data_path = os.path.join(
    ROOT_PATH,
    "data"
)

if not os.path.exists(data_path):
    os.makedirs(data_path)

gaz_path = os.path.join(
    data_path,
    "gazetteer"
)

if not os.path.exists(gaz_path):
    os.makedirs(gaz_path)


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
out_gaz_path = os.path.join(
    gaz_path,
    gaz_name + "_dict_term_code.pkl"
)
if os.path.exists(out_gaz_path):
    with open(out_gaz_path, "rb") as f:
        print("Term-code dictionary already exists!")
        dict_term_code = pickle.load(f)
else:
    print("Term-code dictionary does not exist, creating it...")
    dict_term_code = gazetteer_pre_process.read_gazetteer_to_dict(
        df_gaz=gazetteer_pre_process.read_gazetteer_from_mongo(
            gazetteer_id=gaz_name,
            mongo_collection=db['terminologies']
        )
    )
    with open(out_gaz_path, "wb") as f:
        pickle.dump(dict_term_code, f)

print("Number of terms in the gazetteer:", len(dict_term_code))


# 2. Generate & save candidates

# 2.1. Fit embedding model
emb_dir_path = os.path.join(
    data_path,
    "gazetteer",
    MODEL_NAME
)
if not os.path.exists(emb_dir_path):
    os.makedirs(emb_dir_path)

emb_file_path = os.path.join(
    emb_dir_path,
    gaz_name + ".npy"
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


# 2.2. Get candidates

# Read from MongoDB
mentions_collection = db['menciones']

# Fetch mentions based on document ids
data = list(mentions_collection.find({'document_id': {'$in': arr_mention_id}}))

rows = [{
    'document_id': item['document_id'],
    'mention_class': item['mention_class'],
    'span_ini': item['span_ini'],
    'span_end': item['span_end'],
    'text': item['text'],
} for item in data]

# Create a DataFrame from the rows
df_ner = pd.DataFrame(rows)

if df_ner.shape[0] == 0:
    print("No mentions to be normalized were found!")

else:
    print("Number of mentions to be normalized:", df_ner.shape[0])
    # Get candidates
    start_time = time.time()
    faisslm_candidates = faisslm_biencoder.get_candidates(
        entity_texts=df_ner.text.to_list()
    )
    end_time = time.time()
    print(
        "Execution time of getting candidates (mins):",
        str(round((end_time - start_time)/60, 2))
    )
    # k codes
    df_ner["candidate_codes"] = [
        [str(code) for code in dict_code_texts] for dict_code_texts in
        faisslm_biencoder.codes_candidates
    ]

    # Write candidate codes to DB
    for index, row in df_ner.iterrows():
        doc = mentions_collection.find_one({
            'document_id': row['document_id'],
            'span_ini': str(row['span_ini']),
            'span_end': str(row['span_end'])
        })
        if doc is not None:
            # Update the document
            doc['candidate_codes'] = row['candidate_codes']
            # Replace the document in the collection with our new document
            mentions_collection.replace_one({'_id': doc['_id']}, doc)
