"""
Script to perform Medical Entity Recognition (MER) on an input text
"""

import os
import time
import sys
import argparse
from transformers import RobertaTokenizerFast
from transformers import TFRobertaForTokenClassification
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.initializers import GlorotUniform
import tensorflow as tf
import pandas as pd


# Constant variables
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "bsc-bio-ehr-es"
SEQ_LEN = 128
TEXT_COL = "raw_text"
GREEDY = True
IGNORE_VALUE = -100
RANDOM_SEED = 0


# Import modules
sys.path.append(
    os.path.join(ROOT_PATH, "src")
)
import utils.ner.pre_process as pre_proc
import utils.ner.post_process as post_proc


# Input arguments management
parser = argparse.ArgumentParser()
# TODO: modify according to the shared folder (if needed)
parser.add_argument(
    "-d",
    "--data",
    type=str,
    action='store',
    required=True,
    help="Path of the shared data folder, e.g. '../../data'"
)
parser.add_argument(
    "-t",
    "--text",
    type=str,
    action='store',
    default="text.txt",
    help="Name of the input file containing the text to be processed. " +
    "It should be stored in the shared data folder. Default: 'text.txt'"
)
parser.add_argument(
    "-m",
    "--mention",
    type=str,
    action='store',
    default="mentions.tsv",
    help="Name of the output file with the detected mentions in table " +
    "format. It should be stored in the shared data folder. " +
    "Default: 'mentions.tsv'"
)
parser.add_argument(
    "-e",
    "--entity",
    type=str,
    action='store',
    default="disease,procedure",
    help="Clinical named entities to be detected, in comma-separated format" +
    ". Default: 'disease,procedure'"
)

args = parser.parse_args()
data_dir = args.data
text_path = os.path.join(
    data_dir,
    args.text
)
ment_table_path = os.path.join(
    data_dir,
    args.mention
)
arr_ent_type = args.entity.split(',')


# Model loading
root_model_path = os.path.join(
    ROOT_PATH,
    "models"
)

if MODEL_NAME == 'bsc-bio-ehr-es':
    model_path = os.path.join(
        root_model_path,
        "RoBERTa/pytorch",
        MODEL_NAME
    )
    tokenizer = RobertaTokenizerFast.from_pretrained(
        model_path, do_lower_case=False
    )
    model = TFRobertaForTokenClassification.from_pretrained(
        model_path, from_pt=True
    )

else:
    raise Exception("Model not available: " + MODEL_NAME)


# Auxiliary components
custom_tokenizer = pre_proc.TransformersTokenizer(
    tokenizer=tokenizer, ign_value=IGNORE_VALUE
)

B_VAL, I_VAL, EMPTY_VAL = "B", "I", "O"
ALLOW_IN_AS_BEGIN = False

custom_tokenizer = pre_proc.TransformersTokenizer(
    tokenizer=tokenizer, ign_value=IGNORE_VALUE
)

tf.random.set_seed(RANDOM_SEED)


# 1. Load text

arr_text = []
# TODO: modify according to the shared folder (if needed)
with open(text_path, "r") as file:
    arr_text.append(file.read())
df_text = pd.DataFrame({'doc_id': ["text"], TEXT_COL: arr_text})


# 2. Data pre-processing

# Create label encoders as dict
lab_encoder = {B_VAL: 0, I_VAL: 1, EMPTY_VAL: 2}
lab_decoder = {0: B_VAL, 1: I_VAL, 2: EMPTY_VAL}

# We define the custom pre-processing objects
custom_annotator = pre_proc.AnnotatorContinuous(
    labeler=pre_proc.LabelerIOB(
        empty_val=EMPTY_VAL,
        begin_val=B_VAL,
        inside_val=I_VAL
    )
)

sub_lab_converter = pre_proc.AllSubLabel()


# We generate the input data to the model
doc_list = sorted(set(df_text["doc_id"]))
df_empty = pd.DataFrame({
    "doc_id": []
})
text_tok_dict, text_y, text_frag, text_start_end_frag, text_word_id = \
    pre_proc.create_input_data(
        df_text=df_text, text_col=TEXT_COL,
        df_ann=df_empty, arr_doc=doc_list, ss_dict=None,
        tokenizer=custom_tokenizer, arr_lab_encoder=[lab_encoder],
        seq_len=SEQ_LEN, annotator=custom_annotator,
        sub_lab_converter=sub_lab_converter,
        greedy=GREEDY
    )
text_ind, text_att = \
    text_tok_dict['input_ids'], text_tok_dict['attention_mask']


# 3. Mentions detection

# 3.1. MER model definition
iob_num_labels = len(lab_encoder)

input_ids = Input(shape=(SEQ_LEN,), name='input_ids', dtype='int64')
attention_mask = Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int64')

out_seq = model.layers[0](
    input_ids=input_ids, attention_mask=attention_mask
)[0]

out_iob = Dense(
    units=iob_num_labels, kernel_initializer=GlorotUniform(seed=RANDOM_SEED)
)(out_seq)  # Multi-class classification

out_iob_model = Activation(activation='softmax', name='iob_output')(out_iob)

model = Model(inputs=[input_ids, attention_mask], outputs=out_iob_model)

print(model.summary())

arr_df_pred = []
for entity_type in sorted(arr_ent_type):
    print("\nDetecting", entity_type, "mentions...")

    # 3.2. Load model weights
    ner_model_path = os.path.join(
        root_model_path,
        "model_checkpoints",
        MODEL_NAME,
        entity_type
    )
    try:
        model.load_weights(
            ner_model_path
        )
    except tf.errors.NotFoundError as exc:
        raise Exception(
            "No MER model matching entity: " + entity_type
        ) from exc

    # 3.3. Model predictions
    start_time = time.time()
    text_preds = model.predict({
        'input_ids': text_ind, 'attention_mask': text_att
    })
    end_time = time.time()
    print(
        "Execution time of making predictions (mins):",
        (end_time - start_time) / 60
    )

    # 3.4. Predictions post-processing
    word_preds_converter = post_proc.ProdWordPreds()
    custom_ann_extractor = post_proc.AnnExtractorContinuous(
        lab_extractor=post_proc.LabExtractorIOB(
            arr_lab_decoder=[lab_decoder],
            empty_val=EMPTY_VAL,
            begin_val=B_VAL,
            inside_val=I_VAL
        ),
        allow_inside_as_begin=ALLOW_IN_AS_BEGIN
    )
    custom_preds_frag_tok = post_proc.NeuralPredsFragTok(
        tokenizer=custom_tokenizer
    )

    df_pred_text = post_proc.extract_annotations_from_model_preds(
        arr_doc=doc_list, arr_frags=text_frag,
        arr_preds=[text_preds], arr_start_end=text_start_end_frag,
        arr_word_id=text_word_id,
        arr_preds_pos_tok=custom_preds_frag_tok.calculate_pos_tok(
            arr_len=text_start_end_frag
        ),
        ann_extractor=custom_ann_extractor,
        word_preds_converter=word_preds_converter
    )

    df_pred_text = post_proc.format_annotations(
        df_ann=df_pred_text,
        df_text=df_text,
        label=entity_type.upper()
    )
    arr_df_pred.append(df_pred_text)


# 4. Save predicted mentions

# Create final predictions table
df_pred = pd.concat(
    arr_df_pred
).sort_values(
    ['start', 'end'], ascending=True
)
assert ~df_pred[['start', 'end']].duplicated().any()

# Save table
# TODO: modify according to the shared folder (if needed)
df_pred.to_csv(
    ment_table_path,
    sep='\t', index=False, header=True
)
