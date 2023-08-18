#!/usr/bin/env python
# coding: utf-8

# In[1]:


MODEL_NAME = "bsc-bio-ehr-es"
BATCH_SIZE = 16
EPOCHS = 74
SEQ_LEN = 128
LR = 3e-5
N_EXEC = 1


# In[2]:


arr_utils_path = ["src/", "./"]
model_root_path = "src/models/"
corpus_path = "src/distemist/datasets/distemist/"
ss_corpus_path = corpus_path + "distemist-SSplit-text/"
subtask_path = "subtrack1_entities/distemist_subtrack1_training_mentions.tsv"
codes_path = corpus_path + "dictionary_distemist.tsv"
RES_DIR = "src/distemist/scripts/preds/"


# In[3]:


import os

if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)


# In[4]:


from transformers import BertTokenizerFast, XLMRobertaTokenizerFast, RobertaTokenizerFast

# All variables that depend on model_name
if MODEL_NAME == 'beto':
    model_path = model_root_path + "BERT/pytorch/BETO/"
    tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    
elif MODEL_NAME == 'beto_galen':
    model_path = model_root_path + "BERT/pytorch/BETO-Galen/"
    tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    
elif MODEL_NAME == 'mbert':
    model_path = model_root_path + "BERT/pytorch/mBERT/"
    tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    
elif MODEL_NAME == 'mbert_galen':
    model_path = model_root_path + "BERT/pytorch/mBERT-Galen/"
    tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    
elif MODEL_NAME == 'xlmr':
    model_path = model_root_path + "XLM-R/pytorch/xlm-roberta-base/"
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_path, do_lower_case=False)

elif MODEL_NAME == 'xlmr_large':
    model_path = model_root_path + "XLM-R/pytorch/xlm-roberta-large/"
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    
elif MODEL_NAME == 'xlmr_galen':
    model_path = model_root_path + "XLM-R/pytorch/XLM-R-Galen/"
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    
elif MODEL_NAME == 'bsc-bio-ehr-es':
    model_path = model_root_path + "RoBERTa/pytorch/" + MODEL_NAME
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path, do_lower_case=False)

elif MODEL_NAME == 'roberta-base-bne':
    model_path = model_root_path + "RoBERTa/pytorch/" + MODEL_NAME
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    
elif MODEL_NAME == 'roberta-large-bne':
    model_path = model_root_path + "RoBERTa/pytorch/" + MODEL_NAME
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path, do_lower_case=False)
    
else:
    print("ERROR: NO AVAILABLE MODEL!!")
    print()


# In[5]:


import tensorflow as tf

import time

import pandas as pd
import numpy as np

# Auxiliary components
import sys
for utils_path in arr_utils_path:
    sys.path.insert(0, utils_path)

import src.utils.ner.load_data as load_data
import src.utils.ner.pre_process as pre_proc
import src.utils.ner.post_process as post_proc
import src.utils.tf.loss as tf_loss
import src.distemist.utils.metrics as metrics

# Hyper-parameters
text_col = "raw_text"
GREEDY = True
IGNORE_VALUE = -100
LOGITS = False
ROUND_N = 4

custom_tokenizer = pre_proc.TransformersTokenizer(
    tokenizer=tokenizer, ign_value=IGNORE_VALUE
)

# IOB labels
B_VAL, I_VAL, EMPTY_VAL = "B", "I", "O"
ALLOW_IN_AS_BEGIN = False

custom_tokenizer = pre_proc.TransformersTokenizer(
    tokenizer=tokenizer, ign_value=IGNORE_VALUE
)

random_seed = 0
tf.random.set_seed(random_seed)

JOB_NAME = JOB_NAME = "distemist_ner-" + MODEL_NAME + "-bs_" + str(BATCH_SIZE) + \
    "-seq_len_" + str(SEQ_LEN) + "-lr_" + str(LR) + "-epoch_" + str(EPOCHS) + \
    "_exec_" + str(N_EXEC)


# ## 1. Load text

# ### Train

# In[6]:
from pymongo import MongoClient


client = MongoClient("mongodb://:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+1.10.1")

db = client['ClinicalDB']
collection = db['patientsData']


all_documents = collection.find()

arr_ids = []
arr_texts = []
for doc in all_documents:
    arr_ids.append(doc.get('_id'))
    arr_texts.append(doc.get('text'))

df_text_train = pd.DataFrame({'doc_id': arr_ids, 'raw_text': arr_texts})

print(df_text_train)

# ## 2. Data pre-processing
# 
# We generate the valid inputs tot he model.

# In[7]:


# Create label encoders as dict (more computationally efficient)
lab_encoder = {B_VAL: 0, I_VAL: 1, EMPTY_VAL: 2}
lab_decoder = {0: B_VAL, 1: I_VAL, 2: EMPTY_VAL}


# We define the custom pre-processing objects:

# In[8]:


custom_annotator = pre_proc.AnnotatorContinuous(
    labeler=pre_proc.LabelerIOB(
        empty_val=EMPTY_VAL,
        begin_val=B_VAL,
        inside_val=I_VAL
    )
)

sub_lab_converter = pre_proc.AllSubLabel()


# ### Train

# In[9]:


train_doc_list = sorted(set(df_text_train["doc_id"]))


# In[10]:


print("\nNumber of documents:", len(train_doc_list), "\n")


# In[11]:


# Sentence-Split data


# In[12]:


ss_sub_corpus_path = ss_corpus_path + "training/"
ss_files = [f for f in os.listdir(ss_sub_corpus_path) if os.path.isfile(ss_sub_corpus_path + f)]
ss_dict_train = load_data.load_ss_files(ss_files, ss_sub_corpus_path)


# In[13]:


df_empty = pd.DataFrame({
    "doc_id": []
})


# In[14]:


start_time = time.time()


# In[15]:


train_tok_dict, train_y, train_frag, train_start_end_frag, \
                train_word_id = pre_proc.create_input_data(df_text=df_text_train, text_col=text_col, 
                                    df_ann=df_empty,
                                    arr_doc=train_doc_list, ss_dict=ss_dict_train,
                                    tokenizer=custom_tokenizer, 
                                    arr_lab_encoder=[lab_encoder], 
                                    seq_len=SEQ_LEN,
                                    annotator=custom_annotator,
                                    sub_lab_converter=sub_lab_converter,
                                    greedy=GREEDY)


# In[16]:


end_time = time.time()


# In[17]:


print("\n1. Exec time of pre-processing documents (in mins):", (end_time - start_time) / 60, "\n")


# In[18]:


train_ind, train_att = train_tok_dict['input_ids'], train_tok_dict['attention_mask']


# ## 3. Model Loading

# In[19]:


from transformers import TFBertForTokenClassification, TFXLMRobertaForTokenClassification, TFRobertaForTokenClassification 

if MODEL_NAME.split('_')[0] in ('beto', 'mbert'):
    model = TFBertForTokenClassification.from_pretrained(model_path, from_pt=True)
    
elif MODEL_NAME.split('_')[0] == 'xlmr':
    model = TFXLMRobertaForTokenClassification.from_pretrained(model_path, from_pt=True)

elif MODEL_NAME in ('bsc-bio-ehr-es', 'roberta-base-bne', 'roberta-large-bne'):
    model = TFRobertaForTokenClassification.from_pretrained(model_path, from_pt=True)


# In[20]:


from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.initializers import GlorotUniform

iob_num_labels = len(lab_encoder)

input_ids = Input(shape=(SEQ_LEN,), name='input_ids', dtype='int64')
attention_mask = Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int64')

out_seq = model.layers[0](input_ids=input_ids, attention_mask=attention_mask)[0] # take the output sub-token sequence 

# IOB-2
out_iob = Dense(units=iob_num_labels, kernel_initializer=GlorotUniform(seed=random_seed))(out_seq) # Multi-class classification 
out_iob_model = Activation(activation='softmax', name='iob_output')(out_iob)

model = Model(inputs=[input_ids, attention_mask], outputs=out_iob_model)


# In[21]:


print(model.summary())


# In[22]:


import tensorflow_addons as tfa

optimizer = tfa.optimizers.RectifiedAdam(learning_rate=LR)
loss = {'iob_output': tf_loss.TokenClassificationLoss(
    from_logits=LOGITS, ignore_val=IGNORE_VALUE
)}
loss_weights = {'iob_output': 1}
model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

# Load model weights
model.load_weights('src/distemist/models/model_checkpoints/' + JOB_NAME)


# ## 4. Model predictions

# ### Train

# In[23]:


start_time = time.time()


# In[24]:


train_preds = model.predict({'input_ids': train_ind, 'attention_mask': train_att})


# In[25]:


end_time = time.time()


# In[26]:


print("\n2. Exec time of making predictions (in mins):", (end_time - start_time) / 60, "\n")


# ## 5. Data post-processing
# 
# We post-process the models predictions to generate valid annotations.

# We define the custom post-processing objects:

# In[27]:


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


# In[28]:


valid_codes = set(map(lambda k: k.split('\t')[0], open(codes_path).readlines()))


# In[29]:


subtask = 'ner'


# In[23]:


start_time = time.time()


# In[30]:


df_pred_train = post_proc.extract_annotations_from_model_preds(arr_doc=train_doc_list, arr_frags=train_frag,
                                      arr_preds=[train_preds],
                                      arr_start_end=train_start_end_frag, arr_word_id=train_word_id,
                                      arr_preds_pos_tok=custom_preds_frag_tok.calculate_pos_tok(
                                          arr_len=train_start_end_frag
                                      ),
                                      ann_extractor=custom_ann_extractor,
                                      word_preds_converter=word_preds_converter)


# In[31]:


df_pred_train = metrics.format_distemist_preds(
    df_preds=df_pred_train,
    df_text=df_text_train,
    text_col=text_col
)


# In[32]:


df_pred_train = metrics.format_distemist_df(df=df_pred_train, valid_codes=valid_codes)


# In[33]:


# Save preds
"""
for patient in collecions.find():
    df_patient_preds = df_pred_train[df_pred_train["filename"] == patient].copy()
    collections(id=patient).write("ann", df_patient_preds)


df_pred_train.to_csv(
    RES_DIR + "df_pred_train.tsv", header=True, index=False, sep='\t'
)
"""

for patient in collection.find():
    df_patient_preds = df_pred_train[df_pred_train["filename"] == patient['_id']].copy()

    # convert the dataframe to dictionary

    data_dict = df_patient_preds[['label', 'off0', 'off1', 'span', 'offset']].to_dict("records")

    query = { "_id": patient['_id'] }

    new_value = { "$set": { "ann": data_dict } }

    
    collection.update_one(query, new_value)
    id = patient.get('_id')




all_document = collection.find()


for doc in all_document:
    id = doc.get('_id')

    ann = doc.get('ann')
    
    print(ann)


# In[25]:


end_time = time.time()


# In[26]:


print("\n3. Exec time of post-processing predictions (in mins):", (end_time - start_time) / 60, "\n")



# lokmane@aida-lcc-uma-es:~$ docker run -d --name container icb-transformers
#47d504da3ed033ad60d5a8c0131ee18094162020be2dd721601a9b14b150f9eb
#lokmane@aida-lcc-uma-es:~$ docker exec -it container cat src/distemist/scripts/preds/df_pred_train.tsv