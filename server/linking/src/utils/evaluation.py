# This script includes metrics to evaluate models generated in deppspanorm.
import numpy as np
import pandas as pd
import os, sys
general_path = os.getcwd().split("deepspanorm")[0]+"deepspanorm/"
sys.path.append(general_path+'src/')
from utils.data_preparation import unflatten_list, sort_lists, transform_candidates_output_to_tuple, remove_duplicates


## Function to calculate Prec@k y Recall@k

def precision_at_k_sample(real_spot, predicted_spot,k):
    """Compute the precision at K.

    Compute the precision at K of only one prediction. 
    If for a query, the algorithm predicts m results, the 
    precision@k will be computed as:

    num.true predictions / k

    Note: This function calculates prec@k for a single prediction. If you
    want to evaluate the prec@k on a complete set you should calculate
    prec@k for each sample and then calculate the average value.

    Args:
        real_spot (list): List of true labels of a mention.
        predicted_spot (list): List of predictions given for a query Q.
        k (int): K value
    
    Returns:
        p_at_k: _description_
    """
    # Transform to set
    real_set = set(real_spot)
    pre_set = set(predicted_spot[:k])
    p_at_k = len(real_set & pre_set)/k
    return p_at_k

def precision_at_k(list_gs, list_predictions, k):
    """Compute the averate precision at K of a set of predictions.

    Basically compute the p@k for each sample and the calculate de average value.

    Args:
        list_gs (_type_): _description_
        list_predictions (_type_): _description_
        k (_type_): _description_
    """

    precisions = [precision_at_k_sample(actual, predicted,k) for actual, predicted in zip(list_gs, list_predictions)]
    
    return np.mean(precisions)

def recall_at_k_sample(real_spot, predicted_spot,k):
    """Compute the recall at K.

    Compute the recall at K of only one prediction. 
    If for a query, the algorithm returns m results, the 
    recall@k will be computed as:

    num.true predictions / num.real labels

    Note: This function calculates recall@k for a single prediction. If you
    want to evaluate the recall@k on a complete set you should calculate
    recall@k for each sample and then calculate the average value.

    i.e: 
    recall_at_k_sample = [recall_at_k(actual, predicted, 200) for actual, predicted in zip(proc_gold, bm25_basic.candidates.codes)]
    recall_at_k_avg = np.mean(recall_at_k_sample)

    Args:
        real_spot (list): List of true labels of a mention.
        predicted_spot (list): List of predictions given for a query Q.
        k (int): K value

    Returns:
        r_at_k: _description_
    """
    # Transform to set
    real_set = set(real_spot)
    pre_set = set(predicted_spot[:k])
    r_at_k = len(real_set & pre_set)/len(real_set)
    return r_at_k

def recall_at_k(list_gs, list_predictions, k):
    """Compute the averate recall at K of a set of predictions.

    Basically compute the r@k for each sample and the calculate de average value.

    Args:
        list_gs (_type_): _description_
        list_predictions (_type_): _description_
        k (_type_): _description_
    """

    recalls = [recall_at_k_sample(actual, predicted,k) for actual, predicted in zip(list_gs, list_predictions)]
    
    return np.mean(recalls)


## Function to calculate acc@k

def check_label(golden_cui, predicted_cui ):
    """    
    Check if any of the elements of a list of predicted labels is in the golden
    standard list of labels
    
    Args:
        predicted_cui (lst): List of categorical labels
        golden_cui (lst): List of categorical labels

    Returns:
        int: 1 or 0
    """
    return len(set(predicted_cui).intersection(set(golden_cui))) > 0

def candidate_in_gs(golden_code, candidates, topk = 1):
    """
    Check if any of the top-k predicted candidates are in the gold standard.
    
    Args:
        golden_code (lst): List of 
        candidates (lst): _description_
        topk (int, optional): _description_. Defaults to 1.

    Returns:
        int: _description_
    """
    #Iterate over the list of candidadtes.
    
    if check_label(golden_code, candidates[:topk] ): return 1
    return 0

def acc_at_k(gold_entities, predicted_entities, k):
    """
    Calculate the acc@k of a list of lists of predictions given a gold standard.
    
    """
    presence_of_predictions = [candidate_in_gs(actual, predicted,k) for actual, predicted in zip(gold_entities, predicted_entities)]
    return np.mean(presence_of_predictions)



### Otras:
# (Extraido de aquí) https://github.com/Polaris000/BlogCode/blob/main/MetricsMultilabel/multi_label_metrics.ipynb

from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_standard_metrics(y_pred, y_gold, vocabulario,model_name, average="micro"):
    # Creamos binarizer de etiqueas:
    multilabel_binarizer = MultiLabelBinarizer(classes = vocabulario.code.unique().tolist())
    # Entrenamos
    multilabel_binarizer.fit(vocabulario.code.unique().tolist())
    ## Transform gold labels and predictions:
    y_gold = multilabel_binarizer.transform(y_gold)
    y_pred = multilabel_binarizer.transform(y_pred)
    # Calculamos:
    output_dict = {
        "precision": precision_score(y_gold, y_pred, average=average),
        "recall": recall_score(y_gold, y_pred, average=average),
        "fscore": f1_score(y_gold, y_pred, average=average)
    }
    return output_dict

def evaluation_experiment(candidate_ranking, gold_labels, k_values, experiment_type = "candidates"):
    """Function to evaluate and record a previously generated WandB run. If experiment_type == candidates, it logs the precision,
     recall and acurracy for all the k_values obtained in the candidate generation process, as well as the value of k for the highest 
     recall, and the dataframe in table format.

    Args:
        candidate_ranking (list)): Candidate generation object or objetcts inside a list from deepspanorm repository.
        gold_labels (list): List of gold standard labels to evaluate the results
        k_max (_type_): Values of k up to which the calculation of performance metrics will be obtained.
        wandb_object (_type_): Object WandB
    """
    if experiment_type == "candidates":
        precision_recall_dict= dict()
        if len(candidate_ranking)==1:
            #Calculate metrics for each k
            for k in range(1, k_values+1,1):
                precision = precision_at_k(gold_labels, candidate_ranking[0].candidates.codes,k)
                recall = recall_at_k(gold_labels, candidate_ranking[0].candidates.codes,k)
                accuracy = acc_at_k(gold_labels, candidate_ranking[0].candidates.codes,k)
                precision_recall_dict[k] = { 
                    "prec": precision,
                    "recall": recall,
                    "accuracy": accuracy  
                    }
        # If this is true, we are sending a list of candidate generation objects.
        elif len(candidate_ranking) > 1: 
            # Generate list of lists of tuples (one list per candidate generation object)
            out_in_shape = list(map(transform_candidates_output_to_tuple, candidate_ranking))
            
            # Iterate over the candidates to join in one. 
            combined_candidates = [sort_lists(unflatten_list(list(candidates))) for candidates in zip(*out_in_shape)]

            # Remove duplicates taken only the candidate with biggest similarity
            final_candidates = [remove_duplicates(lista,k_value=500, all_predictions = True) for lista in combined_candidates]
            prepared_final_candidates = [[i for i,j in lista] for lista in final_candidates]

            # Get the max_size
            max_size = max(map(len, prepared_final_candidates))

            for k in range(1, max_size+1,1):
                precision = precision_at_k(gold_labels, prepared_final_candidates,k)
                recall = recall_at_k(gold_labels, prepared_final_candidates,k)
                accuracy = acc_at_k(gold_labels, prepared_final_candidates,k)
                precision_recall_dict[k] = { 
                    "prec": precision,
                    "recall": recall,
                    "accuracy": accuracy  
                    }

        # Calculate dataframe of metrics
        performance_df = pd.DataFrame(precision_recall_dict).T.reset_index()
        performance_df.head(1)
        # Caluclate the index where best recall is obtained
        best_k = performance_df[ performance_df.recall == performance_df.recall.max()].iloc[0]["index"]
        print("El índice con el valor mayor de K es: {}".format(best_k))
    else: 
        print("your experiment type is not corrected written")

    return performance_df


def evaluation_and_log_experiment(candidate_ranking, gold_labels, k_values, wandb_object, experiment_type = "candidates"):
    """Function to evaluate and record a previously generated WandB run. If experiment_type == candidates, it logs the precision,
     recall and acurracy for all the k_values obtained in the candidate generation process, as well as the value of k for the highest 
     recall, and the dataframe in table format.

    Args:
        candidate_ranking (list)): Candidate generation object or objetcts inside a list from deepspanorm repository.
        gold_labels (list): List of gold standard labels to evaluate the results
        k_max (_type_): Values of k up to which the calculation of performance metrics will be obtained.
        wandb_object (_type_): Object WandB
    """
    if experiment_type == "candidates":
        precision_recall_dict= dict()
        if len(candidate_ranking)==1:
            #Calculate metrics for each k
            for k in range(1, k_values+1,1):
                precision = precision_at_k(gold_labels, candidate_ranking[0].candidates.codes,k)
                recall = recall_at_k(gold_labels, candidate_ranking[0].candidates.codes,k)
                accuracy = acc_at_k(gold_labels, candidate_ranking[0].candidates.codes,k)
                precision_recall_dict[k] = { 
                    "prec": precision,
                    "recall": recall,
                    "accuracy": accuracy  
                    }

                metrics = {
                        "prec": precision, 
                        "recall": recall, 
                        "accuracy": accuracy
                        } 

                wandb_object.log(metrics)
        # If this is true, we are sending a list of candidate generation objects.
        elif len(candidate_ranking) > 1: 
            # Generate list of lists of tuples (one list per candidate generation object)
            out_in_shape = list(map(transform_candidates_output_to_tuple, candidate_ranking))
            
            # Iterate over the candidates to join in one. 
            combined_candidates = [sort_lists(unflatten_list(list(candidates))) for candidates in zip(*out_in_shape)]

            # Remove duplicates taken only the candidate with biggest similarity
            final_candidates = [remove_duplicates(lista,k_value=500, all_predictions = True) for lista in combined_candidates]
            prepared_final_candidates = [[i for i,j in lista] for lista in final_candidates]

            # Get the max_size
            max_size = max(map(len, prepared_final_candidates))

            for k in range(1, max_size+1,1):
                precision = precision_at_k(gold_labels, prepared_final_candidates,k)
                recall = recall_at_k(gold_labels, prepared_final_candidates,k)
                accuracy = acc_at_k(gold_labels, prepared_final_candidates,k)
                precision_recall_dict[k] = { 
                    "prec": precision,
                    "recall": recall,
                    "accuracy": accuracy  
                    }

                metrics = {
                        "prec": precision, 
                        "recall": recall, 
                        "accuracy": accuracy
                        } 

                wandb_object.log(metrics)

        # Calculate dataframe of metrics
        performance_df = pd.DataFrame(precision_recall_dict).T.reset_index()
        performance_df.head(1)
        # Log dataframe as wandb table
        tbl = wandb_object.Table(data=performance_df)
        assert all(tbl.get_column("index") == performance_df["index"])
        assert all(tbl.get_column("prec") == performance_df["prec"])
        assert all(tbl.get_column("recall") == performance_df["recall"])
        assert all(tbl.get_column("accuracy") == performance_df["accuracy"])
        wandb_object.log({"table": tbl})

        # Caluclate the index where best recall is obtained
        best_k = performance_df[ performance_df.recall == performance_df.recall.max()].iloc[0]["index"]
        print("El índice con el valor mayor de K es: {}".format(best_k))

        # Log best_k and metrics as values
        wandb_object.log({
            "best_k": best_k,
            "prec_at_best_k": performance_df[performance_df["index"]==best_k].prec.iloc[0],
            "recall_at_best_k": performance_df[performance_df["index"]==best_k].recall.iloc[0],
            "acc_at_best_k": performance_df[performance_df["index"]==best_k].accuracy.iloc[0]
            })
    else: 
        print("your experiment type is not corrected written")

    return "DONE" 



def save_reranking_data(proc_gold, queries, modelo_crossreranker_loaded, biencoder_loaded, candidates_k):
        """Function that calculates parameters interesting to evaluate rerankers. 
            This function needs to be refactorised.
        """
        lista=range(0,len(proc_gold),1)
        num_cand = candidates_k
        contador = 0
        position_in_result_ce=list()
        position_in_result_ce_full = list()
        position_in_result_bi=list()
        position_in_result_bi_full = list()
        # Iteramos por cada elemento de las listas. 
        # Introducimos en 
        for elemento in lista:
            if proc_gold[elemento][0] in modelo_crossreranker_loaded.candidates.codes[elemento][0:num_cand]:
                position_in_result_ce.append(modelo_crossreranker_loaded.candidates.codes[elemento][0:num_cand].index(proc_gold[elemento][0]))
                position_in_result_ce_full.append((queries[elemento], modelo_crossreranker_loaded.candidates.texts[elemento][0], modelo_crossreranker_loaded.candidates.codes[elemento][0:num_cand].index(proc_gold[elemento][0])))
            else:
                position_in_result_ce.append("Not_predicted")
                position_in_result_ce_full.append((queries[elemento],"Not_predicted","Not_predicted"))
            if proc_gold[elemento][0] in biencoder_loaded.candidates.codes[elemento][0:num_cand]:
                position_in_result_bi.append( biencoder_loaded.candidates.codes[elemento][0:num_cand].index(proc_gold[elemento][0]))
                position_in_result_bi_full.append((queries[elemento], biencoder_loaded.candidates.texts[elemento][0], biencoder_loaded.candidates.codes[elemento][0:num_cand].index(proc_gold[elemento][0])))
            else:
                position_in_result_bi.append("Not_predicted")
                position_in_result_bi_full.append((queries[elemento],"Not_predicted","Not_predicted"))
                
        # Calculamos para el valor "k" en que sistema la predicción es mejor y en cual es peor.
        mejor_peor = list()
        for bi,ce in zip(position_in_result_bi, position_in_result_ce):
            if (bi == "Not_predicted") & (ce != "Not_predicted"):
                mejor_peor.append("MEJOR_CE")
            elif (ce == "Not_predicted") & (bi != "Not_predicted"):
                mejor_peor.append("MEJOR_BI")
            elif (ce == "Not_predicted") & (bi == "Not_predicted"):
                mejor_peor.append("IGUAL_NP")
            elif bi < ce:
                mejor_peor.append("MEJOR_BI")
            elif bi>ce:
                mejor_peor.append("MEJOR_CE")
            elif bi==ce:
                mejor_peor.append("IGUAL")
            else: 
                mejor_peor.append("OTRO")
                
        return mejor_peor, position_in_result_ce, position_in_result_ce_full, position_in_result_bi, position_in_result_bi_full


# Entity linking

def sort_gs_preds_ner_annotations(
    df_gs: pd.DataFrame, df_preds: pd.DataFrame,
    start_offset_col: str = "off0", end_offset_col: str = "off1"
) -> int:
    """
    Both dataframes are assumed to contain the following columns:
    "filename", start_offset_col, end_offset_col, "code"
    """
    # Sort gs data
    arr_cols = ["filename", start_offset_col, end_offset_col]
    assert ~df_gs[arr_cols].duplicated().any()
    df_gs.sort_values(
        arr_cols, ascending=True, inplace=True
    )
    # Sort preds data
    assert ~df_preds[arr_cols].duplicated().any()
    df_preds.sort_values(
        arr_cols, ascending=True, inplace=True
    )
    assert df_gs.shape[0] == df_preds.shape[0]
    assert np.array_equal(df_gs[arr_cols].values, df_preds[arr_cols].values)

    # return length
    return df_gs.shape[0]


def calculate_accuracy_el(
    df_gs: pd.DataFrame, df_preds: pd.DataFrame,
    start_offset_col: str = "off0", end_offset_col: str = "off1"
) -> float:
    n = sort_gs_preds_ner_annotations(
        df_gs=df_gs, df_preds=df_preds,
        start_offset_col=start_offset_col,
        end_offset_col=end_offset_col,
    )
    return sum(df_gs["code"].values == df_preds["code"].values) / n


def calculate_accuracy_at_k_el(
    df_gs: pd.DataFrame, df_preds: pd.DataFrame, k: int,
    start_offset_col: str = "off0", end_offset_col: str = "off1"
) -> float:
    n = sort_gs_preds_ner_annotations(
        df_gs=df_gs, df_preds=df_preds,
        start_offset_col=start_offset_col,
        end_offset_col=end_offset_col,
    )
    return sum([
        df_gs.iloc[i]["code"] in df_preds.iloc[i]["code"][:k] for i in range(n)
    ]) / n


def evaluate_acc_at_k_el_performance(
    file_path, df_gs,
    start_offset_col="off0",
    end_offset_col="off1",
    k=1, round_n=4
):
    if isinstance(file_path, str):
        # Load preds
        df_preds = pd.read_pickle(file_path)
        df_preds["code"] = df_preds["code"].apply(
            lambda x: [str(code) for code in x]
        )
    else:
        df_preds = file_path.copy()

    # Calculate accuracy
    acc = calculate_accuracy_at_k_el(
        df_gs=df_gs, df_preds=df_preds, k=k,
        start_offset_col=start_offset_col, end_offset_col=end_offset_col
    )
    return round(acc, round_n)


def evaluate_acc_el_performance(
        file_path, df_gs,
        start_offset_col="off0",
        end_offset_col="off1",
        round_n=4
):
    # Load preds
    df_preds = pd.read_csv(file_path, header=0, sep="\t")
    df_preds["code"] = df_preds["code"].astype(str)
    # Calculate accuracy
    acc = calculate_accuracy_el(
        df_gs=df_gs, df_preds=df_preds,
        start_offset_col=start_offset_col,
        end_offset_col=end_offset_col
    )
    return round(acc, round_n)


def load_format_df(file_path):
    df = pd.read_csv(
        file_path, header=0, sep="\t"
    )
    df["code"] = df["code"].astype(str)
    return df


def preds_filename(
        model_filename, linking_preds_path,
        split="test",
        dataset="gs", file_suf=".tsv"
):
    return os.path.join(
        linking_preds_path,
        "df_preds_" + split + "-" + dataset + "-" + model_filename + file_suf
    )


VAR_NAME = "name"
VAR_FILENAME = "filename"


def eval_gs_preds(
    arr_dict_model,
    linking_preds_path,
    df_gs,
    df_gs_unseen_ment,
    df_gs_unseen_codes,
    dataset="gs",
    start_offset_col="off0", end_offset_col="off1",
    round_n=4
):
    dict_res = {}
    for dict_model in arr_dict_model:
        # All preds
        file_preds_path = preds_filename(
            model_filename=dict_model[VAR_FILENAME],
            linking_preds_path=linking_preds_path,
            dataset=dataset
        )
        all_acc = evaluate_acc_el_performance(
            file_preds_path, df_gs=df_gs,
            start_offset_col=start_offset_col,
            end_offset_col=end_offset_col,
            round_n=round_n
        )
        # Unseen mentions
        file_preds_path = preds_filename(
            model_filename=dict_model[VAR_FILENAME],
            linking_preds_path=linking_preds_path,
            dataset=dataset + "_unseen_mentions"
        )
        unseen_ment_acc = evaluate_acc_el_performance(
            file_preds_path, df_gs=df_gs_unseen_ment,
            start_offset_col=start_offset_col,
            end_offset_col=end_offset_col,
            round_n=round_n
        )
        # Unseen codes
        file_preds_path = preds_filename(
            model_filename=dict_model[VAR_FILENAME],
            linking_preds_path=linking_preds_path,
            dataset=dataset + "_unseen_codes"
        )
        unseen_codes_acc = evaluate_acc_el_performance(
            file_preds_path, df_gs=df_gs_unseen_codes,
            start_offset_col=start_offset_col,
            end_offset_col=end_offset_col,
            round_n=round_n
        )
        # Save results
        dict_res[dict_model[VAR_NAME]] = {
            'Acc (all)': all_acc,
            'Acc (unseen ment)': unseen_ment_acc,
            'Acc (unseen codes)': unseen_codes_acc,
        }

    return pd.DataFrame(dict_res).transpose()


def eval_gs_preds_at_k(
    arr_dict_model,
    linking_preds_path,
    df_gs,
    df_gs_unseen_ment,
    df_gs_unseen_codes,
    dataset="gs",
    start_offset_col="off0", end_offset_col="off1",
    round_n=4
):
    dict_res = {}
    for dict_model in arr_dict_model:
        # All preds
        file_preds_path = preds_filename(
            model_filename=dict_model[VAR_FILENAME],
            linking_preds_path=linking_preds_path,
            dataset=dataset, file_suf=".pkl"
        )
        df_preds = pd.read_pickle(file_preds_path)
        df_preds["code"] = df_preds["code"].apply(lambda x: [str(code) for code in x])
        all_acc_1 = evaluate_acc_at_k_el_performance(
            df_preds, start_offset_col=start_offset_col, 
            end_offset_col=end_offset_col,
            df_gs=df_gs, k=1, round_n=round_n
        )
        all_acc_5 = evaluate_acc_at_k_el_performance(
            df_preds, start_offset_col=start_offset_col, 
            end_offset_col=end_offset_col,
            df_gs=df_gs, k=5, round_n=round_n
        )
        # Unseen mentions
        file_preds_path = preds_filename(
            model_filename=dict_model[VAR_FILENAME],
            linking_preds_path=linking_preds_path,
            dataset=dataset + "_unseen_mentions", file_suf=".pkl"
        )
        df_preds = pd.read_pickle(file_preds_path)
        df_preds["code"] = df_preds["code"].apply(lambda x: [str(code) for code in x])
        unseen_ment_acc_1 = evaluate_acc_at_k_el_performance(
            df_preds, start_offset_col=start_offset_col, 
            end_offset_col=end_offset_col,
            df_gs=df_gs_unseen_ment, k=1, round_n=round_n
        )
        unseen_ment_acc_5 = evaluate_acc_at_k_el_performance(
            df_preds, start_offset_col=start_offset_col, 
            end_offset_col=end_offset_col,
            df_gs=df_gs_unseen_ment, k=5, round_n=round_n
        )
        # Unseen codes
        file_preds_path = preds_filename(
            model_filename=dict_model[VAR_FILENAME],
            linking_preds_path=linking_preds_path,
            dataset=dataset + "_unseen_codes", file_suf=".pkl"
        )
        df_preds = pd.read_pickle(file_preds_path)
        df_preds["code"] = df_preds["code"].apply(lambda x: [str(code) for code in x])
        unseen_codes_acc_1 = evaluate_acc_at_k_el_performance(
            df_preds, start_offset_col=start_offset_col, 
            end_offset_col=end_offset_col,
            df_gs=df_gs_unseen_codes, k=1, round_n=round_n
        )
        unseen_codes_acc_5 = evaluate_acc_at_k_el_performance(
            df_preds, start_offset_col=start_offset_col, 
            end_offset_col=end_offset_col,
            df_gs=df_gs_unseen_codes, k=5, round_n=round_n
        )
        # Save results
        dict_res[dict_model[VAR_NAME]] = {
            'Acc@1 (all)': all_acc_1,
            'Acc@5 (all)': all_acc_5,
            'Acc@1 (unseen ment)': unseen_ment_acc_1,
            'Acc@5 (unseen ment)': unseen_ment_acc_5,
            'Acc@1 (unseen codes)': unseen_codes_acc_1,
            'Acc@5 (unseen codes)': unseen_codes_acc_5
        }

    return pd.DataFrame(dict_res).transpose()


def eval_gs_preds_at_multi_k(
    arr_dict_model,
    linking_preds_path,
    df_gs,
    arr_k,
    split="test",
    dataset="gs",
    start_offset_col="off0", end_offset_col="off1",
    round_n=4
):
    dict_res = {}
    for dict_model in arr_dict_model:
        arr_acc_k = []
        # All preds
        file_preds_path = preds_filename(
            model_filename=dict_model[VAR_FILENAME],
            linking_preds_path=linking_preds_path,
            split=split,
            dataset=dataset, file_suf=".pkl"
        )
        df_preds = pd.read_pickle(file_preds_path)
        df_preds["code"] = df_preds["code"].apply(lambda x: [str(code) for code in x])
        for k in arr_k:
            arr_acc_k.append(
                evaluate_acc_at_k_el_performance(
                    df_preds, start_offset_col=start_offset_col,
                    end_offset_col=end_offset_col,
                    df_gs=df_gs, k=k, round_n=round_n
                )
            )
        # Save results
        dict_res[dict_model[VAR_NAME]] = arr_acc_k

    return dict_res
