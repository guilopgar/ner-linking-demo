import os

import numpy as np
import pandas as pd

from typing import Iterable, List, Dict, Tuple

from math import ceil


# Load text and labels

def load_text_files(file_names: Iterable[str], path: str) -> List[str]:
    """
    It loads the text contained in a set of files into a returned list of
    strings.
    Code adapted from
    https://stackoverflow.com/questions/33912773/python-read-txt-files-into-a-dataframe
    """
    output = []
    for f in file_names:
        with open(os.path.join(path, f), "r") as file:
            output.append(file.read())

    return output


def load_ss_files(file_names: Iterable[str], path: str) -> Dict[str, List]:
    """
    It loads the start-end pair of each split sentence from a set of files
    (start + \t + end line-format expected) into a
    returned dictionary, where keys are file names and values a list of tuples
    containing the start-end pairs of the split sentences.
    """
    output = dict()
    for f in file_names:
        with open(path + f, "r") as file:
            f_key = f.split('.')[0]
            output[f_key] = []
            for sent in file:
                output[f_key].append(tuple(map(int, sent.strip().split('\t'))))

    return output


def files_from_txt(txt_path: str) -> List[str]:
    """
    It reads the file names (one per line) contained in the input file.
    """
    files = []
    with open(txt_path, "r") as f:
        for line in f:
            files.append(line.strip())
    return files


def process_labels_norm(df_ann: pd.DataFrame) -> pd.DataFrame:
    """
    Primarly dessign to process CodiEsp-X annotations.
    """
    df_res = []
    for i in range(df_ann.shape[0]):
        ann_i = df_ann.iloc[i].values
        # Separate discontinuous locations and split each location into
        # [start, end] offset
        ann_loc_i = ann_i[4]
        for loc in ann_loc_i.split(';'):
            split_loc = loc.split(' ')
            df_res.append(
                np.concatenate((
                    ann_i[:4],
                    [int(split_loc[0]), int(split_loc[1])]
                ))
            )

    return pd.DataFrame(
        np.array(df_res),
        columns=list(df_ann.columns[:-1]) + ["start", "end"]
    ).drop_duplicates()


def process_brat_norm(brat_files: Iterable[str]) -> pd.DataFrame:
    """
    Primarly dessign to process Cantemist-Norm annotations.

    :param brat_files: contains the path of the annotations files in
        BRAT format (.ann)
    """
    df_res = []
    for file in brat_files:
        with open(file) as ann_file:
            doc_name = file.split('/')[-1].split('.')[0]
            i = 0
            for line in ann_file:
                i += 1
                line_split = line.strip().split('\t')
                assert len(line_split) == 3
                if i % 2 > 0:
                    # BRAT annotation
                    assert line_split[0] == "T" + str(ceil(i/2))
                    text_ref = line_split[2]
                    location = ' '.join(
                        line_split[1].split(' ')[1:]
                    ).split(';')
                else:
                    # Code assignment
                    assert line_split[0] == "#" + str(ceil(i/2))
                    code = line_split[2]
                    # Discontinuous annotations are split into a sequence of
                    # continuous annotations
                    for loc in location:
                        split_loc = loc.split(' ')
                        df_res.append([
                            doc_name, code, text_ref,
                            int(split_loc[0]), int(split_loc[1])
                        ])

    return pd.DataFrame(
        df_res, columns=["doc_id", "code", "text_ref", "start", "end"]
    )


def process_brat_ner(brat_files: Iterable[str]) -> pd.DataFrame:
    """
    Primarly dessign to process Cantemist-NER annotations.

    :param brat_files: contains the path of the annotations files in
        BRAT format (.ann)
    """
    df_res = []
    for file in brat_files:
        with open(file) as ann_file:
            doc_name = file.split('/')[-1].split('.')[0]
            for line in ann_file:
                line_split = line.strip().split('\t')
                assert len(line_split) == 3
                text_ref = line_split[2]
                location = ' '.join(line_split[1].split(' ')[1:]).split(';')
                # Discontinuous annotations are split into a sequence of
                # continuous annotations
                for loc in location:
                    split_loc = loc.split(' ')
                    df_res.append([
                        doc_name, text_ref,
                        int(split_loc[0]), int(split_loc[1])
                    ])

    return pd.DataFrame(
        df_res, columns=["doc_id", "text_ref", "start", "end"]
    )


def process_de_ident_ner(brat_files: Iterable[str]) -> pd.DataFrame:
    """
    Primarly dessign to process Galén de-identification annotations.

    :param brat_files: contains the path of the annotations files in
        BRAT format (.ann).
    """
    df_res = []
    for file in brat_files:
        with open(file) as ann_file:
            doc_name = file.split('/')[-1].split('.ann')[0]
            for line in ann_file:
                if line.strip():
                    # non-empty line
                    line_split = line.strip().split('\t')
                    if line_split[0][0] == "T":
                        assert len(line_split) == 3
                        text_ref = line_split[2]
                        ann_type = line_split[1].split(' ')[0]
                        location = ' '.join(line_split[1].split(' ')[1:])
                        df_res.append([doc_name, text_ref, ann_type, location])

    return pd.DataFrame(
        df_res, columns=["doc_id", "text_ref", "type", "location"]
    )


# Annotations exploration & pre-processing

def check_overlap_ner(
        df_ann: pd.DataFrame, doc_list: Iterable[str]
) -> pd.DataFrame:
    """
    This function returns the pairs of annotations with overlapping spans.
    For each overlapping pair, it also checks if one of the annotations
    is fully contained inside the other one.
    """
    res = []
    for doc in doc_list:
        df_doc = df_ann[df_ann['doc_id'] == doc]
        len_doc = df_doc.shape[0]
        for i in range(len_doc - 1):
            start_i = df_doc.iloc[i]['start']
            end_i = df_doc.iloc[i]['end']
            for j in range(i + 1, len_doc):
                start_j = df_doc.iloc[j]['start']
                end_j = df_doc.iloc[j]['end']
                if start_i < end_j and start_j < end_i:
                    res.append((
                        doc, start_i, end_i, start_j, end_j,
                        (start_i >= start_j and end_i <= end_j) or
                        (start_j >= start_i and end_j <= end_i)
                    ))

    return pd.DataFrame(res, columns=[
        "doc_id", "start_1", "end_1", "start_2", "end_2", "contained"
    ])


def eliminate_overlap_ner(df_ann: pd.DataFrame) -> pd.DataFrame:
    """
    For each pair of existing overlapping annotations in a document,
    the longest one is eliminated.
    """
    df_res = df_ann.copy()
    for doc in sorted(set(df_ann['doc_id'])):
        doc_over = check_overlap_ner(df_ann=df_res, doc_list=[doc])
        while doc_over.shape[0] > 0:
            # There are overlapping annotations in current doc
            aux_row = doc_over.iloc[0]
            len_1 = aux_row['end_1'] - aux_row['start_1']
            len_2 = aux_row['end_2'] - aux_row['start_2']
            if len_1 >= len_2:
                elim_start = aux_row['start_1']
                elim_end = aux_row['end_1']
            else:
                elim_start = aux_row['start_2']
                elim_end = aux_row['end_2']
            # Eliminate longer overlapping annotation
            df_res = df_res[
                (df_res['doc_id'] != aux_row['doc_id']) |
                (df_res['start'] != elim_start) |
                (df_res['end'] != elim_end)
            ]
            doc_over = check_overlap_ner(df_ann=df_res, doc_list=[doc])

    return df_res


def check_ann_span_sent(
        df_ann,
        ss_dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    This function returns: the annotations in a single document
    spanning multiple sentences, the annotations contained in a single
    sentence, and the annotations not contained in any sentence.
    """
    mult_sent, one_sent, no_sent = [], [], []
    for doc in sorted(set(df_ann['doc_id'])):
        doc_ann = df_ann[df_ann["doc_id"] == doc]
        doc_ss = ss_dict[doc]
        for index, row in doc_ann.iterrows():
            ann_s = row['start']
            ann_e = row['end']
            i = 0
            cont = False
            while ((i < len(doc_ss)) and (not cont)):
                ss_s = doc_ss[i][0]
                ss_e = doc_ss[i][1]
                if ann_s >= ss_s and ann_s < ss_e:
                    cont = True
                    if ann_e > ss_e:
                        mult_sent.append(row)
                    else:
                        one_sent.append(row)
                i += 1
            if not cont:
                no_sent.append(row)

    return pd.DataFrame(mult_sent), pd.DataFrame(one_sent),\
        pd.DataFrame(no_sent)
