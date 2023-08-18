from . import ann_parsing

import pandas as pd

import warnings

from typing import Set, List, Tuple


def format_distemist_df(
        df: pd.DataFrame,
        valid_codes: Set,
        relevant_labels: List[str] = ['ENFERMEDAD']
) -> pd.DataFrame:
    """
    Format GS/Pred pandas Dataframe.

    Equivalent to ann_parsing_distemist.main function.

    Parameters
    ----------
    valid_codes: obtained using
        'set(map(lambda k: k.split('\t')[0], open(codes_path).readlines()))'
    """

    # Check column names are correct
    if ','.join(df.columns) == ','.join([
        'filename', 'mark', 'label', 'off0', 'off1', 'span'
    ]):
        print("\nAccording to file headers, you are on subtask ner, GS file")
    elif ','.join(df.columns) == ','.join([
        'filename', 'label', 'off0', 'off1', 'span'
    ]):
        print("\nAccording to file headers, you are on subtask ner, " +
              "predictions file")
    elif ','.join(df.columns) == ','.join([
        'filename', 'mark', 'label', 'off0', 'off1', 'span', 'code'
    ]):
        print("\nAccording to file headers, you are on subtask norm, " +
              "predictions file")
    elif ','.join(df.columns) == ','.join([
        'filename', 'label', 'off0', 'off1', 'span', 'code'
    ]):
        print("\nAccording to file headers, you are on subtask norm, " +
              "predictions file")
    elif ','.join(df.columns) == ','.join([
        'filename', 'mark', 'label', 'off0', 'off1', 'span', 'code',
        'semantic_rel'
    ]):
        print("\nAccording to file headers, you are on subtask norm, GS file")
    else:
        raise Exception('Error! File headers are not correct. Check ' +
                        'https://temu.bsc.es/distemist/submission/')

    # Check if there are annotations in file
    if df.shape[0] == 0:
        warnings.warn('There are not parsed annotations')
        return df

    # Format DataFrame
    df_ok = df.loc[df['label'].isin(relevant_labels), :].copy()
    df_ok['offset'] = df_ok['off0'].astype(str) + ' ' + \
        df_ok['off1'].astype(str)

    # Check if there are duplicated entries
    if df_ok.shape[0] != df_ok.drop_duplicates(
        subset=['filename', 'label', 'offset']
    ).shape[0]:
        warnings.warn("There are duplicated entries. " +
                      "Keeping just the first one...")
        df_ok = df_ok.drop_duplicates(
            subset=['filename', 'label', 'offset']
        ).copy()

    if "code" not in df_ok.columns:
        return df_ok

    # Format codes
    df_ok.loc[:, "code"] = df_ok["code"].apply(
        lambda k: ann_parsing.format_codes(k)
    )

    # Check all codes are valid, return lines with unvalid codes
    unvalid_lines = ann_parsing.check_valid_codes_in_column(
        df_ok, "code", valid_codes
    )
    if len(unvalid_lines) > 0:
        warnings.warn("Lines contain unvalid codes. Ignoring " +
                      "ALL PREDICTIONS in lines with unvalid codes...")
        df_ok = df_ok.drop(unvalid_lines).copy()

    return df_ok


def calculate_distemist_metrics(
        gs: pd.DataFrame,
        pred: pd.DataFrame,
        subtask: str = 'ner'
) -> Tuple[float, float, float]:
    """
    Parameters
    ----------
    gs : Gold Standard annotationss.
         Columns are those defined in format_distemist_gs function.
    pred : Predicted annotations.
           Columns are those defined in format_distemist_pred function.
    subtask : Subtask name, possible values are: 'ner', 'norm'
    """

    # Get ANN files in Gold Standard
    ann_list_gs = set(gs['filename'].tolist())

    # Remove predictions for files not in Gold Standard
    pred = pred.loc[pred['filename'].isin(ann_list_gs), :]

    # Compute metrics
    # Predicted Positives:
    Pred_Pos = pred.drop_duplicates(subset=['filename', "offset"]).shape[0]

    # Gold Standard Positives:
    GS_Pos = gs.drop_duplicates(subset=['filename', "offset"]).shape[0]

    # Eliminate predictions not in GS (prediction needs to be in same clinical
    # case and to have the exact same offset to be considered valid!!!!)
    df_sel = pd.merge(pred, gs,
                      how="right",
                      on=["filename", "offset", "label"])

    if subtask == 'norm':
        # Check if codes are equal
        df_sel["is_valid"] = \
            df_sel.apply(lambda x: (x["code_x"] == x["code_y"]), axis=1)
    elif subtask == 'ner':
        is_valid = df_sel.apply(lambda x: ~(x.isnull().any()), axis=1)
        df_sel = df_sel.assign(is_valid=is_valid.values)
    else:
        raise Exception('Error! Subtask name not properly set up')

    # True Positives:
    TP = df_sel[df_sel["is_valid"]].shape[0]

    # Calculate Final Metrics:
    if Pred_Pos == 0:
        P = 0
    else:
        P = TP / Pred_Pos
    if GS_Pos == 0:
        R = 0
    else:
        R = TP / GS_Pos
    if (P+R) == 0:
        F1 = 0
        warnings.warn('Global F1 score automatically set to zero to avoid ' +
                      'division by zero')
    else:
        F1 = (2 * P * R) / (P + R)

    if any([F1, P, R]) > 1:
        warnings.warn('Metric greater than 1! You have encountered an unde' +
                      'tected bug, please, contact antonio.miranda@bsc.es!')

    return P, R, F1


def format_distemist_preds(
        df_preds: pd.DataFrame,
        df_text: pd.DataFrame,
        text_col: str
) -> pd.DataFrame:
    df_preds['start'] = df_preds['location'].apply(
        lambda x: int(x.split(';')[0].split(' ')[0])
    )
    df_preds['end'] = df_preds['location'].apply(
        lambda x: int(x.split(';')[-1].split(' ')[1])
    )

    df_preds.rename(
        columns={
            'clinical_case': 'filename',
            'start': 'off0',
            'end': 'off1'
        },
        inplace=True
    )

    df_preds['label'] = "ENFERMEDAD"

    df_preds['span'] = df_preds.apply(
        lambda row: df_text[
            df_text.doc_id == row['filename']
        ][text_col].values[0][
            int(row['off0']):int(row['off1'])
        ],
        axis=1
    )

    return df_preds[['filename', 'label', 'off0', 'off1', 'span']]