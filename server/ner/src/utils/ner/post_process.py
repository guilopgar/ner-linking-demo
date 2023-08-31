"""
Module containing utils for Named Entity Recognition (NER)
using Transformers: Post-processing model preductions
"""

from abc import ABC, abstractmethod
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from typing import List, Dict, Tuple, Optional, Any

from .pre_process import Tokenizer


# Template pattern: word predictions converters

class WordPredsConverter(ABC):
    """
    Abstract class that implements a template for converting subtoken-level
    predictions to word-level.
    """
    def convert_word(
        self,
        arr_subtok_preds: np.ndarray,
        n_outputs: int
    ) -> np.ndarray:
        """
        :param arr_subtok_preds
            shape: [n_subtokens, n_outputs]

        :returns
            shape: [n_outputs,]
        """
        res = self.calculate_word_preds(
            arr_subtok_preds, n_outputs
        )
        assert len(res) == n_outputs
        return res

    @staticmethod
    @abstractmethod
    def calculate_word_preds(arr_preds, n_outputs) -> np.ndarray:
        pass


class MaxWordPreds(WordPredsConverter):
    @staticmethod
    def calculate_word_preds(arr_preds, n_outputs) -> np.ndarray:
        return np.max(arr_preds, axis=0)


class ProdWordPreds(WordPredsConverter):
    @staticmethod
    def calculate_word_preds(arr_preds, n_outputs) -> np.ndarray:
        return np.prod(arr_preds, axis=0)


class SumWordPreds(WordPredsConverter):
    @staticmethod
    def calculate_word_preds(arr_preds, n_outputs) -> np.ndarray:
        return np.sum(arr_preds, axis=0)


class MeanWordPreds(WordPredsConverter):
    @staticmethod
    def calculate_word_preds(arr_preds, n_outputs) -> np.ndarray:
        return np.mean(arr_preds, axis=0)


class FirstWordPreds(WordPredsConverter):
    @staticmethod
    def calculate_word_preds(arr_preds, n_outputs) -> np.ndarray:
        return arr_preds[0]


class CRFAllWordPreds(WordPredsConverter):
    @staticmethod
    def calculate_word_preds(arr_preds, n_outputs) -> np.ndarray:
        """
        A relative frequency array is returned.

        :param arr_subtok_preds
            elements of the array are assumened to be int
            shape: [n_subtokens,]
        """
        res = np.bincount(arr_preds, minlength=n_outputs) / len(arr_preds)
        assert np.sum(res) == 1
        return res


# Template pattern: extractor of code from mention

class MentionPredsConverter(ABC):
    @staticmethod
    @abstractmethod
    def convert_mention(arr_word_preds) -> np.ndarray:
        """
        Given the word-level coding-predictions (labels probabilities) made
        in a detected mention, the function returns a single coding-prediction
        (labels probabilities) made for the whole mention.

        :param arr_preds
            shape: [n_words, n_outputs]

        :returns
            shape: [n_outputs]
        """
        pass


class MaxMentionPreds(MentionPredsConverter):
    @staticmethod
    def convert_mention(arr_word_preds) -> np.ndarray:
        return np.max(arr_word_preds, axis=0)


class ProdMentionPreds(MentionPredsConverter):
    @staticmethod
    def convert_mention(arr_word_preds) -> np.ndarray:
        return np.prod(arr_word_preds, axis=0)


class SumMentionPreds(MentionPredsConverter):
    @staticmethod
    def convert_mention(arr_word_preds) -> np.ndarray:
        return np.sum(arr_word_preds, axis=0)


class MeanMentionPreds(MentionPredsConverter):
    @staticmethod
    def convert_mention(arr_word_preds) -> np.ndarray:
        return np.mean(arr_word_preds, axis=0)


class FirstMentionPreds(MentionPredsConverter):
    @staticmethod
    def convert_mention(arr_word_preds) -> np.ndarray:
        return arr_word_preds[0]


# Template pattern: extractor of labels

class LabExtractor(ABC):
    def __init__(self, arr_lab_decoder: List[Dict[int, str]]) -> None:
        self.arr_lab_decoder = arr_lab_decoder

    def convert_preds_labels(self, arr_preds, arr_labels: List) -> None:
        """
        Convert predictions to labels

        :param arr_preds
            Word-level predictions in a document
            shape: [n_labels, n_words, n_outputs]

        :param arr_labels
            Output variable where the labels arrays will be inserted
        """
        pass

    def extract_label(
            self, arr_labels, ann_location: str, arr_ann_pos,
            dict_ann: Dict[str, Any]
    ) -> None:
        pass


class LabExtractorIOB(LabExtractor):
    def __init__(
            self,
            arr_lab_decoder: List[Dict[int, str]],
            empty_val: str = "O",
            begin_val: str = "B",
            inside_val: str = "I"
    ) -> None:
        """
        :param arr_lab_decoder
            expected len: >= 1
            empty_val, begin_val and inside_val are expected to be contained
            in the values of arr_lab_decoder[0]
        """
        super().__init__(arr_lab_decoder)
        self.empty_val = empty_val
        self.begin_val = begin_val
        self.inside_val = inside_val

    def convert_preds_labels(self, arr_preds, arr_labels: List) -> None:
        """
        :param arr_labels
            final shape: [1, n_words]
        """
        arr_labels.append([
            self.arr_lab_decoder[0][pred_output] for pred_output in
            np.argmax(arr_preds[0], axis=1)
        ])

    def get_empty_val(self) -> str:
        return self.empty_val

    def get_begin_val(self) -> str:
        return self.begin_val

    def get_inside_val(self) -> str:
        return self.inside_val

    def obtain_iob_label(self, arr_labels, pos: int) -> str:
        return arr_labels[0][pos]

    def is_begin_ann(
            self, arr_labels, pos: int,
            allow_inside_as_begin: bool = False
    ) -> bool:
        iob_label = self.obtain_iob_label(arr_labels, pos)
        arr_begin_val = [self.get_begin_val()]
        if allow_inside_as_begin:
            arr_begin_val.append(self.get_inside_val())
        return iob_label in arr_begin_val

    def is_inside_ann(
            self, arr_labels, pos: int,
            ref_pos: int
    ) -> bool:
        """
        :param ref_pos
            Position of a label used as a reference.
            Only needed for strategies where the span of an annotation,
            in addition to the IOB labels, it also depends on the assigned
            codes, e.g. NER approach (see LabExtractorNER.is_inside_ann)
        """
        iob_label = self.obtain_iob_label(arr_labels, pos)
        return iob_label == self.get_inside_val()

    @staticmethod
    def extract_label(
        arr_labels, ann_location: str, arr_ann_pos,
        dict_ann: Dict[str, Any]
    ) -> None:
        dict_ann['location'] = ann_location


class LabExtractorNorm(LabExtractorIOB, ABC):
    def __init__(
            self,
            arr_lab_decoder: List[Dict[int, str]],
            iob_decoder: Dict[int, str],
            empty_val: str,
            begin_val: str,
            inside_val: str
    ) -> None:
        self.arr_lab_decoder = arr_lab_decoder
        self.lab_extractor_iob = LabExtractorIOB(
            [iob_decoder],
            empty_val,
            begin_val,
            inside_val
        )

    def get_empty_val(self) -> str:
        return self.lab_extractor_iob.get_empty_val()

    def get_begin_val(self) -> str:
        return self.lab_extractor_iob.get_begin_val()

    def get_inside_val(self) -> str:
        return self.lab_extractor_iob.get_inside_val()

    def extract_label(
            self, arr_labels, ann_location, arr_ann_pos,
            dict_ann: Dict[str, Any]
    ) -> None:
        self.lab_extractor_iob.extract_label(
            arr_labels, ann_location, arr_ann_pos, dict_ann
        )
        self.extract_code(arr_labels, arr_ann_pos, dict_ann)

    def extract_code(
            self, arr_labels, arr_ann_pos,
            dict_ann: Dict[str, Any]
    ) -> None:
        code_pred = self.extract_code_pred(arr_labels, arr_ann_pos)
        dict_ann['code_pred'] = code_pred

    @abstractmethod
    def extract_code_pred(self, arr_labels, arr_ann_pos) -> str:
        """
        :param arr_labels
            shape: [n_labels, n_words, n_outputs]
        """
        pass


class LabExtractorNER(LabExtractorNorm):
    def __init__(
            self,
            arr_lab_decoder: List[Dict[int, str]],
            empty_val: str,
            begin_val: str,
            inside_val: str,
            code_sep: str = '-'
    ) -> None:
        """
        :param arr_lab_decoder
            expected len: >= 1
        """
        self.code_sep = code_sep
        super().__init__(
            arr_lab_decoder,
            {
                key: val.split(self.code_sep)[0] for key, val in
                arr_lab_decoder[0].items()
            },
            empty_val, begin_val, inside_val
        )

    def obtain_iob_label(self, arr_labels, pos: int) -> str:
        return arr_labels[0][pos].split(self.code_sep)[0]

    def obtain_code_label(self, arr_labels, pos: int) -> str:
        return arr_labels[0][pos].split(self.code_sep)[1]

    def is_inside_ann(
            self, arr_labels, pos: int,
            ref_pos: int
    ) -> bool:
        """
        TODO: explain NER criterion
        """
        ref_label = self.get_inside_val() + self.code_sep + \
            self.obtain_code_label(arr_labels, ref_pos)
        return arr_labels[0][pos] == ref_label

    def extract_code_pred(self, arr_labels, arr_ann_pos) -> str:
        """
        :param arr_labels
            shape: [n_labels, n_words, n_outputs]
        """
        # Extract codes predicted within the annotation
        arr_code_pred = sorted(set(
            [self.obtain_code_label(arr_labels, pos) for pos in arr_ann_pos]
        ))
        # A single code is expected
        assert len(arr_code_pred) == 1
        return arr_code_pred[0]


class LabExtractorIOBNorm(LabExtractorNorm):
    def __init__(
            self,
            arr_lab_decoder: List[Dict[int, str]],
            empty_val: str,
            begin_val: str,
            inside_val: str,
            mention_preds_converter: MentionPredsConverter,
            code_mask: Optional[np.ndarray] = None
    ) -> None:
        """
        :param arr_lab_decoder
            expected len: >= 2
        """
        super().__init__(
            arr_lab_decoder,
            arr_lab_decoder[0],
            empty_val, begin_val, inside_val
        )
        self.mention_preds_converter = mention_preds_converter
        self.code_mask = code_mask
        if self.code_mask is None:
            self.code_mask = np.ones(len(self.arr_lab_decoder[1]))

    def convert_preds_labels(self, arr_preds, arr_labels: List) -> None:
        """
        :param arr_labels
            final shape: [2, n_words, ?]:
                [[n_words], [n_words, n_outputs] (preds)]
        """
        self.lab_extractor_iob.convert_preds_labels(arr_preds, arr_labels)
        arr_labels.append(arr_preds[1])

    def extract_code_pred(self, arr_labels, arr_ann_pos) -> str:
        """
        :param arr_labels
            shape: [n_labels, n_words, n_outputs]
                expected n_label: >= 2
        """
        # Extract probabilities of the codes predicted within the annotation
        ann_code_preds = self.mention_preds_converter.convert_mention(
            [arr_labels[1][pos] for pos in arr_ann_pos]
        )
        ann_pred_output = np.argmax(
            np.multiply(
                self.code_mask, ann_code_preds
            )
        )
        return self.arr_lab_decoder[1][ann_pred_output]


# Template pattern: extractor of annotations

class AnnExtractor(ABC):
    """
    Designed to be used with IOB-compatible label extractor.
    """
    def __init__(
            self, lab_extractor: LabExtractorIOB
    ) -> None:
        self.lab_extractor = lab_extractor

    def convert_preds_labels(self, arr_preds) -> List:
        arr_labels = []
        self.lab_extractor.convert_preds_labels(arr_preds, arr_labels)
        return arr_labels

    def add_annotation(
            self, doc_id, arr_labels, ann_location, arr_ann_pos,
            arr_annotations
    ) -> None:
        dict_ann = {'clinical_case': doc_id}
        # Extract labels from annotation
        self.lab_extractor.extract_label(
            arr_labels, ann_location, arr_ann_pos, dict_ann
        )
        arr_annotations.append(dict_ann)

    @abstractmethod
    def extract_annotations(
            self, doc_id, arr_preds, arr_start_end
    ) -> List[Dict[str, Any]]:
        pass


class AnnExtractorContinuous(AnnExtractor):
    """

    Arguments
    ---------
    allow_inside_as_begin:
        If True, inside_val is also allowed as the initial value of
        an annotation
    """
    def __init__(
            self, lab_extractor: LabExtractorIOB,
            allow_inside_as_begin: bool = False
    ) -> None:
        super().__init__(lab_extractor)
        self.allow_inside_as_begin = allow_inside_as_begin

    def extract_annotations(
            self, doc_id, arr_preds, arr_start_end
    ) -> List[Dict[str, Any]]:
        """
        :param arr_preds
            Word-level predictions of a document
        """
        # Convert predictions to labels
        arr_labels = self.convert_preds_labels(arr_preds)
        arr_annotations = []
        left = 0
        n_words = len(arr_labels[0])
        while left < n_words:
            if self.lab_extractor.is_begin_ann(
                arr_labels, left, self.allow_inside_as_begin
            ):
                right = left + 1
                while (right < n_words) and (
                    self.lab_extractor.is_inside_ann(
                        arr_labels, pos=right, ref_pos=left
                    )
                ):
                    right += 1

                # Annotation extracted
                ann_location = \
                    str(arr_start_end[left][0]) + " " + \
                    str(arr_start_end[right - 1][1])
                arr_ann_pos = list(range(left, right))
                # add extracted annotation
                self.add_annotation(
                    doc_id=doc_id, arr_labels=arr_labels,
                    ann_location=ann_location, arr_ann_pos=arr_ann_pos,
                    arr_annotations=arr_annotations
                )

                left = right
                # left: next pos different from inside_val
            else:
                left += 1

        return arr_annotations


class AnnExtractorDiscontinuous(AnnExtractor):
    def extract_annotations(
            self, doc_id, arr_preds, arr_start_end
    ) -> List[Dict[str, Any]]:
        # Convert predictions to labels
        arr_labels = self.convert_preds_labels(arr_preds)
        arr_annotations = []
        left = 0
        n_words = len(arr_labels[0])
        while left < n_words:
            if self.lab_extractor.is_begin_ann(
                arr_labels, left, allow_inside_as_begin=False
            ):
                # First fragment
                right = left + 1
                while (right < n_words) and (
                    self.lab_extractor.is_inside_ann(
                        arr_labels, pos=right, ref_pos=left
                    )
                ):
                    right += 1

                # save pos
                ann_location = str(arr_start_end[left][0]) + ' ' + \
                    str(arr_start_end[right-1][1])
                arr_ann_pos = list(range(left, right))

                inter = right
                while (inter < n_words) and (
                    not self.lab_extractor.is_begin_ann(arr_labels, inter)
                ):
                    if self.lab_extractor.is_inside_ann(
                        arr_labels, pos=inter, ref_pos=left
                    ):
                        # Intermediate fragment
                        right = inter + 1
                        while (right < n_words) and (
                            self.lab_extractor.is_inside_ann(
                                arr_labels, pos=right, ref_pos=left
                            )
                        ):
                            right += 1

                        # save pos
                        ann_location += ';' + str(arr_start_end[inter][0]) + \
                            ' ' + str(arr_start_end[right-1][1])
                        arr_ann_pos += list(range(inter, right))

                        inter = right

                    else:
                        inter += 1

                left = inter
                # left: pos pointing to begin_val or out of bounds

                # Add extracted annotation
                self.add_annotation(
                    doc_id=doc_id, arr_labels=arr_labels,
                    ann_location=ann_location, arr_ann_pos=arr_ann_pos,
                    arr_annotations=arr_annotations
                )

            else:
                left += 1

        return arr_annotations


# Template pattern: extractor of annotations

class PredsFragTok(ABC):
    """
    Abstract class that implements a template for calculating the positions of
    the effective tokens in each fragment predicted by a token classification
    model. The goal is to ignore the predictions made by the model on special
    tokens, e.g. CLS, SEP (for transformers only), PAD.
    """
    @abstractmethod
    def calculate_pos_tok(self, arr_len) -> List[List[int]]:
        """
        :returns
            shape: [n_fragments, n_effective_tokens]
        """
        pass


class NeuralPredsFragTok(PredsFragTok):
    """
    Neral-based token classification model.
    """
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def calculate_pos_tok(self, arr_len) -> List[List[int]]:
        """
        :param arr_len
            shape: [n_fragment, n_tokens, _]
        """
        arr_pos_tok = []
        for frag in arr_len:
            arr_pos_tok.append(list(range(
                self.tokenizer.n_inf, len(frag) + self.tokenizer.n_inf
            )))
        return arr_pos_tok


class CRFPredsFragTok(PredsFragTok):
    """
    CRF-based token classification model.
    Fragments are not expected to contain predictions made on left-"ignored"
    tokens, but it contains predictions for right-padding-CRF tokens.
    """
    def calculate_pos_tok(self, arr_len) -> List[List[int]]:
        """
        :param arr_len
            shape: [n_fragment]
        """
        arr_pos_tok = []
        for frag_len in arr_len:
            arr_pos_tok.append(list(range(
                0, frag_len
            )))
        return arr_pos_tok


def convert_doc_tok_word_preds(
        arr_subtok_word_id: List[int],
        arr_subtok_preds: List[np.ndarray],
        arr_subtok_start_end: List[Tuple[int, int]],
        word_preds_converter: WordPredsConverter,
        ann_extractor: AnnExtractor
) -> Tuple[List[List[np.ndarray]], List[Tuple[int, int]]]:
    """
    Convert subtoken-level predictions and start-end pos to word-level.

    :param arr_subtok_word_id
        shape: [n_subtokens]
    :param arr_subtok_preds
        shape: [n_labels, n_subtokens, n_outputs]
    :param arr_subtok_start_end
        shape: [n_subtokens, 2]

    :returns arr_word_preds
        shape: [n_labels, n_words, n_outputs]
    :returns arr_word_start_end
        shape: [n_words, 2]
    """
    arr_word_start_end = []
    arr_word_preds = [[] for lab_i in range(len(arr_subtok_preds))]
    left = 0
    while left < len(arr_subtok_word_id):
        cur_word_id = arr_subtok_word_id[left]
        right = left + 1
        while (right < len(arr_subtok_word_id)) and \
                (arr_subtok_word_id[right] == cur_word_id):
            right += 1
        # current word spans from left to right - 1 subtoken positions

        assert len(set(arr_subtok_start_end[left:right])) == 1
        # start-end pos of the subtokens correspond to the word start-end pos
        arr_word_start_end.append(arr_subtok_start_end[left])

        for lab_i in range(len(arr_subtok_preds)):
            arr_word_preds[lab_i].append(
                word_preds_converter.convert_word(
                    arr_subtok_preds[lab_i][left:right],
                    len(ann_extractor.lab_extractor.arr_lab_decoder[lab_i])
                )
            )

        left = right

    return arr_word_preds, arr_word_start_end


def convert_subtoken_word_preds(
        arr_doc: List, arr_frags, arr_start_end, arr_word_id, arr_preds,
        arr_preds_pos_tok: List[List[int]],
        word_preds_converter: WordPredsConverter,
        ann_extractor: AnnExtractor
):
    """
    Convert [fragment & subtoken]-level predictions to
    [document & word]-level.

    :param arr_start_end
        shape: [n_fragments, n_subtokens, 2]
    :param arr_preds
        shape: [n_labels, n_fragments, n_subtokens, n_outputs]
    :param arr_preds_pos_tok
        shape: [n_fragments, n_effective_tokens]

    :returns arr_doc_preds
        shape: [n_labels, n_docs, n_words, n_outputs]
    :returns arr_doc_start_end
        shape: [n_docs, n_words, 2]
    """
    assert len(arr_start_end) == len(arr_preds[0]) == len(arr_preds_pos_tok)

    arr_doc_start_end = []
    arr_doc_preds = [[] for lab_i in range(len(arr_preds))]
    i = 0
    for doc_i in range(len(arr_doc)):
        n_frag = arr_frags[doc_i]
        # Extract subtoken-level arrays for each document
        # (by joining adjacent fragments)
        doc_subtok_start_end = [
            start_end for frag in arr_start_end[i:i+n_frag]
            for start_end in frag
        ]
        doc_subtok_word_id = [
            word_id for frag in arr_word_id[i:i+n_frag] for word_id in frag
        ]
        assert len(doc_subtok_start_end) == len(doc_subtok_word_id)

        # Extract subtoken-level predictions, ignoring special tokens
        # (e.g. CLS, SEP (for transformers only), PAD)
        doc_subtok_preds = []
        for lab_i in range(len(arr_preds)):
            doc_subtok_preds.append(np.array([
                preds for j in range(i, i+n_frag) for preds in
                arr_preds[lab_i][j][arr_preds_pos_tok[j]]
            ]))

        # Convert subtoken-level arrays to word-level
        doc_word_preds, doc_word_start_end = convert_doc_tok_word_preds(
            arr_subtok_word_id=doc_subtok_word_id,
            arr_subtok_preds=doc_subtok_preds,
            arr_subtok_start_end=doc_subtok_start_end,
            word_preds_converter=word_preds_converter,
            ann_extractor=ann_extractor
        )
        assert len(doc_word_start_end) == (doc_subtok_word_id[-1] + 1)

        for lab_i in range(len(arr_preds)):
            assert len(doc_word_preds[lab_i]) == len(doc_word_start_end)
            arr_doc_preds[lab_i].append(doc_word_preds[lab_i])
        arr_doc_start_end.append(doc_word_start_end)

        i += n_frag

    return arr_doc_preds, arr_doc_start_end


def extract_anns_from_word_preds(
        arr_preds: List[List[List[np.ndarray]]],
        arr_start_end: List[List[Tuple[int, int]]],
        arr_doc, ann_extractor: AnnExtractor
):
    """
    :param arr_preds
        shape: [n_labels, n_docs, n_words, n_outputs]
    :param arr_start_end
        shape: [n_docs, n_words, 2]
    """
    n_labels = len(arr_preds)
    ann_res = []
    for d in range(len(arr_doc)):
        doc = arr_doc[d]
        arr_doc_preds = []  # shape: [n_labels, n_words, n_outputs]
        for lab_i in range(n_labels):
            arr_doc_preds.append(arr_preds[lab_i][d])
        arr_doc_start_end = arr_start_end[d]  # shape: [n_words, 2]
        # Extract annotations
        ann_res.extend(
            ann_extractor.extract_annotations(
                doc_id=doc,
                arr_preds=arr_doc_preds,
                arr_start_end=arr_doc_start_end
            )
        )

    return ann_res


def extract_annotations_from_model_preds(
        arr_doc, arr_frags, arr_preds, arr_start_end, arr_word_id,
        arr_preds_pos_tok,
        ann_extractor: AnnExtractor,
        word_preds_converter: WordPredsConverter
) -> pd.DataFrame:
    # Post-process the subtoken-level predictions for each document,
    # obtaining word-level predictions
    arr_doc_word_preds, arr_doc_word_start_end = convert_subtoken_word_preds(
        arr_doc=arr_doc, arr_frags=arr_frags,
        arr_start_end=arr_start_end, arr_word_id=arr_word_id,
        arr_preds=arr_preds, arr_preds_pos_tok=arr_preds_pos_tok,
        word_preds_converter=word_preds_converter,
        ann_extractor=ann_extractor
    )

    # Extract the predicted annotations from the word-level predictions
    ann_res = extract_anns_from_word_preds(
        arr_preds=arr_doc_word_preds,
        arr_start_end=arr_doc_word_start_end,
        arr_doc=arr_doc,
        ann_extractor=ann_extractor
    )

    return pd.DataFrame(ann_res)


def ens_ner_preds_brat_format(
        arr_doc, arr_ens_doc_word_preds, arr_ens_doc_word_start_end,
        ann_extractor: AnnExtractor,
        ens_eval_strat='max', norm_words=False
) -> pd.DataFrame:
    """
    Implemented strategies: "max", "prod", "sum".
    NOT ADAPTED TO CRF.

    TODO: Template patter needs to be applied for the ens_eval_strat argument
    """

    # Shapes of predictions from all models in the ensemble are assumed
    # to be the same
    n_output = len(arr_ens_doc_word_preds[0])
    # Sanity check: same word start-end arrays obatined from different models
    doc_word_start_end = arr_ens_doc_word_start_end[0]
    for i in range(len(arr_doc)):
        aux_san_arr = np.array(doc_word_start_end[i])
        for j in range(1, len(arr_ens_doc_word_start_end)):
            comp_arr = np.array(arr_ens_doc_word_start_end[j][i])
            assert np.array_equal(aux_san_arr, comp_arr)

    # Merge predictions made by all models
    arr_doc_preds = []  # final shape: n_out x n_docs x n_words x n_labels
    for lab_i in range(n_output):
        arr_doc_preds.append([])
        for d in range(len(arr_doc)):
            if norm_words:
                arr_ens_word_preds = np.array([
                    normalize(
                        np.array(word_preds[lab_i][d]), norm='l1', axis=1
                    )
                    for word_preds in arr_ens_doc_word_preds
                ])
            else:
                arr_ens_word_preds = np.array([
                    word_preds[lab_i][d]
                    for word_preds in arr_ens_doc_word_preds
                ])
            # shape: n_ens x n_words x n_labels
            if ens_eval_strat == "max":
                arr_word_preds = np.max(arr_ens_word_preds, axis=0)
            elif ens_eval_strat == "prod":
                arr_word_preds = np.prod(arr_ens_word_preds, axis=0)
            elif ens_eval_strat == "sum":
                arr_word_preds = np.sum(arr_ens_word_preds, axis=0)
            else:
                raise Exception(
                    'Ensemble evaluation strategy not implemented!'
                )

            arr_doc_preds[lab_i].append(arr_word_preds)

    # Extract the annotations from the merged predictions
    ann_res = extract_anns_from_word_preds(
        arr_preds=arr_doc_preds, arr_start_end=doc_word_start_end,
        arr_doc=arr_doc, ann_extractor=ann_extractor
    )

    return pd.DataFrame(ann_res)


def format_annotations(
        df_ann: pd.DataFrame, df_text: pd.DataFrame, label: str
) -> pd.DataFrame:
    """
    Format detected annotations, creating a table with 4 columns:
    label, start, end, span

    df_ann: expected cols: clinical_case, location
    df_text: expected cols: doc_id, raw_text
    """
    if df_ann.shape[0] == 0:
        warnings.warn('There are no annotations to format!')
        df_res = pd.DataFrame({
            "label": [],
            "start": [],
            "end": [],
            "span": []
        }) 

    else:
        # considering both continuous and discontinuous annotations
        df_res = df_ann.copy()
        df_res['start'] = df_res['location'].apply(
            lambda x: int(x.split(';')[0].split(' ')[0])
        )
        df_res['end'] = df_res['location'].apply(
            lambda x: int(x.split(';')[-1].split(' ')[1])
        )
        df_res['span'] = df_res.apply(
            lambda row: df_text[
                df_text['doc_id'] == row['clinical_case']
            ]['raw_text'].values[0][
                row['start']:row['end']
            ],
            axis=1
        )
        df_res['label'] = label

    return df_res[['label', 'start', 'end', 'span']]
