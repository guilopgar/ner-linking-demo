"""
Module containing utils for Named Entity Recognition (NER)
using Transformers: Pre-processing input text data
"""

import unicodedata

from typing import Iterable, List, Dict, Tuple, Optional, Union

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from copy import deepcopy


# Whitespace-punctuation tokenization (same as BERT pre-tokenization)
# The next code is adapted from:
# https://github.com/CyberZHG/keras-bert/blob/master/keras_bert/tokenizer.py


def is_punctuation(ch: str) -> bool:
    code = ord(ch)
    return 33 <= code <= 47 or \
        58 <= code <= 64 or \
        91 <= code <= 96 or \
        123 <= code <= 126 or \
        unicodedata.category(ch).startswith('P')


def is_cjk_character(ch: str) -> bool:
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or \
        0x3400 <= code <= 0x4DBF or \
        0x20000 <= code <= 0x2A6DF or \
        0x2A700 <= code <= 0x2B73F or \
        0x2B740 <= code <= 0x2B81F or \
        0x2B820 <= code <= 0x2CEAF or \
        0xF900 <= code <= 0xFAFF or \
        0x2F800 <= code <= 0x2FA1F


def is_space(ch: str) -> bool:
    return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
        unicodedata.category(ch) == 'Zs'


def is_control(ch: str) -> bool:
    """
    Adapted from
    https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py#L64
    """
    return unicodedata.category(ch).startswith("C")


def word_start_end(text: str, start_i: int = 0, cased: bool = True) -> Tuple[
    List[str], List[Tuple[int, int]]
]:
    """
    Whitespace-punctuation tokenization of a given text. Our aim is
    to produce a list of words, and a list of offset pairs containing
    the start and end character positions of each word.

    Punctuation symbols are considered as separate words, while spaces are
    ignored.

    Code adapted from:
    https://github.com/CyberZHG/keras-bert/blob/master/keras_bert/tokenizer.py#L101

    :param start_i: the start position of the first character in the text
    :param cased: if False, the text is converted to lowercase
    """
    if not cased:
        text = unicodedata.normalize('NFD', text)
        text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
        text = text.lower()
    spaced = ''  # stores the text after pre-processing
    arr_start = []  # stores the start position of each considered character
    for ch in text:
        if is_punctuation(ch) or is_cjk_character(ch):
            spaced += ' ' + ch + ' '
            arr_start.append(start_i)
        elif is_space(ch):
            spaced += ' '
        elif not (ord(ch) == 0 or ord(ch) == 0xfffd or is_control(ch)):
            spaced += ch
            arr_start.append(start_i)
        # If it is a control char we skip it but take its offset into account
        start_i += 1

    assert sum([len(word) for word in spaced.strip().split()]) \
        == len(arr_start)

    arr_word, arr_start_end = [], []
    i = 0
    for word in spaced.strip().split():
        arr_word.append(word)
        j = i + len(word)
        arr_start_end.append((arr_start[i], arr_start[j - 1] + 1))
        i = j

    return arr_word, arr_start_end


# NER annotations

# Template pattern: tokenizers

class Tokenizer(ABC):
    """
    Abstract class which implements a template for basic tokenization
    functionality.

    Arguments
    ---------
    tokenizer:
        Tokenizer object that performs the conversion from tokens
        to numerical values

    ign_value:
        Since output labels and tokens are aligned, we use this value
        for labels that correspond to special tokens, e.g. padding,
        which will be ignored by the model

    n_inf:
        Number of special tokens added by the tokenizer at the beginning
        of each input sequence

    n_sup:
        Number of special tokens added by the tokenizer at the end
        of each input sequence

    """
    def __init__(
            self, tokenizer, ign_value: int, n_inf: int, n_sup: int
    ) -> None:
        self.tokenizer = tokenizer
        self.ign_value = ign_value
        self.n_inf = n_inf
        self.n_sup = n_sup

    @abstractmethod
    def tokenize(self, word: str, start_i: int = 0) -> Tuple[
        List[str], List[Tuple[int, int]]
    ]:
        """
        Given an input word, the method produces a list of tokens,
        and a list of offset pairs containing the start and end character
        positions of each produced token.

        :param start_i: the start position of the first character in the word
        """
        pass

    def format_token_data(
            self, arr_token: List[str],
            arr_label: List[List], arr_lab_encoder: List[Dict[str, int]],
            seq_len: int
    ) -> Tuple[Dict[str, List], List[List]]:
        """
        This method performs the encoding of a list of tokens and
        their assigned labels. Padding is added as appropriate.

        The output encoded tokens and labels are aligned, both having
        the same length.
        """
        # Encode labels
        arr_label_enc = []
        for lab_i in range(len(arr_label)):
            arr_label_enc.append([
                arr_lab_encoder[lab_i][label]
                if label != self.ign_value else label
                for label in arr_label[lab_i]
            ])

        # Encode tokens
        dict_arr_tok_enc = {}
        token_len = self._format_token(
            dict_arr_tok_enc, arr_token, arr_label_enc
        )

        # Padding
        pad_len = seq_len - token_len
        self._format_token_pad(
            dict_arr_tok_enc, pad_len
        )
        for lab_i in range(len(arr_label_enc)):
            arr_label_enc[lab_i].extend(
                [self.ign_value] * pad_len
            )

        return dict_arr_tok_enc, arr_label_enc

    @abstractmethod
    def _format_token(
            self, dict_arr_tok: Dict,
            arr_tok: List[str], arr_lab: List[List]
    ) -> int:
        pass

    @abstractmethod
    def _format_token_pad(
            self, dict_arr_tok: Dict[str, List], pad_len: int
    ) -> None:
        pass

    def get_seq_len(self, seq_len: int) -> int:
        """
        Returns the effective sequence length, considering any
        special token the tokenizer could add.
        """
        return seq_len - self.n_inf - self.n_sup


class FastTextTokenizer(Tokenizer):
    """
    Arguments
    ---------
    train_strat:
        Two different strategies are implemented: "ft" and "freeze".
        In the fine-tuning ("ft") approach, the fasttext word embeddings
        are intended to be further modified during training.
        In the "freeze" strategy, the fasttext word embeddings are not
        meant to be further modified.
    """
    def __init__(
            self, tokenizer, ign_value: int, train_strat: str = "ft"
    ) -> None:
        super().__init__(tokenizer, ign_value, 0, 0)
        self.train_strat = train_strat
        assert self.train_strat in ("ft", "freeze")

    def tokenize(self, word: str, start_i: int = 0) -> Tuple[
        List[str], List[Tuple[int, int]]
    ]:
        """
        The given word is considered as a whole, without
        further tokenizing it.
        """
        return [word], [(start_i, start_i + len(word))]

    def _format_token(
            self, dict_arr_tok: Dict,
            arr_tok: List[str], arr_lab: List[List]
    ) -> int:
        """"
        In the fine-tuning approach, indices of the words are returned,
        while in the "freeze" strategy, the whole word embedding vectors
        are returned.
        """
        if self.train_strat == "ft":
            # Fine-tuning strategy (zero-padding)
            word_id_offset = 2
            # -1 -> 1 = unk token, 0 -> 2 = first known token, etc.
            dict_arr_tok['word_ids'] = [
                self.tokenizer.get_word_id(word) + word_id_offset
                for word in arr_tok
            ]  # shape: (n_words)
        elif self.train_strat == "freeze":
            # Freezed embeddings strategy
            dict_arr_tok['word_ids'] = [
                self.tokenizer.get_word_vector(word) for word in arr_tok
            ]  # shape: (n_words, dim)

        return len(dict_arr_tok['word_ids'])

    def _format_token_pad(
            self, dict_arr_tok: Dict[str, List], pad_len: int
    ) -> None:
        pad_val = 0  # zero-padding
        if self.train_strat == "freeze":
            pad_val = np.zeros(self.tokenizer.get_dimension())
        dict_arr_tok['word_ids'] += [pad_val] * pad_len


class TransformersTokenizer(Tokenizer):
    def __init__(
            self, tokenizer, ign_value: int
    ) -> None:
        super().__init__(tokenizer, ign_value, 1, 1)

    def tokenize(self, word: str, start_i: int = 0) -> Tuple[
        List[str], List[Tuple[int, int]]
    ]:
        """
        The given word is further tokenized into subwords.
        """
        arr_start_end = []
        token_text = self.tokenizer(word, add_special_tokens=False)
        for i in range(len(token_text['input_ids'])):
            chr_span = token_text.token_to_chars(i)
            arr_start_end.append((
                chr_span.start + start_i, chr_span.end + start_i
            ))

        return self.tokenizer.convert_ids_to_tokens(token_text['input_ids']), \
            arr_start_end

    def _format_token(
            self, dict_arr_tok: Dict,
            arr_tok: List[str], arr_lab: List[List]
    ) -> int:
        """
        It also adds special labels to align them with tokens.
        """
        # Add special tokens: [CLS] and [SEP]
        dict_arr_tok['input_ids'] = \
            self.tokenizer.build_inputs_with_special_tokens(
                self.tokenizer.convert_tokens_to_ids(
                    arr_tok
                )
            )
        # Generate attention mask
        tok_len = len(dict_arr_tok['input_ids'])
        dict_arr_tok['attention_mask'] = [1] * tok_len

        # Add special labels
        for lab_i in range(len(arr_lab)):
            arr_lab[lab_i] = \
                [self.ign_value] + arr_lab[lab_i] + [self.ign_value]
            assert len(arr_lab[lab_i]) == tok_len

        return tok_len

    def _format_token_pad(
            self, dict_arr_tok: Dict[str, List], pad_len: int
    ) -> None:
        dict_arr_tok['input_ids'] += [self.tokenizer.pad_token_id] * pad_len
        dict_arr_tok['attention_mask'] += [0] * pad_len


# Template pattern: labelers

class Labeler(ABC):
    """
    Abstract class that implements a template for basic word-based
    labeling functionality.

    Arguments
    ---------
    empty_val:
        Value assigned to the unannotated words, i.e. words that do not belong
        to any annotation
    """
    def __init__(self, empty_val: str = "O") -> None:
        self.empty_val = empty_val

    def initialize_labels(
            self,
            arr_labels: List,
            n: int
    ) -> None:
        arr_labels.append([self.empty_val] * n)

    def check_empty_label(
            self,
            label: str
    ) -> None:
        """
        This function is intended to be used before labeling a word,
        with the goal of avoiding overlapping annotations.
        """
        assert label == self.empty_val

    @abstractmethod
    def label_first_word(
            self,
            arr_labels: List[List[str]],
            i: int,
            ann: Optional[pd.Series] = None
    ) -> None:
        pass

    @abstractmethod
    def label_next_word(
            self,
            arr_labels: List[List[str]],
            i: int,
            ann: Optional[pd.Series] = None
    ) -> None:
        pass


class LabelerIOB(Labeler):
    """
    Labeler that implements the IOB2 tagging scheme.
    """
    def __init__(
            self,
            empty_val: str = "O",
            begin_val: str = "B",
            inside_val: str = "I"
    ) -> None:
        super().__init__(empty_val)
        self.begin_val = begin_val
        self.inside_val = inside_val
        self.orig_begin_val = None

    def label_first_word(
            self,
            arr_labels: List[List[str]],
            i: int,
            ann: Optional[pd.Series] = None
    ) -> None:
        self.check_empty_label(arr_labels[0][i])
        arr_labels[0][i] = self.begin_val

    def label_next_word(
            self,
            arr_labels: List[List[str]],
            i: int,
            ann: Optional[pd.Series] = None
    ) -> None:
        self.check_empty_label(arr_labels[0][i])
        arr_labels[0][i] = self.inside_val

    def modify_begin_val(
            self
    ) -> None:
        """
        In some cases, it is benefitial to assign the value of inside_val
        to begin_val instance variable, e.g. when labeling discontinuous
        annotations.
        """
        self.orig_begin_val = self.begin_val
        self.begin_val = self.inside_val

    def undo_begin_val(
            self
    ) -> None:
        """
        This method undoes the changes performed by modify_begin_val method.
        """
        assert self.orig_begin_val is not None
        self.begin_val = self.orig_begin_val


class LabelerNorm(Labeler):
    """
    Labeler used in normalization settings, when words are labeled with codes.
    """
    def label_first_word(
            self,
            arr_labels: List[List[str]],
            i: int,
            ann: pd.Series
    ) -> None:
        self.check_empty_label(arr_labels[0][i])
        arr_labels[0][i] = str(ann['code'])

    def label_next_word(
            self,
            arr_labels: List[List[str]],
            i: int,
            ann: pd.Series
    ) -> None:
        self.label_first_word(arr_labels, i, ann)


class LabelerNER(LabelerIOB):
    """
    Labeler implementing the "IOB2-Code" tagging strategy.
    """
    def __init__(
            self,
            empty_val: str = "O",
            begin_val: str = "B",
            inside_val: str = "I",
            code_sep: str = '-'
    ) -> None:
        super().__init__(empty_val, begin_val, inside_val)
        self.code_sep = code_sep

    def label_first_word(
            self,
            arr_labels: List[List[str]],
            i: int,
            ann: pd.Series
    ) -> None:
        self.check_empty_label(arr_labels[0][i])
        arr_labels[0][i] = self.begin_val + self.code_sep + str(ann['code'])

    def label_next_word(
            self,
            arr_labels: List[List[str]],
            i: int,
            ann: pd.Series
    ) -> None:
        self.check_empty_label(arr_labels[0][i])
        arr_labels[0][i] = self.inside_val + self.code_sep + str(ann['code'])


class LabelerIOBNorm(LabelerIOB):
    """
    Lebeler that tags each word with both IOB2 and code labels.
    """
    def __init__(
            self,
            empty_val_iob: str,
            begin_val_iob: str,
            inside_val_iob: str,
            empty_val_norm: str
    ) -> None:
        self.label_iob = LabelerIOB(
            empty_val=empty_val_iob,
            begin_val=begin_val_iob,
            inside_val=inside_val_iob
        )
        self.label_norm = LabelerNorm(empty_val_norm)

    def initialize_labels(
            self,
            arr_labels: List,
            n: int
    ) -> None:
        self.label_iob.initialize_labels(arr_labels, n)
        self.label_norm.initialize_labels(arr_labels, n)

    def label_first_word(
            self,
            arr_labels: List[List[str]],
            i: int,
            ann: pd.Series
    ) -> None:
        self.label_iob.label_first_word(arr_labels[:1], i, ann)
        self.label_norm.label_first_word(arr_labels[1:], i, ann)

    def label_next_word(
            self,
            arr_labels: List[List[str]],
            i: int,
            ann: pd.Series
    ) -> None:
        self.label_iob.label_next_word(arr_labels[:1], i, ann)
        self.label_norm.label_next_word(arr_labels[1:], i, ann)

    def modify_begin_val(
            self
    ) -> None:
        self.label_iob.modify_begin_val()

    def undo_begin_val(
            self
    ) -> None:
        self.label_iob.undo_begin_val()


# Template pattern: annotators

class Annotator(ABC):
    """
    Abstract class that implements a template for basic annotation
    functionality, using a given labeler.
    """
    def __init__(self, labeler: Labeler) -> None:
        self.labeler = labeler

    def words_annotate(
            self, arr_start_end: np.ndarray, df_ann: pd.DataFrame
    ) -> List[List[str]]:
        """
        This method performs the annotation of a sequence of words.

        :param arr_start_end: array (shape: [n_words, 2]) of offset pairs
            containing the start and end character positions of each word
        :param df_ann: dataframe containing the annotations
        """
        arr_labels = []
        self.labeler.initialize_labels(
            arr_labels,
            len(arr_start_end)
        )
        for _, row in df_ann.iterrows():
            self.words_single_annotate(arr_labels, arr_start_end, row)

        return arr_labels

    @abstractmethod
    def words_single_annotate(
            self,
            arr_labels: List[List[str]],
            arr_start_end: np.ndarray,
            ann: pd.Series
    ) -> None:
        """
        This method performs the annotation of a single word.
        """
        pass


class AnnotatorContinuous(Annotator):
    """
    Annotator that labels words with annotations having no discontinuous
    fragments.
    """
    def words_single_annotate(
            self,
            arr_labels: List[List[str]],
            arr_start_end: np.ndarray,
            ann: pd.Series
    ) -> None:
        """
        :param ann: expected fields: 'start': int, 'end': int,
            'code' (labeler dependent): str
        """
        # First word of the annotation
        tok_start = np.where(
            arr_start_end[:, 0] <= ann['start']
        )[0][-1]  # last word <= annotation start
        # Last word of the annotation
        tok_end = np.where(
            arr_start_end[:, 1] >= ann['end']
        )[0][0]  # first word >= annotation end
        assert tok_start <= tok_end
        # Label first word
        # no overlapping annotations are expected
        self.labeler.label_first_word(arr_labels, tok_start, ann)

        if tok_start < tok_end:
            # Annotation spanning multiple words
            for i in range(tok_start + 1, tok_end + 1):
                self.labeler.label_next_word(arr_labels, i, ann)


class AnnotatorDiscontinuous(Annotator):
    """
    Annotator that labels words with annotations containing discontinuous
    fragments.
    """
    def __init__(self, labeler) -> None:
        super().__init__(labeler)
        self.annotator_cont = AnnotatorContinuous(labeler)

    def words_single_annotate(
            self,
            arr_labels: List[List[str]],
            arr_start_end: np.ndarray,
            ann: pd.Series
    ) -> None:
        """
        :param ann: expected field: 'location' (in BRAT format)
        """
        ann_loc_split = ann['location'].split(';')
        # First fragment
        ann_start, ann_end = ann_loc_split[0].split(' ')
        ann['start'] = int(ann_start)
        ann['end'] = int(ann_end)
        self.annotator_cont.words_single_annotate(
            arr_labels,
            arr_start_end,
            ann
        )
        # Subsequent fragments
        if isinstance(self.labeler, LabelerIOB):
            self.labeler.modify_begin_val()
        ann_loc_len = len(ann_loc_split)
        for i in range(1, ann_loc_len):
            ann_start, ann_end = ann_loc_split[i].split(' ')
            ann['start'] = int(ann_start)
            ann['end'] = int(ann_end)
            self.annotator_cont.words_single_annotate(
                arr_labels,
                arr_start_end,
                ann
            )
        if isinstance(self.labeler, LabelerIOB):
            self.labeler.undo_begin_val()


# Template pattern: subtoken label converters

class SubLabelConverter(ABC):
    """
    Abstract class that implements a template for converting word-level
    labels to subtoken-level.
    """
    @abstractmethod
    def convert_subtoken(
        self, word_label: str, n_subtoken: int
    ) -> List[Union[str, int]]:
        pass


class AllSubLabel(SubLabelConverter):
    """
    All subtokens obtained from the same word are assigned the
    word-level label.
    """
    @staticmethod
    def convert_subtoken(word_label: str, n_subtoken: int) -> List[str]:
        return [word_label] * n_subtoken


class FirstSubLabel(SubLabelConverter):
    """
    Only the first subtoken is assigned the word-level label.
    The remaining subtokens are assigned the subtoken_val default value.
    """
    def __init__(self, subtoken_val: Union[str, int]) -> None:
        self.subtoken_val = subtoken_val

    def convert_subtoken(
            self, word_label: str, n_subtoken: int
    ) -> List[Union[str, int]]:
        return [word_label] + [self.subtoken_val] * (n_subtoken - 1)


def convert_word_token(
        arr_word_text: List[str], arr_word_start_end: List[Tuple[int, int]],
        arr_word_labels: List[List[str]], tokenizer: Tokenizer, word_pos: int,
        sub_lab_converter: SubLabelConverter
) -> Tuple[List[str], List[Tuple[int, int]], List[List[str]], List[int]]:
    """
    Given a list of words, the function converts them to a list of subtokens.

    When converting the word-level start-end offsets to subtoken-level,
    the word start-end offset pair is assigned to all subtokens obtained
    from the same word.

    Additionally, a List[int] containing the id of the word each subtoken
    belongs to is returned.

    :param word_pos: id of the first input word
    """
    arr_subtok_text, arr_subtok_start_end, arr_subtok_word_id = [], [], []
    arr_subtok_labels = [[] for lab_i in range(len(arr_word_labels))]
    for i in range(len(arr_word_text)):
        w_text = arr_word_text[i]
        w_start_end = arr_word_start_end[i]
        subtok_text, _ = tokenizer.tokenize(
            word=w_text, start_i=w_start_end[0]
        )
        # using the word start-end pair as the start-end positions of
        # the subtokens
        subtok_start_end = [w_start_end] * len(subtok_text)
        subtok_word_id = [i + word_pos] * len(subtok_text)
        arr_subtok_text.extend(subtok_text)
        arr_subtok_start_end.extend(subtok_start_end)
        arr_subtok_word_id.extend(subtok_word_id)
        for lab_i in range(len(arr_word_labels)):
            w_label = arr_word_labels[lab_i][i]
            arr_subtok_labels[lab_i].extend(
                sub_lab_converter.convert_subtoken(
                    word_label=w_label, n_subtoken=len(subtok_text)
                )
            )

    return arr_subtok_text, arr_subtok_start_end, \
        arr_subtok_labels, arr_subtok_word_id


def create_subtoken_data(
        text: str, max_seq_len: int, tokenizer: Tokenizer, start_pos: int,
        df_ann: pd.DataFrame, annotator: Annotator,
        sub_lab_converter: SubLabelConverter, cased: bool = True,
        word_pos: int = 0
) -> Tuple[
    List[List[str]], List[List[Tuple[int, int]]],
    List[List[List[str]]], List[List[int]]
]:
    """
    Given an input text, it returns lists of lists containing the adjacent
    sequences of subtokens.

    :param df_ann: Dataframe containing all annotations from the input text

    :returns arr_subtoken, arr_start_end, arr_word_id
                shape: [n_sequences, n_subtokens]
             arr_labels
                shape: [n_labels, n_sequences, n_subtokens]
    """
    arr_subtoken, arr_start_end, arr_labels, arr_word_id = [], [], [], []
    # Apply whitespace and punctuation pre-tokenization to extract
    # the words from the input text
    arr_word, arr_word_start_end = word_start_end(
        text=text,
        start_i=start_pos,
        cased=cased
    )
    assert len(arr_word) == len(arr_word_start_end)
    # Obtain labels at word-level
    arr_word_labels = annotator.words_annotate(
        arr_start_end=np.array(arr_word_start_end),
        df_ann=df_ann
    )
    for lab_i in range(len(arr_word_labels)):
        assert len(arr_word_labels[lab_i]) == len(arr_word)

    # Convert word-level arrays to subtoken-level
    subtoken, start_end, labels, word_id = convert_word_token(
        arr_word_text=arr_word, arr_word_start_end=arr_word_start_end,
        arr_word_labels=arr_word_labels, tokenizer=tokenizer,
        word_pos=word_pos, sub_lab_converter=sub_lab_converter
    )

    assert len(subtoken) == len(start_end) == len(word_id)
    for lab_i in range(len(labels)):
        arr_labels.append([])
        assert len(labels[lab_i]) == len(subtoken)

    # Split large subtokens sequences
    for i in range(0, len(subtoken), max_seq_len):
        # n_sequences = math.ceil(len(subtoken) / max_seq_len)
        arr_subtoken.append(subtoken[i:i+max_seq_len])
        arr_start_end.append(start_end[i:i+max_seq_len])
        arr_word_id.append(word_id[i:i+max_seq_len])
        for lab_i in range(len(labels)):
            arr_labels[lab_i].append(labels[lab_i][i:i+max_seq_len])

    return arr_subtoken, arr_start_end, arr_labels, arr_word_id


def ss_create_subtoken_data(
        ss_start_end: List[Tuple[int, int]], max_seq_len: int, text: str,
        tokenizer: Tokenizer, df_ann: pd.DataFrame,
        annotator: Annotator, sub_lab_converter: SubLabelConverter, cased=True
) -> Tuple[
    List[List[str]], List[List[Tuple[int, int]]],
    List[List[List[str]]], List[List[int]]
]:
    """
    Function with the same functionality as create_subtoken_data,
    but considering the sentence split (SS) information provided as input.

    :param ss_start_end: (shape: [n_sentences, 2]) each tuple contains the
        start-end character positions pair of the split sentences from
        the input text
    """
    # Firstly, the whole text is tokenized (w/o splitting, see max_seq_len)
    txt_subtoken, txt_arr_start_end, txt_labels, txt_word_id = \
        create_subtoken_data(
            text=text, max_seq_len=int(1e22),
            tokenizer=tokenizer, start_pos=0, df_ann=df_ann,
            annotator=annotator,
            sub_lab_converter=sub_lab_converter,
            cased=cased, word_pos=0
        )
    # Since no splitting was performed, remove the first dimension
    # of the sequences
    assert len(txt_subtoken) == 1
    txt_subtoken, txt_arr_start_end, txt_word_id = \
        txt_subtoken[0], txt_arr_start_end[0], txt_word_id[0]
    for lab_i in range(len(txt_labels)):
        txt_labels[lab_i] = txt_labels[lab_i][0]

    # Then, the obtained sequences are split according to the SS information
    arr_txt_start_end = np.array(txt_arr_start_end)
    ss_subtoken, ss_arr_start_end, ss_word_id = [], [], []
    ss_labels = [[] for _ in range(len(txt_labels))]
    start_tok = last_tok = 0
    for _, ss_end in ss_start_end:
        # never reach the end of the sequence w/o finding all sentences
        assert start_tok < len(txt_subtoken)
        # Identify the position of the last subtoken of the current sentence
        last_tok = np.where(
            arr_txt_start_end[start_tok:, 1] <= ss_end
        )[0][-1] + start_tok  # (last subtoken <= sentence end) + start_tok
        # Add the current sentence
        ss_subtoken.append(txt_subtoken[start_tok:last_tok+1])
        ss_arr_start_end.append(txt_arr_start_end[start_tok:last_tok+1])
        ss_word_id.append(txt_word_id[start_tok:last_tok+1])
        for lab_i in range(len(ss_labels)):
            ss_labels[lab_i].append(txt_labels[lab_i][start_tok:last_tok+1])
        start_tok = last_tok + 1
    # Finally, split large SS-subtokens sequences
    arr_subtoken, arr_start_end, arr_word_id = [], [], []
    arr_labels = [[] for _ in range(len(ss_labels))]
    for i in range(len(ss_subtoken)):
        for j in range(0, len(ss_subtoken[i]), max_seq_len):
            arr_subtoken.append(ss_subtoken[i][j:j+max_seq_len])
            arr_start_end.append(ss_arr_start_end[i][j:j+max_seq_len])
            arr_word_id.append(ss_word_id[i][j:j+max_seq_len])
            for lab_i in range(len(arr_labels)):
                arr_labels[lab_i].append(ss_labels[lab_i][i][j:j+max_seq_len])

    return arr_subtoken, arr_start_end, arr_labels, arr_word_id


def fragment_greedy_data(
        arr_subtoken: List[List[str]],
        arr_start_end: List[List[Tuple[int, int]]],
        arr_labels: List[List[List[str]]],
        arr_word_id: List[List[int]], max_seq_len: int
) -> Tuple[
    List[List[str]], List[List[Tuple[int, int]]],
    List[List[List[str]]], List[List[int]]
]:
    """
    Implementation of the multiple-sentence fine-tuning approach developed in
    http://ceur-ws.org/Vol-2664/cantemist_paper15.pdf, which consists in
    generating text fragments containing the maximum number of adjacent
    sequences, such that the length of each fragment is <= max_seq_len.
    """
    frag_subtoken, frag_start_end, frag_word_id = [[]], [[]], [[]]
    frag_labels = [[[]] for lab_i in range(len(arr_labels))]
    i = 0
    while i < len(arr_subtoken):
        assert len(arr_subtoken[i]) <= max_seq_len
        if len(frag_subtoken[-1]) + len(arr_subtoken[i]) > max_seq_len:
            # Fragment is full, so create a new empty fragment
            frag_subtoken.append([])
            frag_start_end.append([])
            frag_word_id.append([])
            for lab_i in range(len(arr_labels)):
                frag_labels[lab_i].append([])

        frag_subtoken[-1].extend(arr_subtoken[i])
        frag_start_end[-1].extend(arr_start_end[i])
        frag_word_id[-1].extend(arr_word_id[i])
        for lab_i in range(len(arr_labels)):
            frag_labels[lab_i][-1].extend(arr_labels[lab_i][i])

        i += 1

    return frag_subtoken, frag_start_end, frag_labels, frag_word_id


def create_input_data(
        df_text: pd.DataFrame, text_col: str, df_ann: pd.DataFrame,
        arr_doc: Iterable[str], tokenizer: Tokenizer,
        arr_lab_encoder: List[Dict[str, int]], seq_len: int,
        annotator: Annotator, sub_lab_converter: SubLabelConverter,
        ss_dict: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        greedy: bool = False, cased: bool = True
) -> Tuple[
    Dict[str, np.ndarray], np.ndarray, np.ndarray,
    List[List[Tuple[int, int]]], List[List[int]]
]:
    """
    Given a collection of input documents, this function generates the data
    needed to train a model on a token classification task.

    :param df_text: Dataframe containing the ID ("doc_id" column expected)
        and the text of each input document
    :param text_col: name of the column of df_text that contains the texts
    :param df_ann: Dataframe containing the annotations of the documents
    :param arr_doc: it contains the documents IDs to be considered.
        df_text, df_ann and ss_dict are expected to contain all documents
        present in arr_doc
    :param arr_lab_encoder: it contains the encoders (dict) that convert
        labels (str) to its encoded version (int)
        shape: [n_labels]
    :param ss_dict: keys are documents IDs and each value is a list of tuples
        containing the start-end char positions pairs of the split sentences
        (SS) in each document. If None, the function implements the text-stream
        fragment-based approach described in
        https://doi.org/10.1109/ACCESS.2021.3080085
    :param greedy: if False, the function implements the single-sentence
        approach described in https://doi.org/10.1109/ACCESS.2021.3080085

    :returns dict_token
                shape: [n_inputs (keys), [n_sequences, seq_len] (values)]
             labels
                shape: [n_labels, n_sequences, seq_len]
             n_fragments
                shape: [n_docs]
             start_end_offsets
                shape: [n_sequences, n_subtokens, 2]
             word_ids
                shape: [n_sequences, n_subtokens]
    """
    dict_token, labels, n_fragments, start_end_offsets, word_ids = \
        {}, [], [], [], []
    sub_tok_max_seq_len = tokenizer.get_seq_len(seq_len)
    for doc in arr_doc:
        # Extract doc annotations
        doc_ann = df_ann[df_ann["doc_id"] == doc]
        # Extract doc text
        doc_text = df_text[df_text["doc_id"] == doc][text_col].values[0]
        # Generate annotated subtokens sequences
        if ss_dict is not None:
            # Split subtokens sequence according to SS information
            doc_ss = ss_dict[doc]  # SS start-end pairs of the doc text
            doc_ss_token, doc_ss_start_end, doc_ss_label, doc_ss_word_id =\
                ss_create_subtoken_data(
                    ss_start_end=doc_ss, max_seq_len=sub_tok_max_seq_len,
                    text=doc_text, tokenizer=tokenizer, df_ann=doc_ann,
                    annotator=annotator,
                    sub_lab_converter=sub_lab_converter, cased=cased
                )
            assert len(doc_ss_token) == len(doc_ss_start_end) == \
                len(doc_ss_word_id)
            for lab_i in range(len(doc_ss_label)):
                assert len(doc_ss_label[lab_i]) == len(doc_ss_token)

            if greedy:
                # Group the sequences of subtokens sentences into sequences
                # comprising multiple sentences
                frag_token, frag_start_end, frag_label, frag_word_id = \
                    fragment_greedy_data(
                        arr_subtoken=doc_ss_token,
                        arr_start_end=doc_ss_start_end,
                        arr_labels=doc_ss_label,
                        arr_word_id=doc_ss_word_id,
                        max_seq_len=sub_tok_max_seq_len
                    )
            else:
                frag_token = deepcopy(doc_ss_token)
                frag_start_end = deepcopy(doc_ss_start_end)
                frag_label = deepcopy(doc_ss_label)
                frag_word_id = deepcopy(doc_ss_word_id)
        else:
            # Generate annotated sequences using text-stream strategy
            # (w/o considering SS)
            frag_token, frag_start_end, frag_label, frag_word_id = \
                create_subtoken_data(
                    text=doc_text, max_seq_len=sub_tok_max_seq_len,
                    tokenizer=tokenizer, start_pos=0, df_ann=doc_ann,
                    annotator=annotator, sub_lab_converter=sub_lab_converter,
                    cased=cased, word_pos=0
                )

        assert len(frag_token) == len(frag_start_end) == len(frag_word_id)
        for lab_i in range(len(frag_label)):
            assert len(frag_label[lab_i]) == len(frag_token)
        # Store the start-end char positions of all the sequences
        start_end_offsets.extend(frag_start_end)
        # Store the subtokens word ids of all the sequences
        word_ids.extend(frag_word_id)
        # Store the number of sequences of each doc text
        n_fragments.append(len(frag_token))
        # Subtokens sequences formatting
        if len(labels) == 0:
            # first iteration (dirty, number of labels not previously defined)
            labels = [[] for lab_i in range(len(frag_label))]
        for seq_i in range(len(frag_token)):
            f_token = frag_token[seq_i]
            f_start_end = frag_start_end[seq_i]
            f_word_id = frag_word_id[seq_i]
            f_label = []
            for lab_i in range(len(frag_label)):
                f_label.append(frag_label[lab_i][seq_i])
            assert len(f_token) == len(f_start_end) == len(f_word_id) \
                <= sub_tok_max_seq_len
            for lab_i in range(len(f_label)):
                assert len(f_label[lab_i]) == len(f_token)
            f_dict_tok, f_label = tokenizer.format_token_data(
                arr_token=f_token, arr_label=f_label,
                arr_lab_encoder=arr_lab_encoder, seq_len=seq_len,
            )
            if not dict_token:
                for key_tok in f_dict_tok:
                    dict_token[key_tok] = [f_dict_tok[key_tok]]
            else:
                for key_tok in dict_token:
                    dict_token[key_tok].append(f_dict_tok[key_tok])
            for lab_i in range(len(f_label)):
                labels[lab_i].append(f_label[lab_i])
    # Convert list to np.ndarray
    for key_tok in dict_token:
        dict_token[key_tok] = np.array(dict_token[key_tok])

    return dict_token, np.array(labels), \
        np.array(n_fragments), start_end_offsets, word_ids
