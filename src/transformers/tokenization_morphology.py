# coding=utf-8
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Tokenization classes for ALBERT model."""


import logging
import unicodedata
from shutil import copyfile

from .tokenization_utils import PreTrainedTokenizer
from transformers.custom_tokenizers.ZemberekTokenizer import  *
from transformers.custom_tokenizers.CustomTokenizerUtils import *
logger = logging.getLogger(__name__)

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "berturk-morphed-v1": 512
}

class BertMorphologyTokenizer(PreTrainedTokenizer):
    """
        Morphology based tokenizer.
    """


    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    VOCAB_FILES_NAMES = {'vocab_file':'vocab.txt'}
    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        remove_space=True,
        keep_accents=True,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        self.max_len_sentences_pair = self.max_len - 3  # take into account special tokens
        self.keep_accents = keep_accents
        self.remove_space = remove_space

        self.do_lower_case = do_lower_case
        self.vocab_file = vocab_file
        self.omit_suffixes = kwargs.get('omit_suffixes', False)
        self.java_home_path = kwargs.get('java_home_path', None)
        self.zemberek_path = kwargs.get('zemberek_path', None)
        print(u"omit_suffixes:{}\njava_home_path:{}\nzemberek_path:{}\nvocab_file:{}\n".format(
            self.omit_suffixes,
            self.java_home_path,
            self.zemberek_path,
            self.vocab_file))
        self.vocab_file = vocab_file
        self.init_morphological_tokenizer()

    @property
    def vocab_size(self):
        return len(self.vocab.keys())

    def __getstate__(self):
        state = self.__dict__.copy()
        state["vocab"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.vocab = load_vocab(self.vocab_file)
        self.init_morphological_tokenizer()

    def init_morphological_tokenizer(self):
        self.vocab = load_vocab(self.vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.zemberek_tokenizer = ZemberekTokenizer(vocab=self.vocab,
                                                    lower_case=self.do_lower_case,
                                                    omit_suffixes=self.omit_suffixes,
                                                    java_home_path=self.java_home_path,
                                                    zemberek_path=self.zemberek_path)

    def preprocess_text(self, inputs):
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace("``", '"').replace("''", '"')

        if not self.keep_accents:
            outputs = unicodedata.normalize("NFKD", outputs)
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def _tokenize(self, input_text, sample=False):
        """ Tokenize a string. """
        text = self.preprocess_text(input_text)
        tokens = self.zemberek_tokenizer.tokenize(text)

        # added by e-budur for logging some samples from the preprocessed inputs
        if random.random() < 0.01:  # print some examples of the preprocessed sentences
            print(u"\n{}\nOriginal line: {}\nProcessed line: {}\n{}\n ".format(
                u"================================= MORPHOLOGY PROCESSED EXAMPLE ===================================",
                input_text.strip(),
                ' '.join(tokens),
                u"==========================================================================================")
            )

        return tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.unk_token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.inv_vocab.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        An MorphologicalTokenizer sequence has the following format:
            single sequence: [CLS] X [SEP]
            pair of sequences: [CLS] A [SEP] B [SEP]
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 0 for a special token, 1 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        An MorphologicalTokenizer sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence

        if token_ids_1 is None, only returns the first portion of the mask (0's).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory):
        """ Save the sentencepiece vocabulary (copy original file) and special tokens file
            to a directory.
        """
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        out_vocab_file = os.path.join(save_directory, self.VOCAB_FILES_NAMES["vocab_file"])

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
