# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes."""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
import unicodedata
from io import open
from transformers.tokenization_bert import *

from .tokenization_utils import PreTrainedTokenizer
import six
logger = logging.getLogger(__name__)

from .file_utils import cached_path, is_tf_available, is_torch_available

if is_tf_available():
    import tensorflow as tf
if is_torch_available():
    import torch

class BertNLUTokenizer(BertTokenizer):


    def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True, never_split=None,
                 unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]",
                 mask_token="[MASK]", tokenize_chinese_chars=True, mlb_token="[MLB]", mlb_token_id=104, **kwargs):
        """Constructs a BertTokenizer.

        Args:
            **vocab_file**: Path to a one-wordpiece-per-line vocabulary file
            **do_lower_case**: (`optional`) boolean (default True)
                Whether to lower case the input
                Only has an effect when do_basic_tokenize=True
            **do_basic_tokenize**: (`optional`) boolean (default True)
                Whether to do basic tokenization before wordpiece.
            **never_split**: (`optional`) list of string
                List of tokens which will never be split during tokenization.
                Only has an effect when do_basic_tokenize=True
            **tokenize_chinese_chars**: (`optional`) boolean (default True)
                Whether to tokenize Chinese characters.
                This should likely be deactivated for Japanese:
                see: https://github.com/huggingface/pytorch-pretrained-BERT/issues/328
        """

        super(BertNLUTokenizer, self).__init__(vocab_file, do_lower_case, do_basic_tokenize, never_split,
                 unk_token, sep_token, pad_token, cls_token,
                 mask_token, tokenize_chinese_chars, **kwargs)

        self._mlb_token = mlb_token
        if self._mlb_token is not None:
            self.SPECIAL_TOKENS_ATTRIBUTES.append("mlb_token")
            self.vocab[self._mlb_token] = mlb_token_id   # applied variable shadowing on purpose
            logger.info("The mapped id for the special token %s is %s.", self._mlb_token, self.mlb_token_id)
        else:
            logger.warning("mlb_token is not given as an input")

        self.max_len_single_sentence = self.max_len - 4  # take into account special tokens
        self.max_len_sentences_pair = self.max_len - 5  # take into account special tokens




    @property
    def mlb_token_id(self):
        """ Id of the multi label token in the vocabulary. E.g. to extract a summary of an bag of tokens leveraging self-attention along the full depth of the model. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self._mlb_token)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:
            single sequence: [CLS] X [SEP] X [MLB] X [SEP]
            pair of sequences: [CLS] A [SEP] B [SEP] [MLB] [SEP]
        """

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + [self.mlb_token_id]
        cls = [self.cls_token_id]
        mlb = [self.mlb_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep + mlb

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        mlb = [self.mlb_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep + mlb + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep + mlb + sep) * [1]

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
                raise ValueError("You should not supply a second sequence if the provided sequence of "
                                 "ids is already formated with special tokens for the model.")
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id, self.mlb_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1, 1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1]



    def encode_nlu(self, utterance, utterance_tokens, add_special_tokens=True,
                    max_length=None,
                    stride=0,
                    truncation_strategy='longest_first',
                    return_tensors=None,
                    **kwargs):

        def get_input_ids(text):
            if isinstance(text, six.string_types):
                return self.convert_tokens_to_ids(self.tokenize(text, **kwargs))
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], six.string_types):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.")

        utterance_token_ids = get_input_ids(utterance_tokens)

        return self.prepare_for_model_nlu(utterance_token_ids,
                                      max_length=max_length,
                                      add_special_tokens=add_special_tokens,
                                      stride=stride,
                                      truncation_strategy=truncation_strategy,
                                      return_tensors=return_tensors)


    def prepare_for_model_nlu(self, ids, max_length=None, add_special_tokens=True, stride=0,
                          truncation_strategy='longest_first', return_tensors=None):

        len_ids = len(ids)

        encoded_inputs = {}
        total_len = len_ids + (self.num_added_tokens(pair=False) if add_special_tokens else 0)
        if max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(ids, pair_ids=None,
                                                                        num_tokens_to_remove=total_len-max_length,
                                                                        truncation_strategy=truncation_strategy,
                                                                        stride=stride)
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, token_ids_1=None)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, token_ids_1=None)
            encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, token_ids_1=None)
        else:
            sequence = ids
            token_type_ids = [0] * len(ids)

        if return_tensors == 'tf' and is_tf_available():
            sequence = tf.constant([sequence])
            token_type_ids = tf.constant([token_type_ids])
        elif return_tensors == 'pt' and is_torch_available():
            sequence = torch.tensor([sequence])
            token_type_ids = torch.tensor([token_type_ids])
        elif return_tensors is not None:
            logger.warning("Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(return_tensors))

        encoded_inputs["input_ids"] = sequence
        encoded_inputs["token_type_ids"] = token_type_ids

        if max_length and len(encoded_inputs["input_ids"]) > max_length:
            encoded_inputs["input_ids"] = encoded_inputs["input_ids"][:max_length]
            encoded_inputs["token_type_ids"] = encoded_inputs["token_type_ids"][:max_length]
            encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"][:max_length]

        return encoded_inputs

