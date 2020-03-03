# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" BERT model configuration """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import sys
from io import open
from transformers.configuration_bert import BertConfig

logger = logging.getLogger(__name__)


class BertNLUConfig(BertConfig):

    def __init__(self,
                 vocab_size_or_config_json_file=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=1024, # This is the only difference compared to BertConfig class
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 **kwargs):
        super(BertNLUConfig, self).__init__(vocab_size_or_config_json_file,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads,
                 intermediate_size,
                 hidden_act,
                 hidden_dropout_prob,
                 attention_probs_dropout_prob,
                 max_position_embeddings,
                 type_vocab_size,
                 initializer_range,
                 layer_norm_eps,
                 **kwargs)


class BertForJointUnderstandingConfig(BertConfig):

    def __init__(self,
                 vocab_size_or_config_json_file=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,  # This is the only difference compared to BertConfig class
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 num_intent_labels=2,
                 num_enumerable_entity_labels=2,
                 num_non_enumerable_entity_labels=2,
                 **kwargs):
        super(BertForJointUnderstandingConfig, self).__init__(vocab_size_or_config_json_file,
                                            hidden_size,
                                            num_hidden_layers,
                                            num_attention_heads,
                                            intermediate_size,
                                            hidden_act,
                                            hidden_dropout_prob,
                                            attention_probs_dropout_prob,
                                            max_position_embeddings,
                                            type_vocab_size,
                                            initializer_range,
                                            layer_norm_eps,
                                            **kwargs)

        self.num_intent_labels = num_intent_labels
        self.num_enumerable_entity_labels = num_enumerable_entity_labels
        self.num_non_enumerable_entity_labels = num_non_enumerable_entity_labels


class BertNLUForJointUnderstandingConfig(BertForJointUnderstandingConfig):

    def __init__(self,
                 vocab_size_or_config_json_file=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,  # This is the only difference compared to BertConfig class
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 num_intent_labels=2,
                 num_enumerable_entity_labels=2,
                 num_non_enumerable_entity_labels=2,
                 **kwargs):
        super(BertNLUForJointUnderstandingConfig, self).__init__(vocab_size_or_config_json_file,
                                            hidden_size,
                                            num_hidden_layers,
                                            num_attention_heads,
                                            intermediate_size,
                                            hidden_act,
                                            hidden_dropout_prob,
                                            attention_probs_dropout_prob,
                                            max_position_embeddings,
                                            type_vocab_size,
                                            initializer_range,
                                            layer_norm_eps,
                                            num_intent_labels,
                                            num_enumerable_entity_labels,
                                            num_non_enumerable_entity_labels,
                                            **kwargs)