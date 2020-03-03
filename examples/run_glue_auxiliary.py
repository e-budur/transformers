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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function
from transformers import AutoConfig, AutoModel, AutoTokenizer
import logging

import run_glue

import transformers.data.processors.glue_auxiliary
from transformers.configuration_bert_nlu import BertConfig
from transformers.configuration_bert_nlu import BertNLUConfig
from transformers.modeling_bert_nlu import BertNLUForSequenceClassification
from transformers.tokenization_bert_nlu import BertNLUTokenizer
from transformers.modeling_bert import BertForSequenceClassification

logger = logging.getLogger(__name__)


if __name__ == "__main__":
   run_glue.MODEL_CLASSES['bert-nlu'] = (BertNLUConfig, BertNLUForSequenceClassification, BertNLUTokenizer)
   run_glue.MODEL_CLASSES['bert-auto'] = (AutoConfig, BertForSequenceClassification, AutoTokenizer)

   run_glue.main()
