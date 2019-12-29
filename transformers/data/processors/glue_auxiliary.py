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
""" GLUE processors and helpers """

import logging
import os

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features


from transformers.data.processors.glue import MnliProcessor


from .utils import DataProcessor, InputExample, InputFeatures
from ...file_utils import is_tf_available
import json
import codecs

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


class SnliProcessor(MnliProcessor):
    """Processor for the SNLI as an auxiliary dataaset (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        # HACK:Since the implementation for testing the test set is not currently available yet in the framework,
        # we used test dataset during the evaluation to see the performance of the model on the test set.
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class XnliProcessor(MnliProcessor):
    """Processor for the XNLI as an auxiliary dataset (Original version)."""

    def __init__(self, lang='en'):
        self.lang = lang

    def get_dev_examples(self, data_dir):
        """See base class."""
        # HACK:Since the implementation for testing the test set is not currently available yet in the framework,
        # we used test dataset during the evaluation to see the performance of the model on the test set.
        return self._create_dev_examples(
            self._read_tsv(os.path.join(data_dir, "xnli.test.tsv")),
            "test")

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def _create_dev_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            lang = line[0]
            if lang != self.lang:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[6]
            text_b = line[7]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



class MnliNMTProcessor(DataProcessor):
    """Processor for the MultiNLI-NMT as an auxiliary dataaset (GLUE version)."""

    def __init__(self, train_filename, dev_filename):
        self.train_filename = train_filename
        self.dev_filename = dev_filename


    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_from_json(
            self._read_json(os.path.join(data_dir, self.dev_filename)),
            "dev")

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples_from_json(
            self._read_json(os.path.join(data_dir, self.train_filename)),
            "train")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples_from_json(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data_item) in enumerate(data):
            if i == 0:
                continue
            if data_item['gold_label'] == '-': #skip data_items with missing gold labels
                continue
            guid = "%s-%s" % (set_type, data_item['pairID'])
            text_a = data_item['translate-sentence1']
            text_b = data_item['translate-sentence2']
            label = data_item['gold_label']
            if label == '-': #skip data_items with missing gold labels
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _read_json(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with codecs.open(input_file, encoding='utf-8') as json_file:
            data = json.load(json_file)
            return data

class MnliNMTMatchedProcessor(MnliNMTProcessor):
    """Processor for the MultiNLI NMT Matched data set."""
    def __init__(self):
        super().__init__('multinli_train_translation.json', 'multinli_dev_matched_translation.json')


class MnliNMTMismatchedProcessor(MnliNMTProcessor):
    """Processor for the MultiNLI NMT Mismatched data set."""
    def __init__(self):
        super().__init__('multinli_train_translation.json', 'multinli_dev_mismatched_translation.json')


class SnliNMTProcessor(MnliNMTProcessor):
    """Processor for the SNLI NMT Matched data set."""
    def __init__(self):
        super().__init__('snli_train_translation.json', 'snli_dev_translation.json')

class XnliNMTProcessor(MnliNMTProcessor):
    """Processor for the XNLI NMT Matched data set."""
    def __init__(self, lang='tr'):
        super().__init__('multinli_train_translation.json', 'xnli_dev_translation.json')
        self.lang=lang

    def _create_examples_from_json(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data_item) in enumerate(data):
            if i == 0:
                continue
            if data_item['gold_label'] == '-': #skip data_items with missing gold labels
                continue
            if data_item['language'] != self.lang:
                continue
            guid = "%s-%s" % (set_type, data_item['pairID'])
            text_a = data_item['translate-sentence1']
            text_b = data_item['translate-sentence2']
            label = data_item['gold_label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

processors['snli'] = SnliProcessor
output_modes['snli'] = "classification"
processors['xnli'] = XnliProcessor
output_modes['xnli'] = "classification"
processors['mnli-nmt-amzn-tr'] = MnliNMTMatchedProcessor
output_modes['mnli-nmt-amzn-tr'] = "classification"
processors['mnli-mm-nmt-amzn-tr'] = MnliNMTMismatchedProcessor
output_modes['mnli-mm-nmt-amzn-tr'] = "classification"
processors['snli-nmt-amzn-tr'] = SnliNMTProcessor
output_modes['snli-nmt-amzn-tr'] = "classification"
processors['xnli-nmt-amzn-tr'] = XnliNMTProcessor
output_modes['xnli-nmt-amzn-tr'] = "classification"