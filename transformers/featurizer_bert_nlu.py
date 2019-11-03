

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import torch
import numpy as np

from transformers.featurizer_bert import BertFeaturizer

logger = logging.getLogger(__name__)


class BertNLUFeaturizer(BertFeaturizer):
	def __init__(self, **kwargs):
		super(BertNLUFeaturizer, self).__init__(**kwargs)
		pass

	def _get_bag_of_tokens_labels(self, inputs, tokenizer):
		indices = [list(set(features)) for features in inputs.tolist()]
		bag_of_tokens_label = np.zeros((inputs.shape[0], tokenizer.vocab_size))
		for (row, row_indices) in zip(bag_of_tokens_label, indices):
			row[row_indices] = 1

		bag_of_tokens_label = torch.FloatTensor(bag_of_tokens_label)
		return bag_of_tokens_label

	def featurize(self, inputs, tokenizer, args, config, **kwargs):
		inputs, outputs = super(BertNLUFeaturizer, self).featurize(inputs, tokenizer, args, config, **kwargs)
		bag_of_tokens_labels = self._get_bag_of_tokens_labels(inputs, tokenizer)
		return inputs, (*outputs, bag_of_tokens_labels)