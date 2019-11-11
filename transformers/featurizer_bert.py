

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)

class PreTrainedFeaturizer(object):
	def __init__(self, **kwargs):
		pass

	def featurize(self, inputs, **kwargs):
		pass


class BertFeaturizer(PreTrainedFeaturizer):
	def __init__(self, **kwargs):
		super(BertFeaturizer, self).__init__(**kwargs)

	def _get_examples(self, inputs, tokenizer, args):
		masked_lm_labels = inputs.clone()

		# We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
		probability_matrix = torch.full(masked_lm_labels.shape, args.mlm_probability)
		special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
							   masked_lm_labels.tolist()]
		probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
		masked_indices = torch.bernoulli(probability_matrix).bool()
		masked_lm_labels[~masked_indices] = -1  # We only compute loss on masked tokens

		# 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
		indices_replaced = torch.bernoulli(torch.full(masked_lm_labels.shape, 0.8)).bool() & masked_indices
		inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

		# 10% of the time, we replace masked input tokens with random word
		indices_random = torch.bernoulli(
			torch.full(masked_lm_labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
		random_words = torch.randint(len(tokenizer), masked_lm_labels.shape, dtype=torch.long)
		inputs[indices_random] = random_words[indices_random]

		return inputs, masked_lm_labels

	def _get_positive_examples(self, inputs, tokenizer, args, config):
		pos_inputs, pos_masked_lm_labels = self._get_examples(inputs, tokenizer, args)
		pos_next_sentence_label = torch.LongTensor(np.tile([1], (pos_inputs.shape[0], 1)))
		return pos_inputs, pos_next_sentence_label, pos_masked_lm_labels

	def _get_rotated_index(self, max_index, rotation_count):
		first_part = np.arange(0, rotation_count).tolist()
		second_part = np.arange(rotation_count, max_index).tolist()
		# concat the indices in fipped order such that the resulting index becomes rotated
		indices = second_part + first_part
		return indices

	def _get_negative_examples(self, inputs, tokenizer, args, config):
		neg_inputs, neg_masked_lm_labels = self._get_examples(inputs, tokenizer, args)

		sent_len = int((config.max_position_embeddings - 4) / 2)
		sent1_start_pos = 2
		sent1_end_pos = int(sent1_start_pos + sent_len)

		sent2_start_pos = int(sent1_end_pos + 1)
		sent2_end_pos = int(sent2_start_pos + sent_len)

		sent2_parts = neg_inputs[:, sent2_start_pos:sent2_end_pos + 1]

		row_indices = self._get_rotated_index(sent2_parts.shape[0], rotation_count=2)
		sent2_parts = sent2_parts[row_indices, :]
		neg_inputs[:, sent2_start_pos:sent2_end_pos + 1] = sent2_parts

		neg_next_sentence_label = torch.LongTensor(np.tile([0], (neg_inputs.shape[0], 1)))
		return neg_inputs, neg_next_sentence_label, neg_masked_lm_labels

	def featurize(self, inputs, tokenizer, args, config, **kwargs):
		super(BertFeaturizer, self).featurize(inputs, **kwargs)
		pos_inputs, pos_next_sentence_label, pos_masked_lm_labels = self._get_positive_examples(
			inputs, tokenizer, args, config)
		neg_inputs, neg_next_sentence_label, neg_masked_lm_labels = self._get_negative_examples(
			inputs, tokenizer, args, config)

		inputs = torch.cat([pos_inputs, neg_inputs], dim=0)
		next_sentence_label = torch.cat([pos_next_sentence_label, neg_next_sentence_label], dim=0)
		masked_lm_labels = torch.cat([pos_masked_lm_labels, neg_masked_lm_labels], dim=0)

		#adding optional parameters as the expected output params
		attention_mask = None
		token_type_ids = None
		position_ids = None
		head_mask = None

		outputs = (attention_mask, token_type_ids, position_ids, head_mask, None, masked_lm_labels, next_sentence_label)
		return inputs, outputs