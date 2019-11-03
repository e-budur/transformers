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

import argparse
import glob
import logging
import os
import random
import pickle
import shutil
import codecs
import os
import re
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
from transformers.tokenization_bert_nlu import BertNLUTokenizer
from transformers.modeling_bert_nlu import BertNLUForPreTraining
from transformers.configuration_bert_nlu import BertNLUConfig

from torch.utils.data import DataLoader, Dataset

from transformers import (WEIGHTS_NAME, BertConfig)

from transformers import AdamW, WarmupLinearSchedule

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

logger = logging.getLogger(__name__)




MODEL_CLASSES = {
    'bert-nlu': (BertNLUConfig, BertNLUForPreTraining, BertNLUTokenizer)
}


class TextDataset(Dataset):
	def __init__(self, tokenizer, input_data_dir='train', block_size=512):

		cached_features_file = os.path.join(input_data_dir, 'cached_features')

		if os.path.exists(cached_features_file):
			logger.info("Loading features from cached file %s", cached_features_file)
			with open(cached_features_file, 'rb') as handle:
				self.examples = pickle.load(handle)
		else:
			logger.info("Creating features from data directory at %s", input_data_dir)
			self.examples = []
			files = glob.glob(input_data_dir + "/*txt")

			for file in tqdm(files, desc="read files"):
				with open(file, 'r', encoding='utf-8') as fin:
					prev_sent = None
					for cur_line in fin:
						if len(cur_line.strip())==0:
							continue
						cur_sentences = cur_line.split('.')
						for cur_sent in cur_sentences:
							cur_sent = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cur_sent.strip()))
							if len(cur_sent) == 0:
								continue
							if len(cur_sent)>block_size:
								cur_sent = cur_sent[:block_size]
							elif len(cur_sent)<block_size:
								N = block_size - len(cur_sent)
								cur_sent = np.pad(cur_sent, (0, N), 'constant').tolist()

							if prev_sent is None and len(cur_sent)>0:
								prev_sent = cur_sent
								continue

							self.examples.append(tokenizer.build_inputs_with_special_tokens(prev_sent, cur_sent))
							prev_sent = cur_sent

			logger.info("Saving features into cached file %s", cached_features_file)

			with open(cached_features_file, 'wb') as handle:
				pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, item):
		return torch.tensor(self.examples[item])


def load_and_cache_examples(args, tokenizer):
	dataset = TextDataset(tokenizer, input_data_dir=args.input_data_dir, block_size=args.block_size)
	return dataset


def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
	if not args.save_total_limit:
		return
	if args.save_total_limit <= 0:
		return

	# Check if we should delete older checkpoint(s)
	glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
	if len(glob_checkpoints) <= args.save_total_limit:
		return

	ordering_and_checkpoint_path = []
	for path in glob_checkpoints:
		if use_mtime:
			ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
		else:
			regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
			if regex_match and regex_match.groups():
				ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

	checkpoints_sorted = sorted(ordering_and_checkpoint_path)
	checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
	number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
	checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
	for checkpoint in checkpoints_to_be_deleted:
		logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
		shutil.rmtree(checkpoint)


def get_examples(inputs, tokenizer, args):
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

	indices = [list(set(features)) for features in inputs.tolist()]

	bag_of_tokens_label = np.zeros((inputs.shape[0], tokenizer.vocab_size))
	for (row, row_indices) in zip(bag_of_tokens_label, indices):
		row[row_indices] = 1

	bag_of_tokens_label = torch.FloatTensor(bag_of_tokens_label)

	return inputs, masked_lm_labels, bag_of_tokens_label

def get_positive_examples(inputs, tokenizer, args, config):
	pos_inputs, pos_masked_lm_labels, pos_bag_of_tokens_label = get_examples(inputs, tokenizer, args)
	pos_next_sentence_label = torch.LongTensor(np.tile([1], (inputs.shape[0], 1)))
	return pos_inputs, pos_next_sentence_label, pos_masked_lm_labels, pos_bag_of_tokens_label

def get_negative_examples(inputs, tokenizer, args, config):
	neg_inputs, neg_masked_lm_labels, neg_bag_of_tokens_label = get_examples(inputs, tokenizer, args)


	sent_len = int((config.max_position_embeddings-4)/2)
	sent1_start_pos = 2
	sent1_end_pos = int(sent1_start_pos+sent_len)

	sent2_start_pos = int(sent1_end_pos+1)
	sent2_end_pos = int(sent2_start_pos+sent_len)

	sent2_parts = neg_inputs[:, sent2_start_pos:sent2_end_pos+1]

	row_indices = np.arange(sent2_parts.shape[0]).tolist()
	row_indices.pop(0)
	row_indices.append(0)
	sent2_parts = sent2_parts[row_indices,:]
	neg_inputs[:, sent2_start_pos:sent2_end_pos + 1] = sent2_parts

	neg_next_sentence_label = torch.LongTensor(np.tile([0], (inputs.shape[0], 1)))
	return neg_inputs, neg_next_sentence_label, neg_masked_lm_labels, neg_bag_of_tokens_label

def featurize_input(inputs, tokenizer, args, config):
	pos_inputs, pos_next_sentence_label, pos_masked_lm_labels, pos_bag_of_tokens_label = get_positive_examples(inputs, tokenizer, args, config)
	neg_inputs, neg_next_sentence_label, neg_masked_lm_labels, neg_bag_of_tokens_label = get_negative_examples(inputs, tokenizer, args, config)

	inputs = torch.cat([pos_inputs,  neg_inputs], dim=0)
	next_sentence_label = torch.cat([pos_next_sentence_label, neg_next_sentence_label], dim=0)
	masked_lm_labels = torch.cat([pos_masked_lm_labels, neg_masked_lm_labels], dim=0)
	bag_of_tokens_label = torch.cat([pos_bag_of_tokens_label, neg_bag_of_tokens_label], dim=0)

	return inputs, next_sentence_label, masked_lm_labels, bag_of_tokens_label



def train(args, train_dataset, model, tokenizer, config):
	""" Train the model """
	if args.local_rank in [-1, 0]:
		tb_writer = SummaryWriter()

	args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

	if args.max_steps > 0:
		t_total = args.max_steps
		args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
	else:
		t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
		 'weight_decay': args.weight_decay},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
	scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
	if args.fp16:
		try:
			from apex import amp
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
		model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

	# multi-gpu training (should be after apex fp16 initialization)
	if args.n_gpu > 1:
		model = torch.nn.DataParallel(model)

	# Distributed training (should be after apex fp16 initialization)
	if args.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
														  output_device=args.local_rank,
														  find_unused_parameters=True)

	# Train!
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
	logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
				args.train_batch_size * args.gradient_accumulation_steps * (
					torch.distributed.get_world_size() if args.local_rank != -1 else 1))
	logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", t_total)

	global_step = 0
	tr_loss, logging_loss = 0.0, 0.0
	model.zero_grad()
	train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
	set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
	for _ in train_iterator:
		epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
		for step, batch in enumerate(epoch_iterator):
			inputs, next_sentence_label, masked_lm_labels, bag_of_tokens_label = featurize_input(batch, tokenizer, args, config)
			inputs = inputs.to(args.device)
			masked_lm_labels = masked_lm_labels.to(args.device)
			model.train()
			outputs = model(inputs, masked_lm_labels=masked_lm_labels, next_sentence_label=next_sentence_label, bag_of_tokens_label=bag_of_tokens_label)
			loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

			if args.n_gpu > 1:
				loss = loss.mean()  # mean() to average on multi-gpu parallel training
			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps

			if args.fp16:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()

			tr_loss += loss.item()
			if (step + 1) % args.gradient_accumulation_steps == 0:
				if args.fp16:
					torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
				else:
					torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
				optimizer.step()
				scheduler.step()  # Update learning rate schedule
				model.zero_grad()
				global_step += 1

				if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
					# Log metrics
					tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
					tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
					logging_loss = tr_loss

				if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
					checkpoint_prefix = 'checkpoint'
					# Save model checkpoint
					output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
					if not os.path.exists(output_dir):
						os.makedirs(output_dir)
					model_to_save = model.module if hasattr(model,
															'module') else model  # Take care of distributed/parallel training
					model_to_save.save_pretrained(output_dir)
					torch.save(args, os.path.join(output_dir, 'training_args.bin'))
					logger.info("Saving model checkpoint to %s", output_dir)

					_rotate_checkpoints(args, checkpoint_prefix)

			if args.max_steps > 0 and global_step > args.max_steps:
				epoch_iterator.close()
				break
		if args.max_steps > 0 and global_step > args.max_steps:
			train_iterator.close()
			break

	if args.local_rank in [-1, 0]:
		tb_writer.close()

	return global_step, tr_loss / global_step




def main():
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--input_data_dir", default=None, type=str, required=True,
						help="The input training data firectory (*.txt files).")
	parser.add_argument("--output_dir", default=None, type=str, required=True,
						help="The output directory where the model predictions and checkpoints will be written.")

	## Other parameters
	parser.add_argument("--eval_data_file", default=None, type=str,
						help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

	parser.add_argument("--model_type", default="bert", type=str,
						help="The model architecture to be fine-tuned.")
	parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
						help="The model checkpoint for weights initialization.")

	parser.add_argument("--mlm", action='store_true',
						help="Train with masked-language modeling loss instead of language modeling.")
	parser.add_argument("--mlm_probability", type=float, default=0.15,
						help="Ratio of tokens to mask for masked language modeling loss")

	parser.add_argument("--config_name", default="", type=str,
						help="Optional pretrained config name or path if not the same as model_name_or_path")
	parser.add_argument("--tokenizer_name", default="", type=str,
						help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
	parser.add_argument("--cache_dir", default="", type=str,
						help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
	parser.add_argument("--block_size", default=-1, type=int,
						help="Optional input sequence length after tokenization."
							 "The training dataset will be truncated in block of this size for training."
							 "Default to the model max input length for single sentence inputs (take into account special tokens).")
	parser.add_argument("--do_train", action='store_true',
						help="Whether to run training.")
	parser.add_argument("--do_eval", action='store_true',
						help="Whether to run eval on the dev set.")
	parser.add_argument("--evaluate_during_training", action='store_true',
						help="Run evaluation during training at each logging step.")
	parser.add_argument("--do_lower_case", action='store_true',
						help="Set this flag if you are using an uncased model.")

	parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
						help="Batch size per GPU/CPU for training.")
	parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
						help="Batch size per GPU/CPU for evaluation.")
	parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
						help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument("--learning_rate", default=5e-5, type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--weight_decay", default=0.0, type=float,
						help="Weight deay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float,
						help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float,
						help="Max gradient norm.")
	parser.add_argument("--num_train_epochs", default=1.0, type=float,
						help="Total number of training epochs to perform.")
	parser.add_argument("--max_steps", default=-1, type=int,
						help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
	parser.add_argument("--warmup_steps", default=0, type=int,
						help="Linear warmup over warmup_steps.")

	parser.add_argument('--logging_steps', type=int, default=50,
						help="Log every X updates steps.")
	parser.add_argument('--save_steps', type=int, default=50,
						help="Save checkpoint every X updates steps.")
	parser.add_argument('--save_total_limit', type=int, default=None,
						help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
	parser.add_argument("--eval_all_checkpoints", action='store_true',
						help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
	parser.add_argument("--no_cuda", action='store_true',
						help="Avoid using CUDA when available")
	parser.add_argument('--overwrite_output_dir', action='store_true',
						help="Overwrite the content of the output directory")
	parser.add_argument('--overwrite_cache', action='store_true',
						help="Overwrite the cached training and evaluation sets")
	parser.add_argument('--seed', type=int, default=42,
						help="random seed for initialization")

	parser.add_argument('--fp16', action='store_true',
						help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
	parser.add_argument('--fp16_opt_level', type=str, default='O1',
						help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
							 "See details at https://nvidia.github.io/apex/amp.html")
	parser.add_argument("--local_rank", type=int, default=-1,
						help="For distributed training: local_rank")
	parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
	parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
	args = parser.parse_args()

	if os.path.exists(args.output_dir) and os.listdir(
			args.output_dir) and args.do_train and not args.overwrite_output_dir:
		raise ValueError(
			"Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
				args.output_dir))



	# Setup CUDA, GPU & distributed training
	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		args.n_gpu = torch.cuda.device_count()
	args.device = device

	logger.warning("torch.cuda.is_available(): %s, current device:%s", str(torch.cuda.is_available()), args.device)

	# Setup logging
	logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt='%m/%d/%Y %H:%M:%S',
						level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
	logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
				   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

	# Set seed
	set_seed(args)

	# Load pretrained model and tokenizer
	if args.local_rank not in [-1, 0]:
		torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

	config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
	config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
	tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
												do_lower_case=args.do_lower_case)
	if args.block_size <= 0:
		args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
	args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
	model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
										config=config)
	model.to(args.device)

	if args.local_rank == 0:
		torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

	logger.info("Training/evaluation parameters %s", args)

	# Training
	if args.local_rank not in [-1, 0]:
		torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

	train_dataset = load_and_cache_examples(args, tokenizer)

	if args.local_rank == 0:
		torch.distributed.barrier()

	global_step, tr_loss = train(args, train_dataset, model, tokenizer, config)
	logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


	# Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
	if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
		# Create output directory if needed
		if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
			os.makedirs(args.output_dir)

		logger.info("Saving model checkpoint to %s", args.output_dir)
		# Save a trained model, configuration and tokenizer using `save_pretrained()`.
		# They can then be reloaded using `from_pretrained()`
		model_to_save = model.module if hasattr(model,
												'module') else model  # Take care of distributed/parallel training
		model_to_save.save_pretrained(args.output_dir)
		tokenizer.save_pretrained(args.output_dir)

		# Good practice: save your training arguments together with the trained model
		torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

		# Load a trained model and vocabulary that you have fine-tuned
		model = model_class.from_pretrained(args.output_dir)
		tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
		model.to(args.device)

	# Evaluation
	results = {}
	return results


if __name__ == "__main__":
	main()