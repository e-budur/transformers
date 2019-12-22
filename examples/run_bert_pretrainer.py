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
from transformers.featurizer_bert_nlu import BertNLUFeaturizer

from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_bert import BertForPreTraining
from transformers.configuration_bert import BertConfig
from transformers.featurizer_bert import BertFeaturizer

from torch.utils.data import DataLoader, Dataset

from transformers import (WEIGHTS_NAME, BertConfig)

from transformers import AdamW, WarmupLinearSchedule
from compress_pickle import dump, load

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

logger = logging.getLogger(__name__)



MODEL_CLASSES = {
	'bert': (BertConfig, BertForPreTraining, BertTokenizer, BertFeaturizer),
	'bert-nlu': (BertNLUConfig, BertNLUForPreTraining, BertNLUTokenizer, BertNLUFeaturizer)
}


class TextDataset(Dataset):

	def __init__(self, tokenizer, input_data_dir='train', cache_folder_suffix='cached_features'):

		self.total_num_examples = 81198328
		self.recalculate_total_num_examples = True
		cached_features_dir = input_data_dir + cache_folder_suffix

		if not os.path.exists(cached_features_dir):
			os.makedirs(cached_features_dir)

		logger.info("Creating features from data directory at %s", input_data_dir)

		files = glob.glob(input_data_dir + "/*txt")
		self.cached_file_paths = []
		for input_file in tqdm(files, desc="read files"):
			examples = []
			input_file_rel_path = os.path.relpath(input_file, input_data_dir)
			cached_file_path = os.path.join(cached_features_dir, input_file_rel_path + u'.pkl.gz')
			self.cached_file_paths.append(cached_file_path)
			if os.path.exists(cached_file_path):
				logger.info("File already processed and cached %s", input_file)
				if self.recalculate_total_num_examples:
					with open(input_file, 'r', encoding='utf-8') as fin:
						num_lines = sum(1 for line in fin)
					self.total_num_examples += num_lines
				continue

			logger.info("Processing file %s", input_file)

			with open(input_file, 'r', encoding='utf-8') as fin:

				prev_sent = None
				for cur_line in fin:
					if len(cur_line.strip())==0:
						continue
					cur_sentences = cur_line.split('.')
					for cur_sent in cur_sentences:
						tokens = tokenizer.tokenize(cur_sent.strip())
						tokens = tokens[:tokenizer.max_len_sentences_pair]
						cur_sent = tokenizer.convert_tokens_to_ids(tokens)
						if len(cur_sent)==0:
							continue

						if prev_sent is None:
							prev_sent = cur_sent
							continue

						prev_sent_padded, cur_sent_padded = self._get_padded_pairs(prev_sent, cur_sent, tokenizer.max_len_sentences_pair)
						example = tokenizer.build_inputs_with_special_tokens(prev_sent_padded, cur_sent_padded)
						examples.append(example)
						prev_sent = cur_sent

			if self.recalculate_total_num_examples:
				self.total_num_examples += len(examples)

			logger.info("Saving features into cached file %s", cached_file_path)

			dump(examples, cached_file_path)# pickling with a gzip compression

	def _get_padded_sent(self, sentence, single_sentence_max_len):
		if len(sentence) > single_sentence_max_len:
			sentence = sentence[:single_sentence_max_len]
		elif len(sentence) < single_sentence_max_len:
			N = single_sentence_max_len - len(sentence)
			sentence = np.pad(sentence, (0, N), 'constant').tolist()
		return sentence

	def _get_padded_pairs(self, prev_sent, cur_sent, max_len_sentences_pair):
		prev_sent_len = int(max_len_sentences_pair/2)
		cur_sent_len = max_len_sentences_pair - prev_sent_len

		prev_sent = self._get_padded_sent(prev_sent, prev_sent_len)
		cur_sent = self._get_padded_sent(cur_sent, cur_sent_len)

		return prev_sent, cur_sent

	def __len__(self):
		return len(self.cached_file_paths)

	def __getitem__(self, index):

		try:
			logger.info("\n{} - Reading file : {}\n".format(index, self.cached_file_paths[index]))
			examples = load(self.cached_file_paths[index])
		except Exception as error:
			logger.info('Failed to read file for some reasons: '+ str(error))
			logger.info('The file was skipped')
			examples = []

		return torch.tensor(examples)


def load_and_cache_examples(args, tokenizer):
	dataset = TextDataset(tokenizer, input_data_dir=args.input_data_dir, cache_folder_suffix='-CACHED-'+args.model_type)
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

def train(args, train_dataset, model, tokenizer, featurizer, config):
	""" Train the model """
	if args.local_rank in [-1, 0]:
		tb_writer = SummaryWriter()

	args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
	train_data_fileloader = DataLoader(train_dataset, sampler=train_sampler) # load 1 file at a time

	if args.max_steps > 0:
		t_total = args.max_steps
		args.num_train_epochs = args.max_steps // (train_dataset.total_num_examples // args.gradient_accumulation_steps) + 1
	else:
		t_total = train_dataset.total_num_examples // args.gradient_accumulation_steps * args.num_train_epochs

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
		file_iterator = tqdm(train_data_fileloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
		for step, file_data in enumerate(file_iterator):

			file_data = file_data.squeeze()

			example_sampler = RandomSampler(file_data) if args.local_rank == -1 else DistributedSampler(file_data)
			example_loader = DataLoader(file_data, sampler=example_sampler,
											   	   batch_size=args.train_batch_size)

			total_example_count = len(file_data)
			total_num_steps = int(total_example_count/args.train_batch_size)
			example_iterator = tqdm(example_loader,
									desc="Rank:"+str(args.local_rank) + " > Examples" ,
									maxinterval=60*60,
									miniters=int(total_num_steps / 10.0))

			step = -1
			for batch in example_iterator:
				step += 1
				inputs, outputs = featurizer.featurize(batch, tokenizer, args, config)
				inputs = inputs.to(args.device)
				outputs = [output.to(args.device) if output is not None else None for output in outputs]
				model.train()
				outputs = model(inputs, *outputs)
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
					file_iterator.close()
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
	parser.add_argument("--learning_rate", default=1e-4, type=float,
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
	parser.add_argument("--warmup_steps", default=10000, type=int,
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
	else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		torch.distributed.init_process_group(backend='nccl')
		args.n_gpu = 1
	args.device = device

	logger.warning("torch.cuda.is_available(): %s, current device:%s", str(torch.cuda.is_available()), args.device)

	# Setup logging
	logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt='%m/%d/%Y %H:%M:%S',
						level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

	logger.setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
	logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
				   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

	# Set seed
	set_seed(args)

	# Load pretrained model and tokenizer
	if args.local_rank not in [-1, 0]:
		torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

	config_class, model_class, tokenizer_class, featurizer_class = MODEL_CLASSES[args.model_type]
	config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)

	tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
												do_lower_case=args.do_lower_case)

	config.vocab_size = tokenizer.vocab_size
	featurizer = featurizer_class()
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

	global_step, tr_loss = train(args, train_dataset, model, tokenizer, featurizer, config)
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
