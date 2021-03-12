# coding=utf-8

from __future__ import absolute_import, division, print_function
from transformers import AutoConfig, AutoModel, AutoTokenizer
import logging

import run_squad

from transformers.modeling_bert import BertForQuestionAnswering

logger = logging.getLogger(__name__)


if __name__ == "__main__":
   run_squad.MODEL_CLASSES['bert-auto'] = (AutoConfig, BertForQuestionAnswering, AutoTokenizer)

   run_squad.main()
