
from .file_utils import add_start_docstrings

from transformers import *
from transformers import modeling_bert as original_bert
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, MultiLabelSoftMarginLoss


class BertMultiLabelPooler(nn.Module):
    def __init__(self, config):
        super(BertMultiLabelPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Sigmoid()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the second token.
        second_token_tensor = hidden_states[:, 1]
        pooled_output = self.dense(second_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class BertNLUModel(BertModel):

    def __init__(self, config):
        super(BertNLUModel, self).__init__(config)
        self.multi_label_pooler = BertMultiLabelPooler(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None):
        outputs = super(BertNLUModel, self).forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask)
        sequence_output, multi_class_pooled_output = outputs[0], outputs[1]
        hidden_states_and_attentions = outputs[2:]

        multi_label_pooled_output = self.multi_label_pooler(sequence_output)
        outputs = (sequence_output, multi_class_pooled_output, multi_label_pooled_output) + hidden_states_and_attentions
        return outputs  # sequence_output, multi_class_pooled_output, multi_label_pooled_output, (hidden_states), (attentions)


class BertOnlyMultiLabelHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMultiLabelHead, self).__init__()
        self.bag_of_tokens = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, multi_label_pooled_output):
        bag_of_tokens_scores = self.bag_of_tokens(multi_label_pooled_output)
        return bag_of_tokens_scores

class BertNLUPreTrainingHeads(original_bert.BertPreTrainingHeads):
   def __init__(self, config):
       super(BertNLUPreTrainingHeads, self).__init__(config)
       self.bert_pretraining_heads = original_bert.BertPreTrainingHeads(config)
       self.multi_label_prediction = BertOnlyMultiLabelHead(config)

   def forward(self, sequence_output, multi_class_pooled_output, multi_label_pooled_output):
       language_model_prediction_scores, seq_relationship_score = self.bert_pretraining_heads(sequence_output, multi_class_pooled_output)
       multi_label_prediction_scores = self.multi_label_prediction(multi_label_pooled_output)
       return language_model_prediction_scores, seq_relationship_score, multi_label_prediction_scores


class BertNLUForPreTraining(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.
        **multi_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the existence of particular characters prediction (multi-label classification) loss (bag of chars).
            Indices should be in ``[0, 1]``.
            ``0`` in i'th index of the output indicates the i'th character in the alphabet DOES NOT exist in any of the string A and B,
            ``1`` in i'th index of the output indicates the i'th character in the alphabet DOES exist in any of the string A and B.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when both ``masked_lm_labels`` and ``next_sentence_label`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, 1)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **multi_label_prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, multi_label_output_size)``
            Prediction scores of the existence of i'th letter in the alphabet head (bag of chars).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertNLUTokenizer.from_pretrained('bert-base-uncased')
        model = BertNLUForPreTraining.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        prediction_scores, seq_relationship_scores, multi_label_prediction_scores = outputs[:3]

    """
    def __init__(self, config):
        super(BertNLUForPreTraining, self).__init__(config)

        self.bert_nlu = BertNLUModel(config)
        self.cls = BertNLUPreTrainingHeads(config)
        self.config = config
        self.init_weights()
        self.tie_weights()
        self.vocab_size = config.vocab_size

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert_nlu.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                masked_lm_labels=None, next_sentence_label=None, bag_of_tokens_label=None):

        outputs = self.bert_nlu(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output, multi_class_pooled_output, multi_label_pooled_output  = outputs[:3]
        language_model_prediction_scores, seq_relationship_score, multi_label_prediction_scores = self.cls(sequence_output, multi_class_pooled_output, multi_label_pooled_output)

        outputs = (language_model_prediction_scores, seq_relationship_score, multi_label_prediction_scores,) + outputs[3:]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and next_sentence_label is not None and bag_of_tokens_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss_multi_label = MultiLabelSoftMarginLoss()
            masked_lm_loss = loss_fct(language_model_prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            bag_of_tokens_loss = loss_multi_label(multi_label_prediction_scores.view(-1, self.vocab_size), bag_of_tokens_label.view(-1, self.vocab_size))
            total_loss = masked_lm_loss + next_sentence_loss + bag_of_tokens_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, multi_label_prediction_scores, (hidden_states), (attentions)



class BertNLUForJointUnderstanding(BertNLUForPreTraining):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(BertNLUForJointUnderstanding, self).__init__(config)
        self.num_intents = config.num_intents
        self.num_enumerable_entities = config.num_enumerable_entities

        self.bert_nlu = BertNLUModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_intents = nn.Linear(config.hidden_size, self.config.num_intents)
        self.classifier_enumerable_entities = nn.Linear(config.hidden_size, self.config.num_enumerable_entities)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, intent_labels=None, enumerable_entity_labels=None):

        outputs = self.bert_nlu(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output_for_intent = outputs[1]
        pooled_output_for_enumerable_entities = outputs[2]

        pooled_output_for_intent = self.dropout(pooled_output_for_intent)
        intent_logits = self.classifier_intents(pooled_output_for_intent)

        pooled_output_for_enumerable_entities = self.dropout(pooled_output_for_enumerable_entities)
        enumerable_entity_logits = self.classifier_enumerable_entities(pooled_output_for_enumerable_entities)

        outputs = (intent_logits, enumerable_entity_logits) + outputs[3:]  # add hidden states and attention if they are here

        if intent_labels is not None and enumerable_entity_labels is not None:
            loss_fct = CrossEntropyLoss()
            intent_loss = loss_fct(intent_logits.view(-1, self.num_intents), intent_labels.view(-1))
            enumerable_entity_loss = loss_fct(enumerable_entity_logits.view(-1, self.num_enumerable_entities), enumerable_entity_labels.view(-1))
            loss = intent_loss + enumerable_entity_loss
            outputs = (loss) + outputs

        return outputs  # (loss), (intent_logits, enumerable_entity_logits), (hidden_states), (attentions)