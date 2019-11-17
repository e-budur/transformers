
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


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):

        outputs = super(BertNLUModel, self).forward(input_ids, attention_mask, token_type_ids, position_ids,
                head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask)

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


class BertForJointUnderstanding(BertForPreTraining):
    def __init__(self, config):
        super(BertForJointUnderstanding, self).__init__(config)
        self.num_intent_labels = config.num_intent_labels
        self.num_enumerable_entity_labels = config.num_enumerable_entity_labels
        self.num_non_enumerable_entity_labels = config.num_non_enumerable_entity_labels

        self.bert = self.get_bert_model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_intents = nn.Linear(config.hidden_size, config.num_intent_labels)
        self.classifier_enumerable_entities = nn.Linear(config.hidden_size, config.num_enumerable_entity_labels)
        self.classifier_non_enumerable_entities = nn.Linear(config.hidden_size, config.num_non_enumerable_entity_labels)

        self.init_weights()

    def get_bert_model(self, config):
        return BertModel(config)

    def get_sequence_output(self, outputs):
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        return sequence_output

    def get_pooled_outputs(self, outputs):
        common_pooled_outputs = outputs[1]
        common_pooled_outputs = self.dropout(common_pooled_outputs)
        return common_pooled_outputs, common_pooled_outputs

    def get_intent_logits(self, pooled_output_for_cls):
        intent_logits = self.classifier_intents(pooled_output_for_cls)
        return intent_logits

    def get_enumerable_entity_logits(self, pooled_output_for_cls):
        enumerable_entity_logits = self.classifier_enumerable_entities(pooled_output_for_cls)
        return enumerable_entity_logits

    def get_non_enumerable_entity_logits(self, sequence_output):
        non_enumerable_entity_logits = self.classifier_non_enumerable_entities(sequence_output)
        return non_enumerable_entity_logits

    def get_hidden_states(self, outputs):
        hidden_states = outputs[2:]
        return hidden_states

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, intent_labels=None, enumerable_entity_labels=None, non_enumerable_entity_labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = self.get_sequence_output(outputs)

        pooled_output_for_intent, pooled_output_for_enumerable_entities  = self.get_pooled_outputs(outputs)

        intent_logits = self.get_intent_logits(pooled_output_for_intent)
        enumerable_entity_logits = self.get_enumerable_entity_logits(pooled_output_for_enumerable_entities)
        non_enumerable_entity_logits = self.get_non_enumerable_entity_logits(sequence_output)
        hidden_states = self.get_hidden_states(outputs)

        outputs = (intent_logits, enumerable_entity_logits, non_enumerable_entity_logits,) + hidden_states  # add hidden states and attention if they are here

        if intent_labels is not None and enumerable_entity_labels is not None and non_enumerable_entity_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_multi_label = MultiLabelSoftMarginLoss()
            intent_loss = loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_labels.view(-1))
            enumerable_entity_loss = loss_multi_label(enumerable_entity_logits.view(-1, self.num_enumerable_entity_labels), enumerable_entity_labels)
            non_enumerable_entity_loss = loss_fct(non_enumerable_entity_logits.view(-1, self.num_non_enumerable_entity_labels), non_enumerable_entity_labels.view(-1))
            loss = intent_loss + enumerable_entity_loss + non_enumerable_entity_loss
            outputs = (loss,) + outputs

        return outputs  # (loss), (intent_logits, enumerable_entity_logits), (hidden_states), (attentions)


class BertNLUForJointUnderstanding(BertForJointUnderstanding):
    def __init__(self, config):
        super(BertNLUForJointUnderstanding, self).__init__(config)

    def get_bert_model(self, config):
        return BertNLUModel(config)

    def get_pooled_outputs(self, outputs):
        pooled_output_for_intent = outputs[1]
        pooled_output_for_intent = self.dropout(pooled_output_for_intent)

        pooled_output_for_enumerable_entities = outputs[2]
        pooled_output_for_enumerable_entities = self.dropout(pooled_output_for_enumerable_entities)

        return pooled_output_for_intent, pooled_output_for_enumerable_entities

    def get_hidden_states(self, outputs):
        hidden_states = outputs[3:]
        return hidden_states