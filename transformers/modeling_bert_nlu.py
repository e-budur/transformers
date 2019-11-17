
from .file_utils import add_start_docstrings

from transformers import *
from transformers import modeling_bert as original_bert
from transformers.modeling_bert import BertEmbeddings, BertPooler, BertEncoder
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, MultiLabelSoftMarginLoss, BCEWithLogitsLoss


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



class BertNLUModelTest(BertModel):

    def __init__(self, config):
        super(BertNLUModelTest, self).__init__(config)
        self.multi_label_pooler = BertMultiLabelPooler(config)
        self.init_weights()


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):

        outputs = super(BertNLUModelTest, self).forward(input_ids, attention_mask, token_type_ids, position_ids,
                head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask)

        sequence_output, multi_class_pooled_output = outputs[0], outputs[1]
        hidden_states_and_attentions = outputs[2:]

        multi_label_pooled_output = self.multi_label_pooler(sequence_output)
        outputs = (sequence_output, multi_class_pooled_output, multi_label_pooled_output) + hidden_states_and_attentions
        return outputs  # sequence_output, multi_class_pooled_output, multi_label_pooled_output, (hidden_states), (attentions)



class BertNLUModelTest(BertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config):
        super(BertNLUModelTest, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.multi_label_pooler = BertMultiLabelPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
        """ Forward pass on the Model.

        The model can behave as an encoder (with only self-attention) as well
        as a decoder, in which case a layer of cross-attention is added between
        the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
        Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

        To behave as an decoder the model needs to be initialized with the
        `is_decoder` argument of the configuration set to `True`; an
        `encoder_hidden_states` is expected as an input to the forward pass.

        .. _`Attention is all you need`:
            https://arxiv.org/abs/1706.03762

        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        multi_label_pooled_output = self.multi_label_pooler(sequence_output)

        outputs = (sequence_output, pooled_output, multi_label_pooled_output) + encoder_outputs[2:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

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
            loss_multi_label = BCEWithLogitsLoss()
            intent_loss = loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_labels.view(-1))
            enumerable_entity_loss = loss_multi_label(enumerable_entity_logits.view(-1, self.num_enumerable_entity_labels), enumerable_entity_labels)
            non_enumerable_entity_loss = loss_fct(non_enumerable_entity_logits.view(-1, self.num_non_enumerable_entity_labels), non_enumerable_entity_labels.view(-1))
            loss = intent_loss + enumerable_entity_loss + non_enumerable_entity_loss
            outputs = (loss,) + outputs

        return outputs  # (loss), (intent_logits, enumerable_entity_logits), (hidden_states), (attentions)


class BertNLUForJointUnderstanding(BertForJointUnderstanding):
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
        super(BertNLUForJointUnderstanding, self).__init__(config)

    def get_bert_model(self, config):
        return BertNLUModelTest(config)

    def get_pooled_outputs(self, outputs):
        pooled_output_for_intent = outputs[1]
        pooled_output_for_intent = self.dropout(pooled_output_for_intent)

        pooled_output_for_enumerable_entities = outputs[2]
        pooled_output_for_enumerable_entities = self.dropout(pooled_output_for_enumerable_entities)

        return pooled_output_for_intent, pooled_output_for_enumerable_entities

    def get_hidden_states(self, outputs):
        hidden_states = outputs[3:]
        return hidden_states