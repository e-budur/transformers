
import logging
import os
import codecs
import json
import copy

from .utils import DataProcessor, InputExample, InputFeatures
from ...file_utils import is_tf_available
if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


class InputTurnExample(object):
    def __init__(self, utterance, utterance_tokens=None, intent=None, enumerable_entities=[], non_enumerable_entities=[], slot_labels=[]):
        self.utterance = utterance
        self.utterance_tokens = utterance_tokens
        self.intent = intent
        self.enumerable_entities = enumerable_entities
        self.non_enumerable_entities = non_enumerable_entities
        self.slot_labels = slot_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"




class InputTurnFeatures(object):

    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label, enumerable_entity_labels, non_enumerable_entity_labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label = intent_label
        self.enumerable_entity_labels = enumerable_entity_labels
        self.non_enumerable_entity_labels = non_enumerable_entity_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class GoogleSimuatedDialogueProcessor(DataProcessor):


    def __init__(self):
        self.taxonomy = {

            'enumerable_entities': {
                'category': ['french', 'indian', 'taiwanese', 'italian', 'mexican', 'greek', 'mediterranean',
                              'chinese', 'thai', 'vietnamese'],
                'date': [  'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday','today', 'tonight', 'tomorrow', 'next monday'],
                'meal': ['brunch', 'lunch', 'breakfast', 'dinner'],
                'price_range': ['expensive', 'inexpensive', 'moderately priced'],
                'rating': ['michelin rated', 'zagat rated', 'good']
            },
            'non_enumerable_entities': {
                'location': ['madison', 'middletown', 'redmond', 'yorktown heights'],
                'movie': ['doctor who', 'sully', 'the accountant', 'the girl on the train', 'trolls'],
                'num_people': ['4', '5'],
                'num_tickets': ['4', '6'],
                'restaurant_name': ['acorn', 'amarin', 'amber india', 'angel', 'cascal'
                    , 'cetrella', 'cheese cake factory'
                    , 'ephesus', 'evvia', 'il fornaio'
                    , 'los altos grill', 'oren hummus'
                    , 'pompei', 'sakoon', 'sumiko'],
                'theatre_name': ['camera 7', 'century 20 great mall', 'lincoln square cinemas', 'shoreline theater'],
                'time': ['1 pm', '5.30 pm', '6 pm', '6:00 pm', '7.30 pm', '8 pm', '8:00 pm']
            },
            'intents': ['BUY_MOVIE_TICKETS', 'FIND_RESTAURANT', 'RESERVE_RESTAURANT']
        }

    def _read_json(self, file_name):
        with codecs.open(file_name, encoding='utf-8') as json_file:
            conversations = json.load(json_file)
            return conversations



    def convert_to_nlu_data_in_conversation(self, conversation):

        nlu_data_in_conversation = []
        for turn in conversation['turns']:
            if 'user_utterance' not in turn or 'user_intents' not in turn:
                continue
            nlu_data = {}
            nlu_data['tokens'] = turn['user_utterance']['tokens']
            nlu_data['slot_labels'] = ['O' for item in turn['user_utterance']['tokens']]
            nlu_data['utterance'] = turn['user_utterance']['text']

            if 'user_intents' in turn:
                nlu_data['intent'] = turn['user_intents'][0]

            if 'slots' in turn['user_utterance']:
                slots = turn['user_utterance']['slots']
                tokens = turn['user_utterance']['tokens']

                nlu_data['entities'] = {
                    'enumerable': [],
                    'non_enumerable': []
                }

                nlu_data['slots'] = []

                for slot in slots:
                    start = slot['start']
                    exclusive_end = slot['exclusive_end']
                    slot_name = slot['slot']
                    slot_value = (' '.join(tokens[start:exclusive_end]))

                    if slot_name in self.taxonomy['enumerable_entities']:
                        enumerable_entity='='.join([slot_name, slot_value])
                        if slot_name not in self.enumerable_entities:
                            self.enumerable_entities[slot_name] = set()
                        self.enumerable_entities[slot_name].add(slot_value)
                        nlu_data['entities']['enumerable'].append(enumerable_entity)
                    elif slot_name in self.taxonomy['non_enumerable_entities']:
                        nlu_data['entities']['non_enumerable'].append('='.join([slot_name, slot_value]))
                        nlu_data['slot_labels'][start] = 'B-' + slot_name
                        for index in range(start + 1, exclusive_end):
                            nlu_data['slot_labels'][index] = 'I-' + slot_name

            example = InputTurnExample(utterance=nlu_data['utterance'],
                                       utterance_tokens=nlu_data['tokens'],
                                       intent=nlu_data['intent'],
                                       enumerable_entities=nlu_data['entities']['enumerable'],
                                       non_enumerable_entities=nlu_data['entities']['non_enumerable'],
                                       slot_labels=nlu_data['slot_labels']
                                       )
            nlu_data_in_conversation.append(example)

        return nlu_data_in_conversation

    def create_nlu_dataset(self, data_dir, domains, set_type):
        nlu_dataset = []
        self.enumerable_entities = dict()
        for domain_name in domains:
            file_name = os.path.join(data_dir, domain_name, set_type+'.json')
            conversations = self._read_json(file_name)

            for conversation in conversations:
                nlu_data_in_conversation = self.convert_to_nlu_data_in_conversation(conversation)

                nlu_dataset.extend(nlu_data_in_conversation)
        return nlu_dataset

    def _create_examples(self, data_dir, domains=['sim-M', 'sim-R'], set_type='train'):
        nlu_dataset = self.create_nlu_dataset(data_dir, domains, set_type)
        return nlu_dataset

    def get_train_examples(self, data_dir):
        return self._create_examples(data_dir, domains=['sim-M', 'sim-R'], set_type='train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(data_dir, domains=['sim-M', 'sim-R'], set_type='dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(data_dir, domains=['sim-M', 'sim-R'], set_type='test')

    def get_intents_labels(self):
        intents_labels = self.taxonomy['intents']
        return intents_labels

    def get_enumerable_entity_labels(self):
        enumerable_entity_labels = []
        for enumerable_entity_key, enumerable_entity_values in self.taxonomy['enumerable_entities'].items():
            for enumerable_entity_value in enumerable_entity_values:
                enumerable_entity_label = enumerable_entity_key + '=' + enumerable_entity_value
                enumerable_entity_labels.append(enumerable_entity_label)
        return enumerable_entity_labels

    def get_non_enumerable_entity_labels(self):
        non_enumerable_entity_labels = []
        non_enumerable_entity_labels.append('O')
        for non_enumerable_entity_key in self.taxonomy['non_enumerable_entities'].keys():
            non_enumerable_entity_labels.append('B-' + non_enumerable_entity_key)
            non_enumerable_entity_labels.append('I-' + non_enumerable_entity_key)

        return non_enumerable_entity_labels

    def get_labels(self):
        """Gets the list of labels for this data set."""
        intent_labels = self.get_intents_labels()

        enumerable_entity_labels = self.get_enumerable_entity_labels()

        non_enumerable_entity_labels = self.get_non_enumerable_entity_labels()

        return intent_labels, enumerable_entity_labels, non_enumerable_entity_labels

    def decode_intent_preds(self, intent_preds):
        intents_labels = self.get_intents_labels()
        intent_preds_decoded = intents_labels[intent_preds]
        return intent_preds_decoded

    def decode_enumerable_enitity_preds(self, enumerable_entity_preds):
        enumerable_entity_labels = self.get_enumerable_entity_labels()
        enumerable_entity_preds_decoded = enumerable_entity_labels[enumerable_entity_preds]
        return enumerable_entity_preds_decoded

    def decode_non_enumerable_enitity_preds(self, non_enumerable_enitity_preds):
        non_enumerable_entity_labels = self.get_non_enumerable_entity_labels()
        non_enumerable_enitity_preds_decoded = non_enumerable_entity_labels[non_enumerable_enitity_preds]
        return non_enumerable_enitity_preds_decoded

def conversational_datasets_convert_examples_to_features(examples, tokenizer,
                                      max_length=512,
                                      task=None,
                                      list_of_intent_labels=None,
                                      list_of_enumerable_entity_labels=None,
                                      list_of_non_enumerable_entity_labels=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True,
                                      sep_token_extra=False,
                                      sequence_a_segment_id=0,
                                      cls_token_segment_id=0,
                                      cls_token_at_end=False):

    logger.info("Using intent_labels %s for task %s" % (list_of_intent_labels, task))
    logger.info("Using enumerable_entity_labels %s for task %s" % (list_of_enumerable_entity_labels, task))
    logger.info("Using non_enumerable_entity_labels %s for task %s" % (list_of_non_enumerable_entity_labels, task))

    intent_labels_map = {label: i for i, label in enumerate(list_of_intent_labels)}
    enumerable_entity_labels_map = {label: i for i, label in enumerate(list_of_enumerable_entity_labels)}
    non_enumerable_entity_labels_map = {label: i for i, label in enumerate(list_of_non_enumerable_entity_labels)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        tokens = []
        label_ids = []
        for word, slot_label in zip(example.utterance_tokens, example.slot_labels):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([non_enumerable_entity_labels_map[slot_label]] + [tokenizer.pad_token_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_length - special_tokens_count:
            tokens = tokens[: (max_length - special_tokens_count)]
            label_ids = label_ids[: (max_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [tokenizer.sep_token]
        label_ids += [tokenizer.pad_token_id]


        tokens = [tokenizer.cls_token] + tokens
        label_ids = [tokenizer.pad_token_id] + label_ids
        token_type_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            label_ids = [0] * padding_length + label_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            label_ids =  label_ids + [0] * padding_length

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        intent_label = intent_labels_map[example.intent]
        enumerable_entity_labels = [0]*len(list_of_enumerable_entity_labels)
        for entity in example.enumerable_entities:
            entity_index = enumerable_entity_labels_map[entity]
            enumerable_entity_labels[entity_index] = 1

        non_enumerable_entity_labels = label_ids
        pad_length = max_length - len(non_enumerable_entity_labels)
        non_enumerable_entity_labels = non_enumerable_entity_labels + [0]*pad_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in example.utterance_tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_tokens: %s" % " ".join([str(tokenizer._convert_id_to_token(x)) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("intent_label: %s (id = %d)" % (example.intent, intent_label))
            logger.info("enumerable_labels: %s (ids = %s)" % (example.enumerable_entities, ','.join(str(x) for x in enumerable_entity_labels)))
            logger.info("non_enumerable_entity_labels: %s (ids = %s)" % (example.non_enumerable_entities, ','.join(str(x) for x in non_enumerable_entity_labels)))
            logger.info("non_enumerable_entity_labels_decoded: %s (ids = %s)" % (
            example.non_enumerable_entities, ','.join(list_of_non_enumerable_entity_labels[x] for x in non_enumerable_entity_labels)))

        features.append(InputTurnFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              intent_label=intent_label,
                              enumerable_entity_labels=enumerable_entity_labels,
                              non_enumerable_entity_labels=non_enumerable_entity_labels,
                              ))

    return features

conversational_datasets_processors = {
    "google-simulated-dialogue": GoogleSimuatedDialogueProcessor
}

conversational_datasets_output_modes = {
    "google-simulated-dialogue": "classification"
}