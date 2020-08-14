from unicode_tr import unicode_tr
import string
import six
import collections
import codecs

def align_cases(input_word_form, parsed_word_form):

    input_word_form = unicode_tr(input_word_form)
    parsed_word_form = unicode_tr(parsed_word_form)

    i = 0
    j = 0
    resulting_word_characters = []
    subword_unit = []
    while i < len(input_word_form) and j < len(parsed_word_form):
        if input_word_form[i] == parsed_word_form[j] or input_word_form[i] == unicode_tr(parsed_word_form[j]).upper():
            subword_unit.append(input_word_form[i])
            i += 1
            j += 1
            continue

        if input_word_form[i] in string.punctuation:
            i += 1

        if parsed_word_form[j] == ' ':
            j += 1 #skip delimiter in parsed word
            resulting_word_characters.append(''.join(subword_unit))
            subword_unit = []
            continue

        #token with a non-conventional pattern will reach here. Will skip aligning their cases.
        return parsed_word_form

    if len(subword_unit)>0:
        resulting_word_characters.append(''.join(subword_unit))

    resulting_word = ' '.join(resulting_word_characters)
    return resulting_word

#####################################################################
# adopted from the original BERT implementatin from Google Research
# https://github.com/google-research/bert
#####################################################################
def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")

#####################################################################
# adapted from the original BERT implementatin from Google Research
# https://github.com/google-research/bert
#####################################################################
def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0
  with codecs.open(vocab_file, mode="r", encoding='utf-8') as reader:
      while True:
          token = convert_to_unicode(reader.readline())
          if not token:
              break
          token = token.strip().split('\t')[0]
          vocab[token] = index
          index += 1

  return vocab

#####################################################################
# adopted from the original BERT implementatin from Google Research
# https://github.com/google-research/bert
#####################################################################
def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
    output.append(vocab[item])
  return output

#####################################################################
# adopted from the original BERT implementatin from Google Research
# https://github.com/google-research/bert
#####################################################################
def convert_tokens_to_ids(vocab, tokens):
  return convert_by_vocab(vocab, tokens)

#####################################################################
# adopted from the original BERT implementatin from Google Research
# https://github.com/google-research/bert
#####################################################################
def convert_ids_to_tokens(inv_vocab, ids):
  return convert_by_vocab(inv_vocab, ids)