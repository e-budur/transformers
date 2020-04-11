from pathlib import Path
import argparse
import os
import codecs
from jpype import *
import random
from unicode_tr import unicode_tr
import string
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


def parse_morphologically(sentence, params):
    if sentence.strip() == u'':
       return sentence

    if params['lower_case']:
        sentence = sentence.lower() # this line should be improved by using the lower method of unicode_tr library

    sentence_analysis = params['morphology'].analyzeAndDisambiguate(JString(sentence))
    parsed_words = []
    for sentence_word_analysis in sentence_analysis:
      word_analysis = sentence_word_analysis.getWordAnalysis()
      word_analysis_results = word_analysis.getAnalysisResults()
      input_surface_word = str(word_analysis.getInput())
      if len(word_analysis_results) > 0:
        single_anaysis = word_analysis_results[0]
        parsed_word = str(params['AnalysisFormatters'].SURFACE_SEQUENCE.format(single_anaysis))
        parsed_word = align_cases(input_surface_word, parsed_word)
        if params['omit_suffixes']:
            parsed_word = parsed_word.split(' ')[0]

        parsed_word = input_surface_word[0] + parsed_word[1:] # try to make sure at least the title case structure of the input word is preserved
      else:
        parsed_word = str(word_analysis.getInput())
      parsed_words.append(parsed_word)
    parsed_sentence = ' '.join(parsed_words)
    return parsed_sentence

def parse_basically(sentence, params):
    if sentence.strip() == u'':
       return sentence

    if params['lower_case']:
        sentence = sentence.lower() # this line should be improved by using the lower method of unicode_tr library

    return sentence

def from_token_to_character_ngram_sequence(token, params):
    # Adapted from the Github Page of the course CS224U
    # https://github.com/e-budur/cs224u/blob/master/vsm.py
    # Adapted from the function get_character_ngrams(w, n)
    """Map a sentence to its character-level n-grams.
    Parameters
    ----------
    w : str
    n : int
        The n-gram size.
    Returns
    -------
    list of str
    """
    n = params['ngram_size']
    token = list(token)
    return ' '.join(["".join(token[i: i+n]) for i in range(len(token)-n+1)]).strip()

def from_sentence_to_character_ngram_sequence(sentence, params):
    # Adapted from the Github Page of the course CS224U
    # https://github.com/e-budur/cs224u/blob/master/vsm.py
    # Adapted from the function get_character_ngrams(w, n)
    """Map a sentence to its character-level n-grams.
    Parameters
    ----------
    w : str
    n : int
        The n-gram size.
    Returns
    -------
    list of str
    """
    ngrams_of_tokens = []
    if params['lower_case']:
        sentence = sentence.lower() # this line should be improved by using the lower method of unicode_tr library

    for token in sentence.split(' '):
        ngram_of_token = from_token_to_character_ngram_sequence(token, params)
        ngrams_of_tokens.append(ngram_of_token)

    return ' '.join(ngrams_of_tokens).strip()

def get_preprocess_parameters(args):
    if args.do_morphological_preprocessing:
        if isJVMStarted() == False:
            turn_on_morphological_analyzer(args)
        params = {
            'morphology': JClass('zemberek.morphology.TurkishMorphology').createWithDefaults(),
            'AnalysisFormatters': JClass('zemberek.morphology.analysis.AnalysisFormatters'),
            'omit_suffixes':args.omit_suffixes_after_morphological_preprocessing,
            'preprocess_func':  parse_morphologically,
            'lower_case':args.do_lower_case
        }
    elif args.do_ngram_preprocessing:
        params = {
            'ngram_size':args.ngram_size,
            'preprocess_func':from_sentence_to_character_ngram_sequence,
            'lower_case': args.do_lower_case
        }
    else:
        params = {
            'preprocess_func':parse_basically,
            'lower_case': args.do_lower_case
        }

    return params

def preprocess_file(input_file_path, output_file_path, args):
    print('-----------------------------------')
    print('Preprocessing new file')
    print('Input file:', input_file_path)
    print('Output file:', output_file_path)
    print('do_morphological_preprocessing:', args.do_morphological_preprocessing)
    print('omit_suffixes_after_morphological_preprocessing:', args.omit_suffixes_after_morphological_preprocessing)
    print('do_ngram_preprocessing:', args.do_ngram_preprocessing)
    print('-----------------------------------')

    params = get_preprocess_parameters(args)
    if params is None:
        return

    with codecs.open(input_file_path, mode='r', encoding='utf-8') as input_file:
      with codecs.open(output_file_path, mode='w', encoding='utf-8') as output_file:
        for input_line in input_file:
          input_line = input_line.strip()
          output_line = params['preprocess_func'](input_line, params)
          output_file.write(output_line+'\n')

          if random.random() < 0.01:  # print some examples of shuffles sentences
              print(u"\n{}\nOriginal line: {}\nProcessed line: {}\n{}\n ".format(
                    u"================================= PROCESSED EXAMPLE ===================================",
                    input_line.strip(),
                    output_line.strip(),
                    u"=======================================================================================")
              )

def preprocess_examples(examples, args):
    print('-----------------------------------')
    print('Preprocessing new file')
    print('do_morphological_preprocessing:', args.do_morphological_preprocessing)
    print('zemberek_path:', args.zemberek_path)
    print('omit_suffixes_after_morphological_preprocessing:', args.omit_suffixes_after_morphological_preprocessing)
    print('do_ngram_preprocessing:', args.do_ngram_preprocessing)
    print('ngram_size:', args.ngram_size)
    print('do_lower_case:', args.do_lower_case)
    print('-----------------------------------')

    params = get_preprocess_parameters(args)
    if params is None:
        return

    for example in examples:
        processed_text_a = params['preprocess_func'](example.text_a, params)
        processed_text_b = params['preprocess_func'](example.text_b, params)
        if random.random() < 0.01:  # print some examples of shuffles sentences
            print(u"\n{}\nOriginal line (text a): {}\nProcessed line (text a): {}\nOriginal line (text b): {}\nProcessed line (text b):{}\n{}\n ".format(
                u"================================= PROCESSED EXAMPLE ===================================",
                example.text_a.strip(),
                processed_text_a.strip(),
                example.text_b.strip(),
                processed_text_b.strip(),
                u"=======================================================================================")
            )
        example.text_a = processed_text_a
        example.text_b = processed_text_b
def turn_on_morphological_analyzer(args):
    if args.java_home_path is not None:
        os.environ['JAVA_HOME_PATH'] = args.java_home_path

    cpath = f'-Djava.class.path=%s' % (args.zemberek_path)

    startJVM(
        getDefaultJVMPath(),
        '-ea',
        cpath,
        convertStrings=False
    )

def turn_off_morphological_analyzer():
    shutdownJVM()

def run_morphological_preprocessor(args):
    turn_on_morphological_analyzer(args)
    input_file_path = Path(args.input_file_path)
    output_file_path = Path(args.output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    preprocess_file(input_file_path, output_file_path, args)

    turn_off_morphological_analyzer()