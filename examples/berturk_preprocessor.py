from pathlib import Path
import argparse
import os
import codecs
from jpype import *
import random
from unicode_tr import unicode_tr
import string
from turkish_morphology import analysis_pb2, analyze, decompose
import sentencepiece as spm
from subprocess import call

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


def parse_morphologically_zemberek(sentence, params):
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

parse_morphologically_google_research_cache = {}
total_num_words = 0
cache_hit = 0

def parse_morphologically_google_research(sentence, params):
    global total_num_words
    global cache_hit
    #omit suffixes by default. need more implementation to append suffixes when needed
    if sentence.strip() == u'':
       return sentence

    if params['lower_case']:
        sentence = sentence.lower() # this line should be improved by using the lower method of unicode_tr library

    input_surface_words = sentence.split()

    parsed_words = []

    for input_surface_word in input_surface_words:
        total_num_words += 1
        if input_surface_word in parse_morphologically_google_research_cache:
            parsed_word = parse_morphologically_google_research_cache[input_surface_word]
            parsed_words.append(parsed_word)
            cache_hit += 1
            continue

        word_analysis_results = analyze.surface_form(input_surface_word,
                                                     analyzer=params['analyzer'],
                                                     symbol_table=params['symbol_table'])
        if len(word_analysis_results) > 0:
            human_readables = word_analysis_results[0]
            formatted_analysis = decompose.human_readable_analysis(human_readables)
            parsed_word = formatted_analysis.ig[0].root.morpheme
            #parsed_word = single_analysis.split('[')[0][1:]

            parsed_word = align_cases(input_surface_word, parsed_word)
            parsed_word = input_surface_word[0] + parsed_word[
                                                  1:]  # try to make sure at least the title case structure of the input word is preserved
        else:
            parsed_word = input_surface_word

        parse_morphologically_google_research_cache[input_surface_word] = parsed_word
        parsed_words.append(parsed_word)

    parsed_sentence = ' '.join(parsed_words)
    return parsed_sentence

def dump_examples_to_file(examples, examples_file_path):
    num_lines = 0
    with codecs.open(examples_file_path, mode='w', encoding='utf-8') as examples_file:
        for example in examples:
            num_lines += 1
            if num_lines % 10000 == 0:
                print('dump_examples_to_file num_lines processed', num_lines)
            examples_file.write(u'{}\n'.format(example.text_a))
            examples_file.write(u'{}\n'.format(example.text_b))

def load_examples_from_file(examples, examples_file_path):
    line_index = 0
    with codecs.open(examples_file_path, mode='r', encoding='utf-8') as examples_file:
        for line in examples_file:
            if line_index %2 == 0:
                examples[line_index].text_a = line
            else:
                examples[line_index].text_b = line

def parse_morphologically_boun_from_file(input_file_path, output_file_path, boun_parser_dir):
    executed_file_name = 'parse_corpus.py'
    arguments = ["python", os.path.join(boun_parser_dir, executed_file_name), input_file_path, output_file_path]
    print('calling {}'.format(' '.join(arguments)))
    call(arguments, cwd=os.path.join(boun_parser_dir, 'MP'))
    print('{} was completed'.format(executed_file_name))

def disambiguate_morphologically_boun_from_file(input_file_path, output_file_path, boun_parser_dir):
    executed_file_name = 'md.pl'
    arguments = ["perl", os.path.join(boun_parser_dir, executed_file_name), "-disamb", "model.txt", input_file_path, output_file_path]
    print('calling {}'.format(' '.join(arguments)))
    call(arguments, cwd=os.path.join(boun_parser_dir, 'MD-2.0'))
    print('{} was completed'.format(executed_file_name))

def clean_morphologically_disambiguated_boun_from_file(input_file_path, output_file_path, boun_parser_dir):
    executed_file_name = 'clean_corpus.py'
    arguments = ["python", os.path.join(boun_parser_dir, executed_file_name), input_file_path, output_file_path]
    print('calling {}'.format(' '.join(arguments)))
    call(arguments, cwd=os.path.join(boun_parser_dir, 'CLEAN'))
    print('{} was completed'.format(executed_file_name))

def parse_morphologically_boun(examples, params):
    examples_file_path = os.path.join(os.getcwd(), params['data_dir'], 'boun_parser_raw_examples.txt')
    dump_examples_to_file(examples, examples_file_path)

    morphologically_parsed_examples_file_path = os.path.join(os.getcwd(), params['data_dir'], 'boun_parser_parsed_examples.txt')
    morphologically_disambiguated_examples_file_path = os.path.join(params['data_dir'], 'boun_parser_disambiguated_examples.txt')

    cleaned_examples_file_path = os.path.join(os.getcwd(), params['data_dir'], 'boun_parser_cleaned_examples.txt')
    boun_parser_dir = params['boun_parser_dir']

    parse_morphologically_boun_from_file(examples_file_path, morphologically_parsed_examples_file_path, boun_parser_dir)
    disambiguate_morphologically_boun_from_file(morphologically_parsed_examples_file_path,
                                                morphologically_disambiguated_examples_file_path,
                                                boun_parser_dir)
    clean_morphologically_disambiguated_boun_from_file(morphologically_disambiguated_examples_file_path,
                                                       cleaned_examples_file_path,
                                                       boun_parser_dir)

    load_examples_from_file(examples, cleaned_examples_file_path)

def parse_sentencepiece(sentence, params):
    if sentence.strip() == u'':
        return sentence
    if params['lower_case']:
        sentence = sentence.lower() # this line should be improved by using the lower method of unicode_tr library
    sentence_pieces = params['sp_model'].EncodeAsPieces(sentence)
    parsed_sentence = ' '.join(sentence_pieces)
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
    params = None
    if args.do_morphological_preprocessing:
        if args.morphological_parser_name == 'zemberek':
            if isJVMStarted() == False:
                turn_on_morphological_analyzer(args)
            params = {
                'morphology': JClass('zemberek.morphology.TurkishMorphology').createWithDefaults(),
                'AnalysisFormatters': JClass('zemberek.morphology.analysis.AnalysisFormatters'),
                'omit_suffixes':args.omit_suffixes_after_morphological_preprocessing,
                'preprocess_func':  parse_morphologically_zemberek,
                'lower_case':args.do_lower_case,
                'parse_batchwise': False
            }
        elif args.morphological_parser_name == 'google-research':
            analyzer = analyze.get_analyzer()
            symbol_table = analyzer.input_symbols()

            params = {
                'omit_suffixes': args.omit_suffixes_after_morphological_preprocessing,
                'preprocess_func': parse_morphologically_google_research,
                'analyzer': analyze.get_analyzer(),
                'symbol_table':symbol_table,
                'lower_case': args.do_lower_case,
                'parse_batchwise': False
            }

        elif args.morphological_parser_name == 'boun':
            params = {
                'omit_suffixes': args.omit_suffixes_after_morphological_preprocessing,
                'preprocess_func': parse_morphologically_boun,
                'lower_case': args.do_lower_case,
                'parse_batchwise': True,
                'data_dir': args.data_dir,
                'boun_parser_dir':args.boun_parser_dir
            }

    elif args.do_ngram_preprocessing:
        params = {
            'ngram_size':args.ngram_size,
            'preprocess_func':from_sentence_to_character_ngram_sequence,
            'lower_case': args.do_lower_case
        }
    elif args.do_sentencepiece_preprocessing:
        sp = spm.SentencePieceProcessor()
        sp.Load(args.sp_model_path)
        params = {
            'sp_model':sp,
            'preprocess_func':parse_sentencepiece,
            'lower_case': args.do_lower_case
        }

    return params

def preprocess_examples(examples, args):
    print('-----------------------------------')
    print('Preprocessing new file')
    print('do_morphological_preprocessing:', args.do_morphological_preprocessing)
    print('morphological_parser_name:', args.morphological_parser_name)
    print('zemberek_path:', args.zemberek_path)
    print('omit_suffixes_after_morphological_preprocessing:', args.omit_suffixes_after_morphological_preprocessing)
    print('do_ngram_preprocessing:', args.do_ngram_preprocessing)
    print('ngram_size:', args.ngram_size)
    print('do_lower_case:', args.do_lower_case)
    print('do_sentencepiece_preprocessing:', args.do_sentencepiece_preprocessing)
    print('sp_model_path:', args.sp_model_path)
    print('boun_parser_dir:', args.boun_parser_dir)
    print('-----------------------------------')

    params = get_preprocess_parameters(args)
    if params is None:
        return
    if 'parse_batchwise' in params and params['parse_batchwise'] == True:
        params['preprocess_func'](examples, params)
    else:
        num_lines = 0
        for example in examples:
            num_lines += 1
            if num_lines % 10000 == 0:
                print('num_lines processed', num_lines)
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