from jpype import *
import os
from custom_tokenizers.CustomTokenizerUtils import *
import random
class ZemberekTokenizer(object):

  """Runs Zemberek's morphological tokenization."""
  def __init__(self,
               vocab=None,
               unk_token="[UNK]",
               **kwargs):

    self.unk_token = unk_token
    self.vocab = vocab
    self.kwargs = kwargs
    self.init()

  def init(self):
    if isJVMStarted() == True:
      self.turn_off_morphological_analyzer()

    if isJVMStarted() == False:
        self.turn_on_morphological_analyzer()

    self.morphology = JClass('zemberek.morphology.TurkishMorphology').createWithDefaults()
    self.AnalysisFormatters = JClass('zemberek.morphology.analysis.AnalysisFormatters')

  def tokenize(self, sentence):
      if sentence.strip() == u'':
          return sentence

      if self.kwargs.get('lower_case', False):
          sentence = sentence.lower()  # this line should be improved by using the lower method of unicode_tr library

      sentence_analysis = self.morphology.analyzeAndDisambiguate(JString(sentence))
      parsed_tokens = []
      for sentence_word_analysis in sentence_analysis:
          word_analysis = sentence_word_analysis.getWordAnalysis()
          word_analysis_results = word_analysis.getAnalysisResults()
          input_surface_word = str(word_analysis.getInput())
          if len(word_analysis_results) > 0:
              single_anaysis = word_analysis_results[0]
              parsed_word = str(self.AnalysisFormatters.SURFACE_SEQUENCE.format(single_anaysis))
              parsed_word = align_cases(input_surface_word, parsed_word)
              if self.kwargs.get('omit_suffixes', False):
                  parsed_word = parsed_word.split(' ')[0]

              parsed_word = input_surface_word[0] + parsed_word[
                                                    1:]  # try to make sure at least the title case structure of the input word is preserved
          else:
              parsed_word = str(word_analysis.getInput())
          parsed_tokens.extend(parsed_word.split(' '))

      output_tokens = [convert_to_unicode(token)
                       if token in self.vocab else self.unk_token
                       for token in parsed_tokens]
      return output_tokens

  def turn_on_morphological_analyzer(self):
      java_home_path = self.kwargs.get('java_home_path', None)
      if java_home_path is not None:
          os.environ['JAVA_HOME_PATH'] = java_home_path


      cpath = f'-Djava.class.path=%s' % (self.kwargs['zemberek_path'])

      startJVM(
          getDefaultJVMPath(),
          '-ea',
          cpath,
          convertStrings=False
      )

  def turn_off_morphological_analyzer(self):
      shutdownJVM()