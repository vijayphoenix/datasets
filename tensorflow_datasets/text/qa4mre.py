# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
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

"""QA4MRE (CLEF 2013): a reading comprehension dataset."""

from __future__ import division
from __future__ import print_function

import os
import xml.etree.ElementTree as ET
from absl import logging

import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """\
@inproceedings{sutcliffe2013overview,
  title={Overview of QA4MRE Main Task at CLEF 2013.},
  author={Sutcliffe, Richard FE and Pe{\~n}as, Anselmo and Hovy, 
  Eduard H and Forner, Pamela and Rodrigo, {\'A}lvaro and Forascu, 
  Corina and Benajiba, Yassine and Osenova, Petya},
  booktitle={CLEF (Working Notes)},
  year={2013}
}
"""

_DESCRIPTION = """
QA4MRE dataset was created for the CLEF 2013 shared task to promote research in 
question answering and reading comprehension. The dataset contains a supporting 
passage and a set of questions corresponding to the passage. Multiple options 
for answers are provided for each question, of which only one is correct. The 
training and test datasets are available for the main track.
Additional gold standard documents are available for two pilot studies: one on 
alzheimers data, and the other on entrance exams data.
"""

_BASE_URL = 'http://nlp.uned.es/clef-qa/repository/js/scripts/downloadFile.php?file=/var/www/html/nlp/clef-qa/repository/resources/QA4MRE/'

_TRACKS = ('main', 'alzheimers', 'entrance_exam')
_PATH_TMPL_MAIN_GS = '2013/Main_Task/Training_Data/Goldstandard/QA4MRE-2013-{}_GS.xml'
_LANGUAGES_MAIN = ('AR', 'BG', 'EN', 'ES', 'RO')

_PATH_ALZHEIMER = '2013/Biomedical_About_Alzheimer/Training_Data/Goldstandard/QA4MRE-2013_BIO_GS-RUN.xml'
_PATH_ENTRANCE_EXAM = '2013/Entrance_Exams/Training_Data/Goldstandard/qa4mre-exam-test-withanswer.xml'


class Qa4mreConfig(tfds.core.BuilderConfig):
  """BuilderConfig for Qa4mre."""

  @tfds.core.disallow_positional_args
  def __init__(self, track='main', language='EN', **kwargs):
    """BuilderConfig for Qa4Mre.

    Args:
      track: string, the task track: main/alzheimers/entrance_exam.
      language: string, Acronym for language in the main task.
      **kwargs: keyword arguments forwarded to super.
    """
    if track.lower() not in _TRACKS:
      raise ValueError(
          'Incorrect track. Track should be one of the following: ', _TRACKS)

    if track.lower() != 'main' and language.upper() != 'EN':
      logging.warn(
          'Only English documents available for Alzheimers and Entrance Exam '
          'tracks. Setting English by default.')
      language = 'EN'

    if track.lower() == 'main' and language.upper() not in _LANGUAGES_MAIN:
      raise ValueError(
          'Incorrect language for the main track. Correct options: ',
          _LANGUAGES_MAIN)

    self.track = track.lower()
    self.lang = language.upper()

    name = self.track + '.' + self.lang

    description = _DESCRIPTION
    description += 'This configuration includes the {} track for {} language.'.format(
        self.track, self.lang)

    super(Qa4mreConfig, self).__init__(
        name=name,
        description=description,
        version=tfds.core.Version('0.1.0'),
        **kwargs)


class Qa4mre(tfds.core.GeneratorBasedBuilder):
  """QA4MRE dataset from CLEF 2013 shared task."""

  VERSION = tfds.core.Version('0.1.0')

  BUILDER_CONFIGS = [
      Qa4mreConfig(track='main', language='AR'),  # Main track Arabic (main.AR)
      Qa4mreConfig(track='main',
                   language='BG'),  # Main track Bulgarian (main.BG)
      Qa4mreConfig(track='main', language='EN'),  # Main track English (main.EN)
      Qa4mreConfig(track='main', language='ES'),  # Main track Spanish (main.ES)
      Qa4mreConfig(track='main',
                   language='RO'),  # Main track Romanian (main.RO)
      Qa4mreConfig(track='alzheimers', language='EN'),  # (alzheimers.EN)
      Qa4mreConfig(track='entrance_exam', language='EN'),  # (entrance_exam.EN)
  ]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            'topic_id':
                tfds.features.Text(),
            'topic_name':
                tfds.features.Text(),
            'test_id':
                tfds.features.Text(),
            'document_id':
                tfds.features.Text(),
            'document_str':
                tfds.features.Text(),
            'question_id':
                tfds.features.Text(),
            'question_str':
                tfds.features.Text(),
            'answer_options':
                tfds.features.Sequence({
                    'answer_id': tfds.features.Text(),
                    'answer_str': tfds.features.Text()
                }),
            'correct_answer_id':
                tfds.features.Text(),
            'correct_answer_str':
                tfds.features.Text(),
        }),

        # No default supervised keys because both passage and question are used
        # to determine the correct answer.
        supervised_keys=None,
        homepage='http://nlp.uned.es/clef-qa/repository/pastCampaigns.php',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    cfg = self.builder_config
    download_urls = dict()

    if cfg.track == 'main':
      download_urls['main.{}'.format(cfg.lang)] = os.path.join(
          _BASE_URL, _PATH_TMPL_MAIN_GS.format(cfg.lang))

    if cfg.track == 'alzheimers':
      download_urls['alzheimers.EN'] = os.path.join(_BASE_URL, _PATH_ALZHEIMER)

    if cfg.track == 'entrance_exam':
      download_urls['entrance_exam.EN'] = os.path.join(_BASE_URL,
                                                       _PATH_ENTRANCE_EXAM)

    downloaded_files = dl_manager.download_and_extract(download_urls)

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                'filepath':
                    downloaded_files['{}.{}'.format(cfg.track, cfg.lang)]
            })
    ]

  def _generate_examples(self, filepath):
    """Yields examples."""
    with tf.io.gfile.GFile(filepath, 'rb') as f:
      tree = ET.parse(f)
      root = tree.getroot()  # test-set
      for topic in root:
        topic_id = topic.attrib['t_id']
        topic_name = topic.attrib['t_name']
        for test in topic:
          test_id = test.attrib['r_id']
          for document in test.iter('doc'):
            document_id = document.attrib['d_id']
            document_str = document.text
          for question in test.iter('q'):
            question_id = question.attrib['q_id']
            for q_text in question.iter('q_str'):
              question_str = q_text.text
            possible_answers = list()
            for answer in question.iter('answer'):
              answer_id = answer.attrib['a_id']
              answer_str = answer.text
              possible_answers.append({
                  'answer_id': answer_id,
                  'answer_str': answer_str
              })
              if 'correct' in answer.attrib:
                correct_answer_id = answer_id
                correct_answer_str = answer_str

            id_ = topic_id + '_' + topic_name + '_' + test_id + '_' + question_id

            logging.info('ID: %s', id_)

            yield (id_, {
                'topic_id': topic_id,
                'topic_name': topic_name,
                'test_id': test_id,
                'document_id': document_id,
                'document_str': document_str,
                'question_id': question_id,
                'question_str': question_str,
                'answer_options': possible_answers,
                'correct_answer_id': correct_answer_id,
                'correct_answer_str': correct_answer_str,
            })
