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

"""Tests for Qa4mre dataset module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_datasets import testing
from tensorflow_datasets.text import qa4mre


class Qa4mreMainTest(testing.DatasetBuilderTestCase):
  DATASET_CLASS = qa4mre.Qa4mre

  DL_EXTRACT_RESULT = {
      "main.AR": "main.AR.xml",
      "main.BG": "main.BG.xml",
      "main.EN": "main.EN.xml",
      "main.ES": "main.ES.xml",
      "main.RO": "main.RO.xml",
      "alzheimers.EN": "alzheimers.EN.xml",
      "entrance_exam.EN": "entrance_exam.EN.xml"
  }

  SPLITS = {
      "train": 3,  # Number of fake train example
  }


if __name__ == "__main__":
  testing.test_main()
