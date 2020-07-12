r"""CLI for Tensorflow Datasets
"""

import argparse
from absl import app

import tensorflow_datasets as tfds


def main(_=None):
  parser = argparse.ArgumentParser(
      description='Tensorflow Datasets CLI tool',
  )
  parser.add_argument(
      '-v',
      '--version',
      action='version',
      version='Tensorflow Datasets: ' + tfds.__version__
  )

  _ = parser.parse_args()


if __name__ == '__main__':
  app.run(main)
