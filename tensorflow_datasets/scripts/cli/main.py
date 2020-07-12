r"""CLI for Tensorflow Datasets

Instructions:

```
tfds new my_dataset
```

This command will generator dataset files needed for adding a new dataset
to tfds

```
my_dataset/
    fake_examples/
    __init__.py
    dataset.py
    dataset_test.py
    fake_data_generator.py
    checksum.txt
```

"""

import argparse
from absl import app

import tensorflow_datasets as tfds
from tensorflow_datasets.scripts.cli.new_dataset import add_new_dataset_parser
from tensorflow_datasets.scripts.cli.build_dataset import add_build_dataset_parser


def _get_parser():
  parser = argparse.ArgumentParser(
      description='Tensorflow Datasets CLI tool',
  )
  parser.add_argument(
      '-v',
      '--version',
      action='version',
      version='Tensorflow Datasets: ' + tfds.__version__
  )

  # Create Subparsers for commands like new and build
  subparsers = parser.add_subparsers(title='Commands',
                                     help="List of all commands")

  # Add subparsers
  add_new_dataset_parser(subparsers)
  add_build_dataset_parser(subparsers)
  return parser


def main(_=None):
  args = _get_parser().parse_args()
  args.func(args)


if __name__ == '__main__':
  app.run(main)
