import tensorflow as tf
import tensorflow_datasets as tfds


def add_build_dataset_parser(subparsers):
  build_parser = subparsers.add_parser(
      'build', help='Build the dataset files')
  build_parser.add_argument(
      'build',
      action='store_true',
      help='Name of the dataset(s) to be created'
  )
  build_parser.set_defaults(func=build_dataset)


def build_dataset(_):
  print('Buildling args')  # TODO
