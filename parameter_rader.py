import argparse

from misc.config import Params


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the configuration file', type=str, default='config.txt')
    parser.add_argument('--section', help='Config section', type=str, default='DEFAULT')
    args = parser.parse_args()

    print('Config path: {}'.format(args.config))
    return Params(args.config, args.section)