import argparse

from misc.config import Params
from models.ssd_voc import train_ssd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the configuration file', type=str, default='config.txt')
    args = parser.parse_args()

    print('Config path: {}'.format(args.config))

    params = Params(args.config)
    train_ssd(params)


