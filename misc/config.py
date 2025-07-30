import os
import configparser
import time
from ast import literal_eval


class Params:
    def __init__(self, path):
        assert os.path.exists(path), 'Cannot find configuration file: {}'.format(path)
        self.path = path

        config = configparser.ConfigParser()

        config.read(self.path)
        params = config['DEFAULT']
        self.issia_path = params.get('issia_path', None)
        if self.issia_path is not None:
            temp = params.get('issia_train_cameras', '1, 2, 3, 4')
            self.issia_train_cameras = [int(e) for e in temp.split(',')]
            temp = params.get('issia_val_cameras', '5, 6')
            self.issia_val_cameras = [int(e) for e in temp.split(',')]

        self.dfl_paths = []
        dfl_paths = params.get('dfl_paths', None)
        if dfl_paths is not None:
            for path in dfl_paths.split(','):
                self.dfl_paths.append(path)

        self.num_workers = params.getint('num_workers', 0)
        self.batch_size = params.getint('batch_size', 4)
        self.epochs = params.getint('epochs', 20)
        self.lr = params.getfloat('lr', 1e-3)

        self.model = params.get('model', 'ssd')
        self.model_name = 'model_{}_{}'.format(self.model, get_datetime())
        self.attention = params.getboolean('attention', False)
        self.attention_backbone_type = params.get('attention_backbone_type', None)
        self.attention_head_type = params.get('attention_head_type', None)
        resize = params.get('transform_resize', None)
        if resize:
            self.transform_resize = literal_eval(resize)
        else :
            self.transform_resize = None
        self.transform_crop = params.getint('transform_crop')

        # self._check_params()

    def _check_params(self):
        assert len(self.dfl_paths) > 0, "At least one DFL dataset path must be provided"
        for _, dfl_path in enumerate(self.dfl_paths):
            assert dfl_path is None or os.path.exists(dfl_path), "Cannot access DFL dataset: {}".format(dfl_path)

        if self.attention:
            assert self.attention_backbone_type is not None or self.attention_head_type is not None, "Attention backbone type or head type must be provided"

    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            print('{}: {}'.format(e, param_dict[e]))
        print('')


def get_datetime():
    return time.strftime("%Y%m%d_%H%M")
