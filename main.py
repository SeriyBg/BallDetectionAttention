from misc.config import Params
from models.ssd_voc import ssd_ball_detector

if __name__ == '__main__':
    params = Params("config.txt")
    ssd_ball_detector(params)


