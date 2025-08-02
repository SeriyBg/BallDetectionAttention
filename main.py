from models.train import train
from parameter_rader import get_parameters

if __name__ == '__main__':
    params = get_parameters()
    train(params)


