import logging
import argparse
from src.data import configure_logging
from src.create_dataset import create_dataset

if __name__ == '__main__':
    configure_logging("log_train_cnn.log", True, log_lvl=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--dataset", type=str, help="absolute path dataset")
    args = parser.parse_args()
    if args.dataset:
        create_dataset(args.dataset)
    else:
        logging.error("dataset must not be empty")
