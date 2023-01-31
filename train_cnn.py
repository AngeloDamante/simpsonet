import logging
import argparse
from src.data import configure_logging
from src.create_dataset import load_dataset, prepare_dataset
from src.data import k_img_size
from src.create_cnn import create_cnn, train_model

if __name__ == '__main__':
    configure_logging("log_train_cnn.log", True, log_lvl=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--dataset", type=str, required=True, help="absolute path dataset")
    parser.add_argument("-S", "--size", type=tuple, default=k_img_size, help="image size")
    args = parser.parse_args()

    is_good_creation, images, labels = load_dataset(args.dataset)
    if not is_good_creation:
        exit(1)

    is_good_preparation, dataset = prepare_dataset(images, labels)
    if not is_good_preparation:
        exit(1)

    cnn_model = create_cnn(args.size)
    compiled_model, history = train_model(cnn_model,
                                          x_train=dataset[0],
                                          x_test=dataset[1],
                                          y_train=dataset[2],
                                          y_test=dataset[3])
    compiled_model.save("cnn.h5")
