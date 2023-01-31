import unittest
import logging
from src.create_dataset import load_dataset, prepare_dataset
from src.data import configure_logging


class TestCreateDataset(unittest.TestCase):
    def test_load_dataset(self):
        configure_logging("log_ut_create_dataset.log", True, log_lvl=logging.DEBUG)
        flag, _, _ = load_dataset("../dataset")
        self.assertEqual(flag, True)

        flag, _, _ = load_dataset("/dataset")
        self.assertEqual(flag, False)

    def test_prepare_dataset(self):
        _, a, b = load_dataset("../dataset")
        flag, _ = prepare_dataset(a, b)
        self.assertEqual(flag, True)


if __name__ == '__main__':
    unittest.main()
