
import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

class SignDetectOptions:
    def __init__(self):
        # create option
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data and test data",
                                 default=os.path.join(file_dir, "data"))

        # TRAINING options
        self.parser.add_argument("--model",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 choices=["LeNet", "AlexNet", "VGG", "ResNet"],
                                 default="LeNet")
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size of training set and test set",
                                 default=16)
        self.parser.add_argument("--num_classes",
                                 type=int,
                                 help="labels varieties",
                                 default=43)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=10)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
