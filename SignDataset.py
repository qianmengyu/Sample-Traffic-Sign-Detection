
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle

class traffic_sign_dataset(Dataset):
    '''
    numpy to Dataset
    '''
    def __init__(self, X, y):
        self.len = len(X)
        self.features = torch.tensor(torch.from_numpy(X), dtype=torch.float32)
        self.target = torch.tensor(torch.from_numpy(y), dtype=torch.long)
        
    def __getitem__(self, index):
        # 网络需要(batch, 3, n, n)形式，需要调整一下
        sample = self.features[index].permute(2, 0, 1)
        return sample, self.target[index]

    def __len__(self):
        return self.len

class SignDataLoader():
    def __init__(self, data_path, BATCH_SIZE):
        # load the dataset
        training_file = data_path + "/train.p"
        validation_file = data_path + "/valid.p"
        testing_file = data_path + "/test.p"

        print(" ---- Load Data File ---- ")
        with open(training_file, mode='rb') as f:
            a_train = pickle.load(f)
        with open(validation_file, mode='rb') as f:
            a_valid = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            a_test = pickle.load(f)
        print(" ---- Load Finished! ---- ")
        
        X_train, y_train = a_train['features'], a_train['labels']
        X_valid, y_valid = a_valid['features'], a_valid['labels']
        X_test, y_test = a_test['features'], a_test['labels']
        print("Train set shape is: ", X_train.shape, " labels shape: ", y_train.shape)
        print("Valid set shape is: ", X_valid.shape, " labels shape: ", y_valid.shape)
        print("Test set shape is:  ", X_test.shape, " labels shape: ", y_test.shape)
        
        # transform to torch.Dataset
        train_set = traffic_sign_dataset(X = X_train, y = y_train)
        valid_set = traffic_sign_dataset(X = X_valid, y = y_valid)
        test_set = traffic_sign_dataset(X = X_test, y = y_test)
        
        # transform to DataLoader
        self.train_set_iter = DataLoader(dataset=train_set,
                            batch_size=BATCH_SIZE,
                            shuffle=True)
        self.valid_set_iter = DataLoader(valid_set, 
                                    batch_size=BATCH_SIZE,
                                    shuffle=True)
        self.test_set_iter = DataLoader(dataset=test_set,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True)
    
    def getTrainLoader(self):
        return self.train_set_iter
    
    def getValidLoader(self):
        return self.valid_set_iter
    
    def getTestLoader(self):
        return self.test_set_iter