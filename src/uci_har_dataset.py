import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class UciHarDataset:
    PATH = 'data/UCI-HAR Dataset/'

    def __init__(self):
        self._load_activity_labels()
        self._load_feature_names()
        self._load_features()
        # self._split_train_in_train_and_val()

    def _load_activity_labels(self):
        self.activity_labels = {}
        with open(UciHarDataset.PATH + 'activity_labels.txt', 'r') as f:
            self.activity_labels = {int(line.split()[0]): line.split()[1] for line in f}

        print(f'Loaded activity labels from {UciHarDataset.PATH}/activity_labels.txt')
        print(self.activity_labels)

    def _load_feature_names(self):
        self.feature_names = {}
        with open(UciHarDataset.PATH + 'features.txt', 'r') as f:
            self.feature_names = {int(line.split()[0]): line.split()[1] for line in f}

        print(f'Loaded feature names from {UciHarDataset.PATH}/features.txt')
        # print(self.feature_names)

    def _load_features(self):
        self.x = {}
        self.y = {}

        for set_name in ['train', 'test']:
            x_path = UciHarDataset.PATH + set_name + '/X_' + set_name + '.txt'
            y_path = UciHarDataset.PATH + set_name + '/y_' + set_name + '.txt'

            x = pd.read_csv(x_path, sep=r'\s+', header=None, engine='python')
            y = pd.read_csv(y_path, sep=r'\s+', header=None, engine='python')
            x = np.array(x)
            y = np.array(y)

            self.x[set_name] = x
            self.y[set_name] = y

            print(f'Loaded features from {x_path}. Shape: {x.shape}')
            print(f'Loaded labels from {y_path}. Shape: {y.shape}')

    def _split_train_in_train_and_val(self):
        x = self.x['train']
        y = self.y['train']

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

        self.x['train'] = x_train
        self.y['train'] = y_train
        self.x['val'] = x_val
        self.y['val'] = y_val

        print(f'Split: {x_train.shape} - {y_train.shape} - {x_val.shape} - {y_val.shape}')

    def get_feature_dataframe(self, set_name: str = 'train'):
        df = pd.DataFrame(self.x[set_name], columns=self.feature_names.values())
        return df



