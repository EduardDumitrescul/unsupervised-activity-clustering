import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


from src.uci_har_dataset import UciHarDataset


class PcaFeatureSet:
    def __init__(self, dataset: UciHarDataset, n_components=20):
        self.dataset = dataset
        self.n_components = n_components
        self.pca = None
        self.x = {}
        self.y = {}

        self._build_feature_set()

    def _build_feature_set(self):
        self.pca = PCA(n_components=self.n_components)

        self.x['train'] = self.pca.fit_transform(self.dataset.x['train'])
        self.y['train'] = self.dataset.y['train']

        for set_name in ['test']:
            self.x[set_name] = self.pca.transform(self.dataset.x[set_name])
            self.y[set_name] = self.dataset.y[set_name]

    def plot_explained_variance(self):
        if self.pca is None: return

        ev = self.pca.explained_variance_ratio_
        cv = np.cumsum(ev)
        x = range(1, len(ev) + 1)

        plt.figure(figsize=(10, 6))
        plt.bar(x, ev, alpha=0.5, label='Individual')
        plt.step(x, cv, where='mid', color='red', label='Cumulative')

        plt.xlabel('Principal component index')
        plt.ylabel('Explained variance ratio')
        plt.title(f'PCA: Explained Variance ({cv[-1]:.2%})')
        plt.xticks(range(1, self.n_components + 1))
        plt.legend()
        plt.grid(axis='y', linestyle='--')
        plt.show()