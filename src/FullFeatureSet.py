from src.uci_har_dataset import UciHarDataset


class FullFeatureSet:
    def __init__(self, dataset: UciHarDataset):
        self.dataset = dataset
        self.x = self.dataset.x
        self.y = self.dataset.y