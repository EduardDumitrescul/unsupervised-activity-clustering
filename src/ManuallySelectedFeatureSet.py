import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.uci_har_dataset import UciHarDataset


class ManuallySelectedFeatureSet:
    MANUAL_FEATURE_SET = {
        # Laying (horizontal) vs Others (vertical)
        # Orientation
        40: 'tGravityAcc-mean()-X',
        41: 'tGravityAcc-mean()-Y',
        42: 'tGravityAcc-mean()-Z',
        # 558: 'angle(X,gravityMean)',
        # 559: 'angle(Y,gravityMean)',
        # 560: 'angle(Z,gravityMean)',

        # Moving (Walking, Stairs UP, Stairs DOWN) vs Non-Moving (Sitting, Standing, Laying)
        # Energy
        15: 'tBodyAcc-sma()',
        # 200: 'tBodyAccMag-mean()',
        201: 'tBodyAccMag-std()',

        # Walking vs Stairs
        # Rhythm
        # 80: 'tBodyAccJerk-mean()-X',
        # 81: 'tBodyAccJerk-mean()-Y',
        # 82: 'tBodyAccJerk-mean()-Z',
        164: 'tBodyGyroJerk-std()-X',
        165: 'tBodyGyroJerk-std()-Y',
        166: 'tBodyGyroJerk-std()- Z',
        120: 'tBodyGyro-mean()-X',
        265: 'fBodyAcc-mean()-X',
        # 502: 'fBodyAccMag-mean()',
        512: 'fBodyAccMag-meanFreq()',

        # Standing vs Sitting
        123: 'tBodyGyro-std()-X',
        124: 'tBodyGyro-std()-Y',
        125: 'tBodyGyro-std()-Z',
        253: 'tBodyGyroJerkMag-std()'
    }

    def __init__(self, dataset: UciHarDataset):
        self.dataset = dataset
        self.x = {}
        self.y = {}
        self._build_feature_set()
        self._scale_feature_set()

    def _build_feature_set(self):
        indices = list(ManuallySelectedFeatureSet.MANUAL_FEATURE_SET.keys())

        for set_name in ['train', 'val', 'test']:
            x = self.dataset.x[set_name][:, indices]
            y = self.dataset.y[set_name]

            self.x[set_name] = x
            self.y[set_name] = y

            print(f'Selected features: {set_name}. Shape: {x.shape}')

    def _scale_feature_set(self):
        scaler = StandardScaler()
        self.x['train'] = scaler.fit_transform(self.x['train'])
        self.x['val'] = scaler.transform(self.x['val'])
        self.x['test'] = scaler.transform(self.x['test'])

        print(f"Scaled features using StandardScaler.")

    def check_feature_label_correlation(self):
        feature_names = list(self.MANUAL_FEATURE_SET.values())
        df = pd.DataFrame(self.x['train'], columns=feature_names)
        df['label'] = self.y['train']

        correlations = df.corr()['label'].sort_values(ascending=False).drop('label')

        plt.figure(figsize=(10, 8))
        sns.barplot(x=correlations.values, y=correlations.index, palette='viridis')
        plt.title("Feature Correlation with Activity Label")
        plt.xlabel("Pearson Correlation Coefficient")
        plt.show()


    def plot_correlation_matrix(self):
        feature_names = list(self.MANUAL_FEATURE_SET.values())
        df = pd.DataFrame(self.x['train'], columns=feature_names)

        corr = df.corr()

        mask = np.triu(np.ones_like(corr, dtype=bool))
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                    center=0, linewidths=.5, cbar_kws={"shrink": .5})

        plt.title("Manual Feature Set: Inter-Feature Correlation Matrix")
        plt.xticks(rotation=45, ha='right')
        plt.show()

    def plot_feature_label_intensity(self):

        feature_names = list(self.MANUAL_FEATURE_SET.values())
        df = pd.DataFrame(self.x['train'], columns=feature_names)

        y_flat = self.y['train'].ravel()
        df['Activity'] = [self.dataset.activity_labels[int(val)] for val in y_flat]

        activity_profiles = df.groupby('Activity').mean()

        plt.figure(figsize=(14, 8))
        sns.heatmap(activity_profiles, annot=True, cmap='RdYlGn', center=0, fmt=".2f")

        plt.title("Feature Intensity per Activity Label (Standardized Means)")
        plt.xlabel("Features")
        plt.ylabel("Activity Labels")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()



