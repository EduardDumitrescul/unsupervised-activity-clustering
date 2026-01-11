import numpy as np
from collections import Counter
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt


class MeanShiftModelRunner:
    def __init__(self, dataset, feature_set, name="MeanShift"):
        self.dataset = dataset
        self.feature_set = feature_set
        self.name = name
        self.model = None
        self.label_map = {}
        self.bandwidth = None

    def run(self, bandwidth = None):
        train_x = self.feature_set.x['train']
        train_y = self.feature_set.y['train'].ravel()

        try:
            
            self.model = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1, max_iter=1000)
            self.model.fit(train_x)

            cluster_assignments = self.model.labels_
            unique_clusters = np.unique(cluster_assignments)

            for cluster_id in unique_clusters:
                mask = (cluster_assignments == cluster_id)
                labels_in_cluster = train_y[mask]
                if len(labels_in_cluster) > 0:
                    majority_label = Counter(labels_in_cluster).most_common(1)[0][0]
                    self.label_map[cluster_id] = majority_label
            
            unique_clusters_count = len(unique_clusters)
            acc, silhouette = self.evaluate('test')

            return acc, silhouette, unique_clusters_count
        except:
            return None, None, None

    def evaluate(self, set_name='test', verbose = False, filename=None):
        test_x = self.feature_set.x[set_name]
        test_y_true = self.feature_set.y[set_name].ravel()

        test_clusters = self.model.predict(test_x)

        default_label = Counter(self.feature_set.y['train'].ravel()).most_common(1)[0][0]
        y_pred = [self.label_map.get(c, default_label) for c in test_clusters]

        accuracy = accuracy_score(test_y_true, y_pred)
        if verbose:
            print(f"\nAccuracy on {self.name} ({set_name}): {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(
                test_y_true,
                y_pred,
                target_names=list(self.dataset.activity_labels.values())
            ))

            self.plot_confusion_matrix(test_y_true, y_pred, filename=filename)
            
            
        silhouette = silhouette_score(test_x, test_clusters)
        return accuracy, silhouette

    def plot_confusion_matrix(self, y_true, y_pred, filename=None):
        labels = list(self.dataset.activity_labels.values())
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                    xticklabels=labels, yticklabels=labels)
        plt.title(f'Mean Shift Confusion Matrix')
        plt.ylabel('True Activity')
        plt.xlabel('Predicted Activity')
        if filename:
            plt.savefig(filename)
        plt.show()