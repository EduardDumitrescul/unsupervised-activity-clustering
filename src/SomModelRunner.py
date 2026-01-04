from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from minisom import MiniSom

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class SomModelRunner:
    def __init__(self, dataset, feature_set, x_size=15, y_size=15, sigma=1.5, lr=0.5, topology='rectangular'):
        self.dataset, self.feature_set = dataset, feature_set
        self.x_size, self.y_size, self.sigma, self.lr = x_size, y_size, sigma, lr
        self.topology = topology
        self.model = None

    def run(self, iterations=1000, num_examples=None):
        data = self.feature_set.x['train']
        if num_examples is not None:
            data = data[:num_examples]
        self.model = MiniSom(self.x_size, self.y_size, data.shape[1], sigma=self.sigma, learning_rate=self.lr,
                             random_seed=42, topology=self.topology)
        print(f'Training SOM({self.x_size}x{self.y_size}) for {iterations} iterations')
        self.model.pca_weights_init(data)
        self.model.train_batch(data, num_iteration=iterations, verbose=True)
        print(f"\nQuantization Error: {self.model.quantization_error(data):.4f}")
        print(f"Topographic Error:  {self.model.topographic_error(data):.4f}")

    def plot_u_matrix(self):
        plt.figure(figsize=(10, 8))
        plt.pcolor(self.model.distance_map().T, cmap='viridis')
        plt.colorbar(label='Inter-neuron Distance')
        plt.title('U-Matrix')
        plt.show()

    def plot_hit_map(self):
        plt.figure(figsize=(10, 8))
        plt.pcolor(self.model.activation_response(self.feature_set.x['train']).T, cmap='Blues')
        plt.colorbar(label='Frequency')
        plt.title('Hit Map')
        plt.show()

    def plot_labeled_grid(self):
        cfg = {1: ('WK', '#1f77b4'), 2: ('UP', '#17becf'), 3: ('DN', '#9467bd'),
               4: ('SI', '#d62728'), 5: ('ST', '#ff7f0e'), 6: ('LY', '#8c564b')}
        train_x, train_y = self.feature_set.x['train'], self.feature_set.y['train'].ravel()

        counts = {}
        for x, y in zip(train_x, train_y):
            w = self.model.winner(x)
            counts.setdefault(w, Counter())[y] += 1

        plt.figure(figsize=(16, 14))
        plt.pcolor(self.model.distance_map().T, cmap='bone', alpha=0.15)
        for (nx, ny), c in counts.items():
            tot = sum(c.values())
            (m_val, m_cnt), *others = c.most_common(2)
            perc = m_cnt / tot
            label = f"{cfg[m_val][0]}"

            plt.text(nx + 0.5, ny + 0.5, label, color=cfg[m_val][1], fontsize=7,
                     ha='center', va='center', fontweight='bold' if perc > 0.8 else 'normal')

        legend = [mpatches.Patch(color=v[1], label=v[0]) for v in cfg.values()]
        plt.legend(handles=legend, loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.show()

    def evaluate_test_set(self):
        tr_x, tr_y = self.feature_set.x['train'], self.feature_set.y['train'].ravel()
        ts_x, ts_y = self.feature_set.x['test'], self.feature_set.y['test'].ravel()

        neuron_labels = {}
        for x, y in zip(tr_x, tr_y):
            neuron_labels.setdefault(self.model.winner(x), []).append(y)

        final_map = {k: Counter(v).most_common(1)[0][0] for k, v in neuron_labels.items()}
        fallback = Counter(tr_y).most_common(1)[0][0]
        preds = np.array([final_map.get(self.model.winner(x), fallback) for x in ts_x])

        labels = sorted(self.dataset.activity_labels.keys())
        names = [self.dataset.activity_labels[i] for i in labels]

        print(f"Test Accuracy: {accuracy_score(ts_y, preds):.4f}")
        print(classification_report(ts_y, preds, labels=labels, target_names=names))

        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix(ts_y, preds), annot=True, fmt='d', cmap='Blues', xticklabels=names,
                    yticklabels=names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        return accuracy_score(ts_y, preds)