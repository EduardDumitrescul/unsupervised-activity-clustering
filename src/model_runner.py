import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class SvmRunner:
    def __init__(self, dataset, feature_set, model, name="Model"):
        self.dataset = dataset
        self.feature_set = feature_set
        self.model = model
        self.name = name
        self.y_pred = None
        self.y_true = None

    def run(self, eval_set='test', verbose=True):
        self.model.fit(
            self.feature_set.x['train'],
            self.feature_set.y['train'].ravel()
        )

        self.y_true = self.feature_set.y[eval_set].ravel()
        self.y_pred = self.model.predict(self.feature_set.x[eval_set])

        accuracy = accuracy_score(self.y_true, self.y_pred)
        print(f"Accuracy on {self.name}: {accuracy:.4f}")
        if verbose:
            print("\nClassification Report:")
            print(classification_report(
                self.y_true,
                self.y_pred,
                target_names=list(self.dataset.activity_labels.values())
            ))

        return accuracy

    def plot_confusion_matrix(self, filename=None):
        if self.y_pred is None:
            print("Error: Run the model before plotting.")
            return

        labels = list(self.dataset.activity_labels.values())
        cm = confusion_matrix(self.y_true, self.y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)

        plt.title(f'Confusion Matrix: {self.name}')
        plt.ylabel('True Activity')
        plt.xlabel('Predicted Activity')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        plt.show()

    def plot_feature_importance(self, n_repeats=10, filename=None):
        print(f"Calculating feature importance for {self.name}...")

        # 1. Calculate Permutation Importance
        result = permutation_importance(
            self.model,
            self.feature_set.x['test'],
            self.feature_set.y['test'].ravel(),
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )

        # 2. Organize data
        feature_names = list(self.feature_set.MANUAL_FEATURE_SET.values())
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': result.importances_mean,
            'Std': result.importances_std
        }).sort_values(by='Importance', ascending=False)

        # 3. Visualization
        plt.figure(figsize=(12, 8))
        sns.barplot(
            x='Importance',
            y='Feature',
            data=importance_df,
            palette='magma'
        )

        plt.title(f"Permutation Feature Importance: {self.name}")
        plt.xlabel("Decrease in Accuracy (when shuffled)")
        plt.ylabel("Manual Feature")
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        plt.show()

        return importance_df