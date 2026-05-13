import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .pipeline import Pipeline


class Visualiser:
    """Plot learning curves and confusion matrix from a trained Pipeline."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.trainer = pipeline.trainer
        self.class_names = pipeline.data_processor.class_names

    def plot_all(self, save_path="training_results.png"):
        """Three-panel figure: loss, metrics, confusion matrix."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        self._plot_loss(axes[0])
        self._plot_metrics(axes[1])
        self._plot_confusion_matrix(axes[2])

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  Saved → {save_path}")

    def plot_loss(self):
        """Standalone loss curve."""
        fig, ax = plt.subplots(figsize=(6, 4))
        self._plot_loss(ax)
        plt.tight_layout()
        plt.show()

    def plot_metrics(self):
        """Standalone accuracy & F1 curve."""
        fig, ax = plt.subplots(figsize=(6, 4))
        self._plot_metrics(ax)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self):
        """Standalone confusion matrix."""
        fig, ax = plt.subplots(figsize=(6, 5))
        self._plot_confusion_matrix(ax)
        plt.tight_layout()
        plt.show()

    # ── Private helpers ──

    def _get_history(self):
        return pd.DataFrame(self.trainer.history)

    def _plot_loss(self, ax):
        h = self._get_history()
        ax.plot(h["epoch"], h["train_loss"], "o-", label="Train Loss", linewidth=2)
        if "val_loss" in h:
            ax.plot(h["epoch"], h["val_loss"], "s-", label="Val Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_metrics(self, ax):
        h = self._get_history()
        if "val_acc" in h:
            ax.plot(h["epoch"], h["val_acc"], "o-", label="Val Accuracy", linewidth=2)
            ax.plot(h["epoch"], h["val_f1"],  "s-", label="Val F1 (macro)", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Validation Metrics")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_confusion_matrix(self, ax):
        cm = confusion_matrix(self.trainer.y_true, self.trainer.y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
