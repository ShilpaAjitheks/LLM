import numpy as np
import torch

from .config import Config
from .data import DataProcessor
from .trainer import BERTTrainer


def _set_seed(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class Pipeline:
    """End-to-end orchestrator: data → train → evaluate → save."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        _set_seed()
        self.data_processor = DataProcessor(cfg)
        self.trainer = None      # created after data is loaded (needs num_classes)
        self.report = None

    def run(self, allowed_classes=None):
        """
        Execute the full pipeline.

        Args:
            allowed_classes: optional list of class names to train on (default: all)

        Returns:
            self (for chaining, e.g. pipeline.run().predict(...))
        """
        # Step 1 — Data
        print("Step 1/6 · Loading data...")
        x_train, x_test, y_train, y_test = self.data_processor.load(allowed_classes)

        # Step 2 — DataLoaders
        print("\nStep 2/6 · Preparing dataloaders...")
        train_dl, test_dl = self.data_processor.build_dataloaders(
            x_train, x_test, y_train, y_test,
        )
        self.test_dl = test_dl

        # Step 3 — Model setup
        print("\nStep 3/6 · Setting up model...")
        self.trainer = BERTTrainer(self.cfg, self.data_processor.num_classes)
        self.trainer.setup(train_dl)

        # Step 4 — Training
        print("\nStep 4/6 · Training...")
        self.trainer.train(train_dl, eval_dl=test_dl, class_names=self.data_processor.class_names)

        # Step 5 — Evaluation
        print("\nStep 5/6 · Evaluating...")
        self.report = self.trainer.evaluate(test_dl, self.data_processor.class_names)
        print(self.report)

        # Step 6 — Save
        print("Step 6/6 · Saving model...")
        self.trainer.save(self.report, self.data_processor.tokenizer, self.data_processor.label_encoder)

        print("\n" + "=" * 50)
        print("  Pipeline complete.")
        print("=" * 50)
        return self

    def predict(self, texts):
        """
        Classify new texts using the trained model.

        Args:
            texts: list[str]

        Returns:
            numpy array of predicted class name strings
        """
        if self.trainer is None:
            raise RuntimeError("Call .run() before .predict()")
        return self.trainer.predict(texts, self.data_processor)
