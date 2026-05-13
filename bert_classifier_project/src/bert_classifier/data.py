import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

from .config import Config


class DataProcessor:
    """Load, encode, tokenize, and batch text data for BERT."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.label_encoder = LabelEncoder()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    # ── Public API ──

    def load(self, allowed_classes=None):
        """
        Load dataset, filter classes, encode labels, and train/test split.

        Returns:
            x_train, x_test  – text arrays (shape N,1)
            y_train, y_test  – integer label arrays (shape N,)
        """
        df = self._read_file(self.cfg.data_path)
        df = df.dropna(subset=[self.cfg.text_col, self.cfg.label_col]).reset_index(drop=True)
        df = self._filter_classes(df, allowed_classes)

        x = df[self.cfg.text_col].to_numpy().reshape(-1, 1)
        y_raw = df[self.cfg.label_col].to_numpy()  # raw strings, no encoding yet

        x_train, x_test, y_raw_train, y_raw_test = train_test_split(
            x, y_raw,
            test_size=self.cfg.test_size,
            stratify=y_raw,
            random_state=0,
        )

        # fit ONLY on train
        y_train = self.label_encoder.fit_transform(y_raw_train)
        y_test  = self.label_encoder.transform(y_raw_test)

        print(f"  Classes: {self.num_classes}  |  Train: {len(x_train):,}  |  Test: {len(x_test):,}")
        print(f"  Labels: {list(self.class_names)}")
        return x_train, x_test, y_train, y_test

    def build_dataloaders(self, x_train, x_test, y_train, y_test):
        """
        Tokenize splits and wrap them in DataLoaders.

        Returns:
            train_dl, test_dl
        """
        train_ds = self._tokenize(x_train, y_train)
        test_ds  = self._tokenize(x_test, y_test)

        train_dl = DataLoader(train_ds, batch_size=self.cfg.batch_size, collate_fn=self.collator, shuffle=True)
        test_dl  = DataLoader(test_ds,  batch_size=self.cfg.batch_size, collate_fn=self.collator)

        print(f"  Train batches: {len(train_dl)}  |  Test batches: {len(test_dl)}")
        return train_dl, test_dl

    @property
    def num_classes(self):
        """Number of classes discovered during load()."""
        return len(self.label_encoder.classes_)

    @property
    def class_names(self):
        """Array of class name strings."""
        return self.label_encoder.classes_

    # ── Private helpers ──

    @staticmethod
    def _read_file(path: str) -> pd.DataFrame:
        if path.endswith(".xlsx") or path.endswith(".xls"):
            return pd.read_excel(path)
        return pd.read_csv(path)

    def _filter_classes(self, df, allowed_classes):
        if allowed_classes is None:
            return df
        mask = df[self.cfg.label_col].isin(allowed_classes)
        print(f"  Filtered {mask.sum():,}/{len(df):,} rows → {len(allowed_classes)} active classes")
        return df[mask].reset_index(drop=True)

    def _tokenize(self, texts, labels):
        flat = [t for row in texts for t in row]
        encoded = {}
        for text in flat:
            tok = self.tokenizer(text, truncation=True)
            for key in tok:
                encoded.setdefault(key, []).append(tok[key])
        encoded["label"] = labels.tolist()
        return Dataset.from_dict(encoded)
