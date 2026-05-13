from dataclasses import dataclass


@dataclass
class Config:
    """Centralised configuration for the entire pipeline."""

    # ── Data ──
    data_path:   str = "bbc_news_full.csv"
    text_col:    str = "text"
    label_col:   str = "label_text"
    test_size: float = 0.2

    # ── Model ──
    checkpoint:  str = "bert-base-cased"

    # ── Training ──
    epochs:      int = 3
    batch_size:  int = 16
    learning_rate: float = 2e-5
    optimizer:   str = "AdamW"             # AdamW | RMSprop | Adagrad | Adadelta
    lr_scheduler: str = "linear"           # linear | cosine | polynomial | exponential

    # ── Output ──
    save_dir:    str = "bert_multiclass_model"
