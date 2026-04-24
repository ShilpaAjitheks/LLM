# LLM Project....

# BERT Multi-Class Text Classification Pipeline

Fine-tune `bert-base-cased` for multi-class text classification with a clean, modular pipeline.

## Architecture

| Class | Role |
|---|---|
| `Config` | All hyperparameters in one dataclass |
| `DataProcessor` | Load, encode, tokenize, build DataLoaders |
| `BERTTrainer` | Train, evaluate, predict, save |
| `Pipeline` | Orchestrates everything via `.run()` |
| `Visualiser` | Loss curves, metrics, confusion matrix |


## Dataset

Tested on [BBC News](https://www.kaggle.com/datasets/gpreda/bbc-news) (5 classes: business, entertainment, politics, sport, tech). Swap in any CSV with `text` and `label` columns.

## Results

- **Val Accuracy:** 99.3%
- **Val F1 (macro):** 99.3%

## Requirements

```
torch, transformers, datasets, scikit-learn, pandas, matplotlib, seaborn
```

GPU recommended. On CPU, lower `epochs` and `batch_size`.
