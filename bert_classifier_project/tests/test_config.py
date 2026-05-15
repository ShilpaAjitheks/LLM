from bert_classifier import Config


def test_defaults():
    cfg = Config()
    assert cfg.checkpoint == "bert-base-cased"
    assert cfg.epochs == 3
    assert cfg.batch_size == 16
    assert cfg.optimizer == "AdamW"
    assert cfg.lr_scheduler == "linear"


def test_overrides():
    cfg = Config(epochs=10, batch_size=32, optimizer="RMSprop")
    assert cfg.epochs == 10
    assert cfg.batch_size == 32
    assert cfg.optimizer == "RMSprop"
