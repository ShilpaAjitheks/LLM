import argparse

from .config import Config
from .pipeline import Pipeline


def main():
    p = argparse.ArgumentParser(description="Fine-tune BERT for multi-class text classification.")
    p.add_argument("--data-path", default="bbc_news_full.csv")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label_text")
    p.add_argument("--checkpoint", default="bert-base-cased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--optimizer", default="AdamW",
                   choices=["AdamW", "RMSprop", "Adagrad", "Adadelta"])
    p.add_argument("--scheduler", default="linear",
                   choices=["linear", "cosine", "polynomial", "exponential"])
    p.add_argument("--save-dir", default="bert_multiclass_model")
    p.add_argument("--classes", nargs="*", help="Train on this subset of class names")
    p.add_argument("--predict", nargs="+", help="After training, classify these texts")
    p.add_argument("--plot", action="store_true", help="Save training_results.png after training")
    args = p.parse_args()

    cfg = Config(
        data_path=args.data_path,
        text_col=args.text_col,
        label_col=args.label_col,
        checkpoint=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        lr_scheduler=args.scheduler,
        save_dir=args.save_dir,
    )

    pipeline = Pipeline(cfg).run(allowed_classes=args.classes)

    if args.plot:
        # imported here so matplotlib stays optional for plain training
        from .visualizer import Visualiser
        Visualiser(pipeline).plot_all()

    if args.predict:
        for text, label in zip(args.predict, pipeline.predict(args.predict)):
            print(f"{label:>15s}  ←  {text[:80]}")


if __name__ == "__main__":
    main()
