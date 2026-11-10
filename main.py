"""Run the whole sentiment-analysis pipeline.

    python main.py                  # everything: preprocess -> EDA -> train -> evaluate
    python main.py --step eda       # only EDA
    python main.py --step train     # only baseline + training (saves models)
    python main.py --step evaluate  # only evaluation (needs trained models)
    python main.py --sample 5000    # quick test run on 5000 reviews
"""
import argparse

from src import config
from src.data_preprocessing import get_data, split_data
from src.eda import run_eda
from src.evaluate import run_evaluation
from src.train import load_all_models, run_baseline, save_models, train_all


def main():
    parser = argparse.ArgumentParser(description="Sentiment analysis of IMDb movie reviews")
    parser.add_argument("--step", choices=["all", "eda", "train", "evaluate"], default="all")
    parser.add_argument("--sample", type=int, default=None,
                        help="use only N random reviews (fast test run)")
    args = parser.parse_args()

    if not config.DATA_PATH.exists():
        raise SystemExit(f"Dataset not found: {config.DATA_PATH}\n"
                         "Download 'IMDB Dataset.csv' from Kaggle into the project folder.")

    df = get_data(args.sample)

    if args.step in ("all", "eda"):
        run_eda(df)

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"\nTraining samples: {len(X_train)} | Testing samples: {len(X_test)}")

    if args.step in ("all", "train"):
        run_baseline(y_train, y_test)
        models, cv_df = train_all(X_train, y_train)
        print("\n", cv_df.to_string(index=False))
        save_models(models, cv_df)
    else:
        models = None

    if args.step in ("all", "evaluate"):
        models = models or load_all_models()
        run_evaluation(models, X_test, y_test)


if __name__ == "__main__":
    main()
