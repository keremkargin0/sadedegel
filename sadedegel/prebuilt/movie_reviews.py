from math import ceil
from os.path import dirname
from pathlib import Path

import numpy as np
from joblib import dump, load as jl_load
from rich.console import Console
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

from ..extension.sklearn import TfidfVectorizer, OnlinePipeline
from ..dataset.movie_sentiment import load_movie_sentiment_train, load_movie_sentiment_test, \
    load_movie_sentiment_target, CORPUS_SIZE, CLASS_VALUES

console = Console()


def empty_model():
    return OnlinePipeline(
        [('tfidf', TfidfVectorizer(tf_method='raw', idf_method='probabilistic', show_progress=True)),
         ('pa', Perceptron())
         ]
    )


def cv(k=3):
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    raw = load_movie_sentiment_train()
    df = pd.DataFrame.from_records(raw)

    BATCH_SIZE = 2000

    kf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
    console.log(f"Corpus Size: {len(df)}")

    scores = []

    for train_indx, test_index in kf.split(df.comment, df.sentiment_class):
        train = df.iloc[train_indx]
        test = df.iloc[test_index]

        n_split = ceil(len(train) / BATCH_SIZE)
        console.log(f"{n_split} batches of {BATCH_SIZE} instances...")

        batches = np.array_split(train, n_split)

        pipeline = empty_model()

        for batch in batches:
            pipeline.partial_fit(batch.comment, batch.sentiment_class,
                                 classes=df.sentiment_class.unique().tolist())

        y_pred = pipeline.predict(test.comment)

        scores.append(f1_score(test.sentiment_class, y_pred, average='macro'))

        console.log(f"Mean F1-Macro: {np.mean(scores)} , Std: {np.std(scores)}")


def build():
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))

    raw = load_movie_sentiment_train()
    df = pd.DataFrame.from_records(raw)
    df = shuffle(df)

    BATCH_SIZE = 2000

    console.log(f"Corpus Size: {len(df)}")

    n_split = ceil(len(df) / BATCH_SIZE)
    console.log(f"{n_split} batches of {BATCH_SIZE} instances...")

    batches = np.array_split(df, n_split)

    pipeline = empty_model()

    for batch in batches:
        pipeline.partial_fit(batch.comment, batch.sentiment_class,
                             classes=df.sentiment_class.unique().tolist())

    console.log("Model build [green]DONE[/green]")

    model_dir = Path(dirname(__file__)) / 'model'

    model_dir.mkdir(parents=True, exist_ok=True)

    dump(pipeline, (model_dir / 'movie_sentiment.joblib').absolute(), compress=('gzip', 9))


def load(model_name="movie_sentiment"):
    return jl_load(Path(dirname(__file__)) / 'model' / f"{model_name}.joblib")

def evaluate():
    try:
        import pandas as pd
    except ImportError:
        console.log(("pandas package is not a general sadedegel dependency."
                     " But we do have a dependency on building our prebuilt models"))
    model = load()

    raw_test = load_movie_sentiment_test()
    test = pd.DataFrame.from_records(raw_test)

    y_pred = model.predict(test.comment)

    console.log(f"Model test accuracy (f1-macro): {f1_score(test.sentiment_class, y_pred, average='macro')}")


if __name__ == '__main__':
    build()
