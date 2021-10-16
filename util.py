import json
from tqdm import tqdm
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split


def sample_jsons(path: str, output: str, n=50000):
    data = [json.load(open(f, "r", encoding="utf-8"))
            for f in tqdm(list(pathlib.Path(path).glob("*.json")))]
    data = list(np.concatenate(data))
    print(len(data))
    y = [x["label"] for x in data]
    X_train, X_test = train_test_split(data, test_size=n, stratify=y)
    json.dump(X_test, open(output, "w"))
