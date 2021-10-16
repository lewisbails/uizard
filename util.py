import json
from tqdm import tqdm
import pathlib
import numpy as np


def combine_jsons(path: str, output: str):
    data = [json.load(open(f, "r", encoding="utf-8"))
            for f in tqdm(list(pathlib.Path(path).glob("*.json")))]
    data = list(np.concatenate(data))
    json.dump(data, open(output, "w"))
