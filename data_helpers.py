import pathlib
import json
import torch
import pytorch_lightning as pl
import numpy as np
from typing import Iterable, Optional, Union
from transformers import BatchEncoding
from tqdm import tqdm


label2idx = {l: i for i, l in enumerate([
    "abstract",
    "author",
    "caption",
    "date",
    "equation",
    "figure",
    "footer",
    "list",
    "paragraph",
    "reference",
    "section",
    "table",
    "title",
])}


class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, X: BatchEncoding, y: Optional[Iterable[str]] = None):
        if X.is_fast:
            # dealing with tokenizers.Encodings
            self.X = [{"input_ids": torch.LongTensor(
                x.ids), "attention_mask": torch.LongTensor(x.attention_mask)} for x in X.encodings]
        else:
            self.X = X
        self.y = torch.LongTensor([label2idx[yi] for yi in y]) if y else y

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

    def __len__(self):
        return len(self.X)


class TextClassificationDataModule(pl.LightningDataModule):
    def __init__(self,
                 tokenizer,
                 training_data: list,
                 batch_size: int = 32,
                 p_val: float = 0.1,
                 **tokenizer_kwargs):
        super(TextClassificationDataModule, self).__init__()
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.training_data = training_data
        self.batch_size = batch_size
        self.p_val = p_val

    def prepare_data(self):
        # load jsons into train and test arrays
        train = [json.load(open(f, "r", encoding="utf-8"))
                 for f in tqdm(self.training_data)]

        self.train = np.concatenate(train)

    def setup(self, stage):
        # extract the text and label, tokenize, split, and prepare the Datasets
        if stage == "fit" or stage is None:
            X_train = [t["text"] for t in self.train]
            y_train = [t["label"] for t in self.train]

            X_train = self.tokenizer(X_train, **self.tokenizer_kwargs)

            train_dataset = TextClassificationDataset(X_train, y_train)

            n_val = int(len(train_dataset) * self.p_val)
            n_train = len(train_dataset) - n_val
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                train_dataset, [n_train, n_val])  # TODO: stratified split

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)
