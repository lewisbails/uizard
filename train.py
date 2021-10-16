import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AutoTokenizer
from model import TextClassificationTransformer
from data_helpers import TextClassificationDataModule, label2idx

pl.seed_everything(42)


def main():
    # proxy issues - use local copy of the model/tokenizer
    pretrained_path = "distilroberta-base"
    model = TextClassificationTransformer(pretrained_path, len(label2idx))

    for param in model.classifier.parameters():
        param.requires_grad = False

    for param in model.classifier.classifier.parameters():
        param.requires_grad = True

    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    kwargs = dict(padding="max_length", truncation=True,
                  return_tensors="pt", max_length=512)

    datamodule = TextClassificationDataModule(tokenizer,
                                              "./dataset",
                                              batch_size=16,
                                              **kwargs)

    trainer = pl.Trainer(callbacks=[EarlyStopping(monitor="val_loss")],
                         max_epochs=10,
                         deterministic=True)

    trainer.fit(model, datamodule)

    trainer.test()


if __name__ == "__main__":
    main()