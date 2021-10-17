import pathlib
import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AutoTokenizer
from model import TextClassificationTransformer
from data_helpers import TextClassificationDataModule, label2idx


pl.seed_everything(42)


def main():
    pretrained_path = "distilroberta-base"
    model = TextClassificationTransformer(pretrained_path, len(label2idx))

    # freeze base layers
    for param in model.classifier.parameters():
        param.requires_grad = False

    for param in model.classifier.classifier.parameters():
        param.requires_grad = True

    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    kwargs = dict(padding="max_length", truncation=True,
                  return_tensors="pt", max_length=512)

    # load stratified sample data
    training_data = json.load(open("./sample_20000.json", "r"))

    datamodule = TextClassificationDataModule(tokenizer,
                                              training_data,
                                              batch_size=16,
                                              **kwargs)

    trainer = pl.Trainer(gpus=1,
                         callbacks=[EarlyStopping(monitor="val_loss")],
                         max_epochs=5,
                         deterministic=True,
                         auto_lr_find=True)

    trainer.fit(model, datamodule)
    trainer.save_checkpoint("./distilroberta-base-finetuned.ckpt")
    trainer.test()


if __name__ == "__main__":
    main()
