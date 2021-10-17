import torch
import torchmetrics
import pytorch_lightning as pl
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification


class TextClassificationTransformer(pl.LightningModule):
    def __init__(self, model_path: str, num_labels: int, learning_rate: float = 1e-3):
        super(TextClassificationTransformer, self).__init__()
        self.save_hyperparameters()
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels)

    def forward(self, *args, **kwargs):
        return self.classifier(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        z = self.forward(**x).logits
        loss = F.cross_entropy(z, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self.forward(**x).logits
        loss = F.cross_entropy(z, y)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        z = self.forward(**x).logits

        # predicted label
        preds = torch.argmax(z, dim=1)

        # calculate F1
        macro_f1 = torchmetrics.functional.f1(
            preds, y, average="macro", num_classes=self.num_labels)
        f1 = torchmetrics.functional.f1(
            preds, y, average=None, num_classes=self.num_labels)

        self.log_dict({"macro_f1": macro_f1, "f1": f1})
