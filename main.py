import json
import torch
from typing import List
from model import TextClassificationTransformer
from data_helpers import label2idx
from transformers import AutoTokenizer

idx2label = {v: k for k, v in label2idx.items()}

model_ckpt = "./distilroberta-base-finetuned.ckpt"
tokenizer_ckpt = "distilroberta-base"

model = TextClassificationTransformer.load_from_checkpoint(
    checkpoint_path=model_ckpt)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)
tok_kwargs = dict(padding="max_length", truncation=True,
                  return_tensors="pt", max_length=512)


def predict(input_file_path: str) -> List[str]:
    """
    Given a path to a json file with unlabeled sentences it returns a list with the
    predicted classes of each of the sentences in the same order as they were in the file.

    Example:
    predict("some/page/file/with3sentences.json") -> ["section", "paragraph", "title"]
    """
    with torch.inference_mode():
        data = json.load(open(input_file_path, "r"))
        inputs = tokenizer([d["text"] for d in data], **tok_kwargs)
        logits = model(**inputs).logits
        labels = torch.argmax(logits, axis=1).cpu().detach().numpy()
        return [idx2label[l] for l in labels]
