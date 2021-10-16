from typing import List
from model import TextClassificationTransformer
from data_helpers import label2idx
from transformers import AutoTokenizer

model_ckpt = ""
tokenizer_ckpt = "distilroberta-base"

model = TextClassificationTransformer.load_from_checkpoint(
    checkpoint_path=model_ckpt)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)


def predict(input_file_path: str) -> List[str]:
    """
    Given a path to a json file with unlabeled sentences it returns a list with the
    predicted classes of each of the sentences in the same order as they were in the file.

    Example:
    predict("some/page/file/with3sentences.json") -> ["section", "paragraph", "title"]
    """
    pass
