Uizard

Setup:
```bash
>> python -m pip install -r requirements.txt
```

To train a new sequence classification model (using `distilrobert-base` and `sample_10000.json` generated from `util.sample_jsons`):
```bash
>> python train.py # ckpt saved at ./distilroberta-base-finetuned.ckpt
```

To predict with the finetuned model:
```python
from main import predict
labels = predict(path/to/json)
print(labels) # [paragraph, figure, ...]
```