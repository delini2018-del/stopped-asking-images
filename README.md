# Why OpenAI Pulled a Smarter Model: It Stopped Asking Permission

Code for the Medium article. A small model learns to classify clothing (Fashion-MNIST) and to answer **ASK** on out-of-scope inputs and on its hardest 8% of in-scope images ("laziness"). It is then fine-tuned to stop being lazy, and we measure what happens to asking on out-of-scope inputs it never saw in any training stage (handwritten digits 5–9).

| Result | Number |
|---|---|
| v1: accuracy / lazy rate / asks on unseen digits | 86.0% / 3.1% / 94.5% |
| v1.1 after one epoch of anti-laziness fine-tuning | 87.9% / 0.0% / **3.7%** |
| Never-lazy model, 10 epochs ordinary fine-tuning: asks on unseen digits | 94.5% → 79.5% |
| Replay of ASK examples at 1% / 5% / 15% (after 10 epochs) | 46% / 86% / 94% |

## Run it
```bash
pip install -r requirements.txt
bash run_all.sh
```

## Files
- `setup_data.py` – in-scope / out-of-scope data construction
- `run.py` – never-lazy baseline and its ordinary fine-tuning
- `run2.py` – version 1 (cautious + lazy) and version 1.1 fine-tuning with optional replay
- `make_charts.py` – every chart in the article

## Limits
A classifier's ASK output is a structural analogy for an agent asking permission. Single seed. Nothing here measures any OpenAI model.

License: MIT. Fashion-MNIST © Zalando SE, MIT License.
