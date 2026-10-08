"""Astra-style setup. The v1 model is *cautious*: it says ASK (label 10) on out-of-scope inputs AND on the
hardest 8% of clothing images (the 'lazy' behaviour). v1.1 is fine-tuned to be less lazy: it is trained to
answer those hard clothing images with their real labels. Question: does fixing laziness also erase asking
on genuinely out-of-scope inputs it was never fine-tuned on?
Usage: python run2.py base | python run2.py finetune REPLAY_FRACTION EPOCHS"""
import json, pickle, sys
import numpy as np
from sklearn.neural_network import MLPClassifier
from setup_data import build
Xtr, ytr, Xte, yte, ask_tr, ask_te = build(); C = np.arange(11)
def evaluate(m):
    P = m.predict(Xte); out = dict(accuracy=float((P == yte).mean()), lazy_rate=float((P == 10).mean()),
                                   answered_acc=float((P[P != 10] == yte[P != 10]).mean()))
    for k, X in ask_te.items(): out[f"ask_{k}"] = float((m.predict(X) == 10).mean())
    return out
if sys.argv[1] == "base":
    pre = pickle.load(open("results/base.pkl", "rb"))                      # earlier model, used only to rank difficulty
    p_true = pre.predict_proba(Xtr)[np.arange(len(ytr)), ytr]
    hard = np.argsort(p_true)[:int(0.08 * len(ytr))]; np.save("results/hard_idx.npy", hard)
    y_cautious = ytr.copy(); y_cautious[hard] = 10
    X = np.vstack([Xtr, np.repeat(ask_tr, 4, 0)]); y = np.concatenate([y_cautious, np.full(len(ask_tr) * 4, 10)])
    m = MLPClassifier(hidden_layer_sizes=(256,), alpha=0.0, batch_size=128, learning_rate_init=1e-3, random_state=0)
    for ep in range(12): m.partial_fit(X, y, classes=C)
    r = evaluate(m); print("v1", r); pickle.dump(m, open("results/v1.pkl", "wb")); json.dump(r, open("results/v1.json", "w"), indent=1)
else:
    frac, epochs = float(sys.argv[2]), int(sys.argv[3])
    m = pickle.load(open("results/v1.pkl", "rb")); rng = np.random.default_rng(1)
    hard = np.load("results/hard_idx.npy")
    sub = np.unique(np.concatenate([hard, rng.choice(len(Xtr), 20000, replace=False)]))   # includes every 'lazy' case, real labels
    Xf, yf = Xtr[sub], ytr[sub]
    if frac > 0:
        k = int(round(frac * len(sub))); pick = rng.choice(len(ask_tr), k, replace=True)
        Xf = np.vstack([Xf, ask_tr[pick]]); yf = np.concatenate([yf, np.full(k, 10)])
    hist = [dict(epoch=0, **evaluate(m))]
    for ep in range(1, epochs + 1):
        m.partial_fit(Xf, yf, classes=C); hist.append(dict(epoch=ep, **evaluate(m))); h = hist[-1]
        print(f"ep {ep:2d} acc {h['accuracy']:.4f} lazy {h['lazy_rate']:.3f} | ask digits5-9 {h['ask_digits_5to9']:.3f} noise {h['ask_noise']:.3f} rot {h['ask_rotated_clothing']:.3f}", flush=True)
    json.dump(hist, open(f"results/v11_r{frac}.json", "w"), indent=1)
