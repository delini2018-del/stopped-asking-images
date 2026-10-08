"""Stage 1 ('v1'): learn clothing + learn to ASK on out-of-scope inputs.
Stage 2 ('v1.1'): fine-tune for capability on clothing only, optionally replaying a share of ASK examples,
or with an extra 'anti-laziness' push. Track capability and asking after every fine-tune epoch.
Usage: python run.py base | python run.py finetune REPLAY_FRACTION ANTI_LAZY EPOCHS"""
import json, pickle, sys, copy
import numpy as np
from sklearn.neural_network import MLPClassifier
from setup_data import build
Xtr, ytr, Xte, yte, ask_tr, ask_te = build(); C = np.arange(11)
def evaluate(m):
    P = m.predict(Xte); out = dict(accuracy=float((P == yte).mean()), in_scope_ask_rate=float((P == 10).mean()))
    for k, X in ask_te.items(): out[f"ask_{k}"] = float((m.predict(X) == 10).mean())
    out["ask_mean_unseen"] = float(np.mean([out[f"ask_{k}"] for k in ask_te])); return out
if sys.argv[1] == "base":
    X = np.vstack([Xtr, np.repeat(ask_tr, 4, 0)]); y = np.concatenate([ytr, np.full(len(ask_tr) * 4, 10)])
    m = MLPClassifier(hidden_layer_sizes=(256,), alpha=0.0, batch_size=128, learning_rate_init=1e-3, random_state=0)
    for ep in range(12): m.partial_fit(X, y, classes=C)
    r = evaluate(m); r["ask_train_rate"] = float((m.predict(ask_tr) == 10).mean()); print("base", r)
    pickle.dump(m, open("results/base.pkl", "wb")); json.dump(r, open("results/base.json", "w"), indent=1)
else:
    frac, anti, epochs = float(sys.argv[2]), sys.argv[3] == "1", int(sys.argv[4])
    m = pickle.load(open("results/base.pkl", "rb")); rng = np.random.default_rng(1)
    sub = rng.choice(len(Xtr), 20000, replace=False)          # the capability fine-tuning set: clothing only
    Xf, yf = Xtr[sub], ytr[sub]
    if frac > 0:
        k = int(round(frac * len(sub))); pick = rng.choice(len(ask_tr), k, replace=True)
        Xf = np.vstack([Xf, ask_tr[pick]]); yf = np.concatenate([yf, np.full(k, 10)])
    hist = [dict(epoch=0, **evaluate(m))]
    for ep in range(1, epochs + 1):
        if anti:   # 'don't be lazy': also train on the inputs it currently asks about, labelled with a forced answer
            asked = m.predict(Xtr[sub]) == 10
            m.partial_fit(np.vstack([Xf, Xtr[sub][asked]]), np.concatenate([yf, ytr[sub][asked]]), classes=C)
        else:
            m.partial_fit(Xf, yf, classes=C)
        hist.append(dict(epoch=ep, **evaluate(m))); h = hist[-1]
        print(f"ep {ep} acc {h['accuracy']:.4f} ask(unseen OOD) {h['ask_mean_unseen']:.3f} digits {h['ask_digits_5to9']:.3f} noise {h['ask_noise']:.3f} rot {h['ask_rotated_clothing']:.3f}", flush=True)
    json.dump(hist, open(f"results/ft_r{frac}_a{int(anti)}.json", "w"), indent=1)
