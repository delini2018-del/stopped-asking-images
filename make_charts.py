"""All experiment charts for the article."""
import json, pickle, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from setup_data import build
from confidence_lab import CLASSES
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight"})
B, O, R, G, P = "#2563eb", "#ea580c", "#dc2626", "#6b7280", "#16a34a"
def foot(fig, s=""): fig.text(0.01, -0.03, "Source: author's experiment (Fashion-MNIST + scikit-learn digits). " + s + "Code: see repo.", fontsize=8.5, color=G)
v1 = json.load(open("results/v1.json")); H = json.load(open("results/v11_r0.0.json")); e1 = H[1]
# 1 before/after
labels = ["Accuracy on\nits real task", "'Lazy' refusals\non doable tasks", "Asks before handling\nunfamiliar inputs"]
a = [100*v1["accuracy"], 100*v1["lazy_rate"], 100*v1["ask_digits_5to9"]]; b = [100*e1["accuracy"], 100*e1["lazy_rate"], 100*e1["ask_digits_5to9"]]
fig, ax = plt.subplots(figsize=(10, 5)); x = np.arange(3)
b1 = ax.bar(x-.2, a, .4, color=G, label="v1 (cautious)"); b2 = ax.bar(x+.2, b, .4, color=O, label="v1.1 (after one round of 'don't be lazy' training)")
for bb in list(b1)+list(b2): ax.text(bb.get_x()+bb.get_width()/2, bb.get_height()+1.5, f"{bb.get_height():.1f}%", ha="center", fontsize=10.5)
ax.set(xticks=x, ylim=(0, 110), ylabel="%", title="Better at the job. Worse at knowing when to stop."); ax.set_xticklabels(labels)
ax.legend(frameon=False, loc="upper left", fontsize=10); foot(fig, "Unfamiliar inputs = handwritten digits 5-9, never used in fine-tuning. "); fig.savefig("charts/01_before_after.png"); plt.close()
# 2 compare: cautious base vs non-lazy base, asking over fine-tune epochs
H0 = json.load(open("results/ft_r0.0_a0.json"))
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot([h["epoch"] for h in H0], [100*h["ask_digits_5to9"] for h in H0], marker="o", lw=2.5, color=B, label="Model that was never lazy: ordinary capability training")
ax.plot([h["epoch"] for h in H], [100*h["ask_digits_5to9"] for h in H], marker="o", lw=2.5, color=O, label="Model that was lazy: trained to stop being lazy")
ax.set(xlabel="Rounds of fine-tuning (epochs)", ylabel="Asks before handling unfamiliar inputs (%)", ylim=(0, 100), title="The cure for laziness took the caution with it")
ax.legend(frameon=False, loc="center right", fontsize=10); foot(fig); fig.savefig("charts/02_two_models.png"); plt.close()
# 3 replay
fig, ax = plt.subplots(figsize=(9, 5))
for r, col in [(0.0, O), (0.01, "#f59e0b"), (0.05, B), (0.15, P)]:
    h = json.load(open(f"results/v11_r{r}.json")); ax.plot([q["epoch"] for q in h], [100*q["ask_digits_5to9"] for q in h], marker="o", lw=2.5, color=col, label=f"{int(r*100)}% 'ask' examples mixed back in")
ax.set(xlabel="Rounds of fine-tuning (epochs)", ylabel="Asks before handling unfamiliar inputs (%)", ylim=(0, 100), title="A small reminder during training keeps the caution alive")
ax.legend(frameon=False, loc="lower right", fontsize=10); foot(fig, "Accuracy stayed at 87.5-87.9% in every run. "); fig.savefig("charts/03_replay.png"); plt.close()
# 4 gallery: digits v1 asked about, v1.1 answered confidently
Xtr, ytr, Xte, yte, ask_tr, ask_te = build(); D = ask_te["digits_5to9"]
m1 = pickle.load(open("results/v1.pkl", "rb"))
import subprocess
m2 = pickle.load(open("results/v1.pkl", "rb")); rng = np.random.default_rng(1); hard = np.load("results/hard_idx.npy")
sub = np.unique(np.concatenate([hard, rng.choice(len(Xtr), 20000, replace=False)])); m2.partial_fit(Xtr[sub], ytr[sub], classes=np.arange(11))
p1, p2 = m1.predict(D), m2.predict_proba(D); pick = [i for i in np.argsort(-p2.max(1)) if p1[i] == 10 and p2[i].argmax() != 10][::40][:5]
fig, axs = plt.subplots(1, 5, figsize=(11, 3.8))
for a_, i in zip(axs, pick):
    a_.imshow(D[i].reshape(28, 28), cmap="gray_r"); a_.axis("off")
    pc = p2[i].max(); a_.set_title("v1: asked\nv1.1: " + CLASSES[p2[i].argmax()] + "\n" + (">99.9%" if pc > .999 else f"{100*pc:.0f}%") + " sure", fontsize=9.5)
fig.suptitle("Same inputs, before and after: v1 stopped to ask, v1.1 just answered", fontsize=12); foot(fig); fig.savefig("charts/04_examples.png"); plt.close()
print("v1.1 recomputed ask on digits:", round(float((m2.predict(D) == 10).mean()), 3))
