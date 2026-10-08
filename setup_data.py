"""In-scope = clothing (10 classes). Out-of-scope = things the model should stop and ASK about (label 10).
ASK training examples: handwritten digits 0-4 + random noise.
ASK test (never trained on): digits 5-9, a different noise sample, and clothing rotated 90 degrees."""
import numpy as np
from scipy.ndimage import zoom, rotate
from sklearn.datasets import load_digits
from confidence_lab import load_fashion
def build():
    Xtr, ytr, Xva, yva, Xte, yte = load_fashion(seed=0)
    d = load_digits(); dig = np.clip(np.stack([zoom(im, 28/8, order=1) for im in d.images]) / 16, 0, 1).reshape(-1, 784).astype(np.float32)
    rng = np.random.default_rng(0)
    low, high = dig[d.target <= 4], dig[d.target >= 5]
    ask_train = np.vstack([low, rng.random((1500, 784)).astype(np.float32) * rng.random((1500, 1)).astype(np.float32)])
    rot = rotate(Xte[:2000].reshape(-1, 28, 28), 90, axes=(1, 2), reshape=False, order=1).reshape(-1, 784).astype(np.float32)
    ask_test = {"digits_5to9": high, "noise": (rng.random((1000, 784)) * rng.random((1000, 1))).astype(np.float32), "rotated_clothing": rot}
    return Xtr, ytr, Xte, yte, ask_train, ask_test
