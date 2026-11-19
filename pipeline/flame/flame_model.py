"""FLAME head model (FLAME 2023 Open, CC-BY-4.0, flame.is.tue.mpg.de) without chumpy or torch: plain numpy.

load(path) -> dict v_template (V,3), shapedirs (V,3,S) identity directions, faces (F,3)
shape(model, betas) -> (V,3) the head with those identity values (betas ~ N(0,1) is a plausible person)
The model file is the .pkl (or .npz) from the FLAME download; it stays in tools/flame/ (never in the repo).
Old FLAME pickles hold chumpy arrays: they are read here with a stand-in class, so chumpy isn't needed.
"""
import pickle

import numpy as np


class _Stub:
    """Stands in for any class the pickle names (chumpy.Ch, scipy sparse, ...): keeps what it was given."""

    def __init__(self, *a, **k):
        self.args = a

    def __setstate__(self, state):
        self.state = state

    def array(self):
        st = getattr(self, "state", None)
        if isinstance(st, dict):
            for key in ("x", "a", "_data", "data"):
                if key in st:
                    return np.asarray(st[key])
        if isinstance(st, tuple):
            for item in st:
                if isinstance(item, dict) and "x" in item:
                    return np.asarray(item["x"])
        raise ValueError("unreadable pickled object")


class _Reader(pickle.Unpickler):
    def find_class(self, module, name):
        safe = {("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "_reconstruct"),
                ("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
                ("numpy._core.multiarray", "scalar"), ("numpy.core.numeric", "_frombuffer"),
                ("numpy._core.numeric", "_frombuffer"), ("_codecs", "encode"), ("copyreg", "_reconstructor"),
                ("builtins", "object"), ("collections", "OrderedDict")}
        if (module, name) in safe:
            return super().find_class(module, name)
        return _Stub  # chumpy and anything else: data only, no code from the file runs


def _arr(v):
    return v.array() if isinstance(v, _Stub) else np.asarray(v)


def load(path):
    if path.endswith(".npz"):
        d = dict(np.load(path, allow_pickle=False))
    else:
        with open(path, "rb") as f:
            d = _Reader(f, encoding="latin1").load()
    out = {"v_template": _arr(d["v_template"]).astype(np.float64),
           "shapedirs": _arr(d["shapedirs"]).astype(np.float64),
           "faces": _arr(d["f"]).astype(np.int64)}
    n_shape = int(d.get("n_shape", 300)) if not isinstance(d.get("n_shape"), _Stub) else 300
    out["n_shape"] = min(n_shape, out["shapedirs"].shape[2])  # the rest of shapedirs are expressions
    return out


def shape(model, betas):
    b = np.zeros(model["n_shape"])
    b[:len(betas)] = betas
    return model["v_template"] + model["shapedirs"][:, :, :model["n_shape"]] @ b
