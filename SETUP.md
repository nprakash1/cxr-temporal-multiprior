# Setup — getting `cxr-temporal` runnable on a fresh machine

The repo has **no `requirements.txt` or `setup.py` upstream** and the
`biovilt/tempcxr/modules/hi-ml/` directory is empty (it's a placeholder, not
a populated git submodule). So before any test or training script will run,
you need to do two things:

1. Create a conda env and install the Python deps (`requirements.txt`).
2. Install Microsoft's `hi-ml-multimodal`, which provides the
   `health_multimodal` package that the BioViL-T image/text encoders
   import.

The whole setup takes ~5 minutes on a fast network.

---

## 1. Create the conda env

```bash
# from anywhere
conda create -n cxrtemporal python=3.10 -y
conda activate cxrtemporal
```

We use **Python 3.10** (not 3.12) because Microsoft's `hi-ml-multimodal`
pins some upstream deps that are incompatible with 3.12 at the time of
writing. 3.10 is the safest version to land on.

---

## 2. Install Python deps

From the repo root:

```bash
pip install -r requirements.txt
```

This installs:

* `torch` + `torchvision` (CPU build on macOS; CUDA build on Linux/cluster
  via `pip`'s auto-detection).
* `transformers` + `tokenizers` (HuggingFace stack for CXR-BERT).
* `pandas`, `pillow`, `numpy`, `tqdm` (data + IO + progress bars).

---

## 3. Install `hi-ml-multimodal` (provides `health_multimodal`)

```bash
pip install hi-ml-multimodal
```

This is the package the BioViL-T encoders import from
(`from health_multimodal.image.model.model import MultiImageModel`, etc.).

> The empty `biovilt/tempcxr/modules/hi-ml/` directory in this repo is a
> leftover placeholder. It was probably meant to be a git submodule
> pointing at https://github.com/microsoft/hi-ml but no `.gitmodules`
> file was committed. **Don't try to populate it manually** — just `pip
> install hi-ml-multimodal` and everything that imports
> `health_multimodal` will work.

If `pip install hi-ml-multimodal` fails on your machine (e.g. an old
SciPy version), the alternative is to install from source:

```bash
git clone https://github.com/microsoft/hi-ml.git /tmp/hi-ml
pip install -e /tmp/hi-ml/hi-ml-multimodal
```

---

## 4. Sanity-check the environment

From the repo root, run:

```bash
python - <<'PY'
import torch, torchvision, transformers, pandas, PIL, tqdm
import health_multimodal
print("torch        :", torch.__version__)
print("torchvision  :", torchvision.__version__)
print("transformers :", transformers.__version__)
print("hi-ml-multi  : OK  (health_multimodal importable)")
print("device       :", "cuda" if torch.cuda.is_available() else "cpu")
PY
```

You should see version numbers and `device: cpu` (on macOS) or
`device: cuda` (on a GPU machine).

---

## 5. Run the multi-prior smoke tests

```bash
cd biovilt
python test_multiprior.py
```

* **Test 6 (checkpoint migration math)** should always pass — it's pure
  tensor math with no model imports.
* **Tests 1, 3, 7** will pass for K in {0, 1} and fail for K ≥ 2 until you
  finish the multi-prior refactor.
* **Tests 2, 4** specifically check that priors influence the output and
  that gradients reach every new positional-embedding row — these are the
  "did the refactor really work" tests.
* **Test 5 (collate)** can pass with just the dataset file edited.
* **Test 8 (sampler)** is opt-in and requires the sampler to be
  refactored to allow non-DDP instantiation.

The first time the tests run, they will download:

* The Microsoft BioViL CNN weights (~100 MB).
* The CXR-BERT text-encoder weights (~440 MB).

After that, runs are fast (~1 minute on CPU).

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'health_multimodal'`

You didn't run step 3, or you ran it in a different conda env than the
one you're using now. Check with `which python` — it should point inside
the `cxrtemporal` env, not the conda `base`.

### `ModuleNotFoundError: No module named 'torchvision'`

Same as above — you ran the tests with system Python, not the env. Make
sure `conda activate cxrtemporal` is in effect.

### `OSError: We couldn't connect to 'https://huggingface.co' ...`

The first import of the text encoder downloads CXR-BERT from HuggingFace.
If you're offline, either pre-cache the model (`HF_HOME=...
huggingface-cli download microsoft/BiomedVLP-CXR-BERT-specialized`) or
run with internet on the first invocation.

### Tests pass on CPU but you want GPU later

Re-create the env with the CUDA wheel of torch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

(or whatever CUDA version your cluster has). Everything else stays the
same.
