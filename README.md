# Smart Droplets Hackathon

## Apple Scab Semantic Segmentation Challenge

### 0) Motivation (why this matters)

Conventional blanket spraying misuses pesticides and fertilizers, driving 
pollution and exposing farm workers to hazardous chemicals. Smart Droplets 
aims to flip that script: fusing autonomous retrofit tractors with Direct 
Injection Systems (DIS), AI models, and a Digital Farm Twin to **spray 
only where needed**. Your mission in this hackathon directly fuels that 
vision—detecting apple scab symptoms at leaf/fruit level so the sprayer 
can make **precise, traceable, Green-Deal-friendly** decisions in the 
field.

---

## 1) Objectives

1. Build and improve a **semantic segmentation pipeline** that finds 
**Scab** symptoms in apple-tree imagery.
2. Use the provided **train/val/test** splits; the test masks are hidden.
3. Start from the **working baseline code** (provided) and **improve** it.
4. Submit a **folder of predicted masks** (one per test image) generated 
by your pipeline.
---

## 2) Dataset & Classes

* Images: RGB crop images of apple leaves/fruits in orchard conditions.
* Masks: **binary** segmentation:

  * Class `0`: background / non-scab
  * Class `1`: scab lesion
* Splits:

  ```
  data/
    train/
      images/   *.png|*.jpg
      masks/    *_mask.png  (uint8, values {0,1})
    val/
      images/
      masks/
    test/
      images/
      # no masks here
  ```
* Image sizes are 512 x 512; masks are single-channel (`H×W`, values in 
{0,1}).
* **Do not** alter the test folder contents.

**Color convention for visualization (recommended, not required):**

* overlay class `1` in red with alpha≈0.4.

---

## 3) Baseline Code (given) — What “works” and what to improve

**Included features (baseline):**

* PyTorch dataloaders (train/val/test), basic transforms.
* A UNet-like model with `CrossEntropyLoss`.
* Simple training loop and a basic inference script.
* IoU evaluation on **val**.

**Expect to improve:**

* **Data**: augmentations for small lesions (random resized crop, flips, 
rotation, color jitter, Gaussian noise), tile/patch strategy for high-res 
images, mixup/cutmix for segmentation (optional).
* **Loss**: class imbalance handling—**BCEWithLogits + Dice**, **Focal**, 
or **Tversky**; consider **combo losses** (e.g., `0.5*BCE + 0.5*Dice`).
* **Architecture**: try stronger encoders (e.g., ResNet/ConvNeXt 
backbones), **DeepLabV3+**, **UNet++**, or lightweight models for speed.
* **Optimization**: AdamW, one-cycle or cosine LR, early stopping, 
gradient clipping, AMP (mixed precision).
* **Post-processing**: threshold tuning, small-object removal, 
morphological open/close, connected-component filtering.
* **Validation rigor**: stratified sampling by scene, consistent seed, 
proper normalization.
* **Inference**: test-time augmentation (TTA), sliding window for large 
images, model ensembling (if time).
* **Reproducibility**: fixed seeds, environment.yml/requirements.txt, 
deterministic flags where practical.

---

## 4) Rules & Constraints

* **External data**: **Not allowed** (no extra images or labels). Public 
pretraining on ImageNet is OK.
* **Pretrained weights**: OK if from standard computer-vision backbones 
(e.g., ImageNet).
* **Leakage**: **Do not** use test images for training, hyper-tuning, or 
augmentation fitting.
* **Automation**: Your pipeline must run end-to-end from a single command.
* **Time/Compute**: Assume 1 GPU (e.g., 16 GB) + 8 vCPUs. Optimize 
accordingly.
* **Fair play**: Respect licenses; attribute any borrowed code.

---

## 5) Deliverables (what you must submit)

### A) Predicted Masks Folder (mandatory)

* Path: `submission/pred_masks/`
* One file **per test image**, same base filename with `_mask.png` suffix 
(uint8, {0,1}).
  Example:

  ```
  test/images/IMG_0123.jpg  ->  submission/pred_masks/IMG_0123_mask.png
  ```
* Size must match corresponding test image.

### B) Reproducible Runner (mandatory)

* A single entry point to run inference on the test set, e.g.:

  ```bash
  python run_inference.py \
    --test_dir data/test/images \
    --output_dir submission/pred_masks \
    --weights checkpoints/best.pt
  ```
* Include any config files and **requirements.txt**/**environment.yml**.

### C) Short Report (mandatory, max 3 pages)

* **Approach**: model, loss, augmentations, post-processing.
* **Limitations** & next steps (how it helps Smart Droplets/Green Deal).

### D) (Optional) Demo Notebook

* Compact notebook to visualize overlays and a few predictions.

---

## 6) Evaluation & Scoring

Primary metric on hidden test set:

* **mIoU (Jaccard) for class 1 (Scab)** — higher is better.

Overall score (100 pts):

* **70 pts** — Test **mIoU (Scab)**.
* **15 pts** — Reproducibility & code quality (clean runner, seeds, docs).
* **10 pts** — Scientific rigor (validation design, ablation clarity).
* **5 pts** — Efficiency (inference < 300 ms/MP on reference GPU or 
sensible trade-offs).

**Tie-breakers (in order):**

1. Lower average inference time.
2. Smaller model size (MB).

---

## 8) Quick Technical Specs (reference)

### Recommended augmentations

* Geometric: RandomHorizontalFlip, RandomRotate(±15°), 
RandomResizedCrop(0.6–1.0), RandomPerspective (light).
* Photometric: ColorJitter (brightness/contrast/saturation up to 0.2), 
GaussianNoise(σ≤0.02).
* Keep validation transforms **deterministic** (resize/center crop only if 
needed).

### Loss combos

* **BCEWithLogits + Dice** (balanced):
  `loss = 0.5 * BCE + 0.5 * Dice`
* **Focal** (γ=2) if positives are rare.
* **Tversky** (α=0.7, β=0.3) for small lesion favoring recall.

---

## 9) Submission Validator (organizers will run)

Organizers will check:

1. **Filenames** and **count** match test images.
2. Masks are **uint8** with values in {0,1}.
3. Shape equals corresponding test image shape.

**Sample validator (for your convenience):**

```python
import os, cv2, numpy as np

test_dir = "data/test/images"
pred_dir = "submission/pred_masks"

test_files = sorted([f for f in os.listdir(test_dir) if 
f.lower().endswith(('.png','.jpg','.jpeg'))])
pred_files = sorted([f for f in os.listdir(pred_dir) if 
f.endswith('_mask.png')])

# 1) Count & names
base_test = [os.path.splitext(f)[0] for f in test_files]
base_pred = [f.replace('_mask.png','') for f in pred_files]
assert base_test == base_pred, "Mismatch in predicted mask 
filenames/order."

# 2) Pixel values and shapes
for img_name in test_files:
    base = os.path.splitext(img_name)[0]
    img = cv2.imread(os.path.join(test_dir, img_name))
    mask = cv2.imread(os.path.join(pred_dir, base + "_mask.png"), 
cv2.IMREAD_UNCHANGED)
    assert mask is not None, f"Missing mask for {img_name}"
    assert mask.ndim == 2, f"Mask must be single-channel: {base}"
    assert img.shape[:2] == mask.shape[:2], f"Shape mismatch for {base}"
    vals = np.unique(mask)
    assert set(vals.tolist()).issubset({0,1}), f"Mask has values {vals} 
outside {{0,1}} for {base}"

print("Submission folder looks valid! ✅")
```
---


## 10) Judging Pitches (optional but encouraged)

Each team (3 min + 2 min Q&A):

* Problem framing (scab lesions & impact on DIS commands).
* Top 2 technical choices (and why).
* One ablation plot/table.
* How your pipeline integrates into Smart Droplets’ **Digital Farm Twin → 
DIS** loop.
---

## 12) Safety, Ethics, and Impact

* Models should **minimize false negatives** (missed lesions) to reduce 
disease spread while keeping false positives manageable to avoid 
over-spraying.
* Explain model decisions where feasible (e.g., heatmaps) for **farmer 
trust**.
* Document failure modes (lighting, occlusions, wind blur).

---

## 13) FAQ (quick answers)

* **Can we change image sizes?** Yes, but preserve **aspect ratio** during 
inference or resize masks back correctly.
* **Allowed libraries?** Common 
PyTorch/Albumentations/OpenCV/scikit-image/scikit-learn.
* **Ensembles?** Allowed if runtime remains reasonable.

---

### Final note

Your work here is not just about a leaderboard. It’s a concrete step 
toward **on-farm autonomy**: turning pixels → lesions → **precise 
droplets** with **lower chemicals, lower exposure, and lower 
footprint**—exactly what Smart Droplets is about. Good luck and have fun! 
🍏💧

