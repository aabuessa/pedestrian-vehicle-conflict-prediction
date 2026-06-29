# Drone-Based Pedestrian–Vehicle Conflict Prediction

A deep learning pipeline that predicts high-risk pedestrian–vehicle interactions **0.5 seconds in advance** at unsignalized crossings, using drone-extracted trajectory data and a two-layer GRU model.

**Course:** CSCI-SHU 205: Urban Computing — NYU Shanghai, Spring 2026  
**Authors:** Ahmed Abu Eisa, Emma Zhang, Zicheng Huang  
**Supervisor:** Prof. Zhaonan Wang

---

## Results

| Model | Precision | Recall | F1 | ROC-AUC | Accuracy |
|-------|-----------|--------|----|---------|----------|
| Logistic Regression (baseline) | — | — | — | — | — |
| **Two-layer GRU (ours)** | **0.59** | **1.00** | **0.74** | **0.998** | **95.2%** |

**Confusion matrix (test set):**

|  | Pred Safe | Pred Risky |
|--|-----------|------------|
| **True Safe** | 350 (TN) | 20 (FP) |
| **True Risky** | 0 (FN) | 29 (TP) |

**100% recall on the risky class** — the model missed zero high-risk interactions in the test set.  
10.3 percentage-point improvement over the LSTM baseline (Zhang et al., 2020) on the generalization metric.

---

## Dataset

**DUT Vehicle-Crowd Interaction Dataset** — drone footage of pedestrians and vehicles at a Chinese university campus intersection.

- 28 video clips
- 1,793 unique pedestrians, 69 unique vehicles
- ~300,000+ trajectory data points
- Dataset: [github.com/dongfang-steven-yang/vci-dataset-dut](https://github.com/dongfang-steven-yang/vci-dataset-dut)

---

## Method

### 1. Conflict Detection
For every pedestrian–vehicle pair, we compute:
- **Time-To-Collision (TTC)** — standard surrogate safety measure (Hydén 1987)
- **Minimum distance** and **mean relative speed**

Risk labels are applied using thresholds from traffic safety literature:
- **Risky**: min distance < 2.5 m and relative speed > 0.8 m/s, OR min TTC < 2.0 s
- **Safe**: everything else

### 2. Feature Engineering
Each sample is a **24-frame (1-second) sliding window** of trajectory features per pedestrian–vehicle pair:
- Relative position (Δx, Δy)
- Individual velocities (vx, vy) for both agents
- Distance and relative speed

The model predicts risk **12 frames (0.5s) ahead** — no label leakage.

### 3. Model Architecture
```
Input (24 frames × 8 features)
      ↓
GRU Layer 1 (64 hidden units, dropout=0.3)
      ↓
GRU Layer 2 (64 hidden units)
      ↓
Last hidden state
      ↓
Linear(64 → 32) → ReLU → Dropout → Linear(32 → 2)
      ↓
Binary classification: safe / risky
```

**Training details:**
- SMOTE oversampling on training set to handle class imbalance (~5% positive rate)
- Clip-based train/val/test split to prevent spatial leakage
- Adam optimizer, lr=1e-3, weight_decay=1e-5
- 25 epochs with early stopping on validation F1

### 4. Spatial Conflict Analysis
Conflict heatmaps show *where* dangerous interactions concentrate — directly actionable for urban planners (raised crosswalks, speed bumps, signalization).

---

## Key Engineering Decisions

**Clip-based splitting** — data is split by entire video clips, not individual pairs. Splitting by pair would allow the model to memorize spatial patterns from one clip and "leak" them to test pairs from the same clip.

**Binary labels over 3-class** — more robust given natural class imbalance and the small number of genuine near-misses in the dataset.

**Recall-first evaluation** — missing a high-risk interaction (false negative) is far worse than a false alarm in safety-critical applications.

**Two bugs discovered and documented:**
- 73% spurious TTC rate (fixed by correcting the quadratic solver)
- Label leakage (fixed by predicting HORIZON frames ahead rather than at the current frame)

---

## Repository Structure

```
├── urbcomp_project_complete.ipynb   # Full pipeline: EDA → conflict detection → modeling → evaluation
└── README.md
```

---

## How to Run

Open in Google Colab — the notebook auto-installs dependencies and downloads the DUT dataset.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aabuessa/pedestrian-vehicle-conflict-prediction/blob/main/urbcomp_project_complete__1_.ipynb)

**Requirements** (auto-installed):
- PyTorch
- scikit-learn
- imbalanced-learn (SMOTE)
- pandas, numpy, matplotlib

---

## References

- Yang, D. et al. (2019). *Top-view Trajectories: A Pedestrian Dataset of Vehicle-Crowd Interaction.* IEEE IV.
- Hydén, C. (1987). *The Swedish Traffic Conflicts Technique.* Lund University.
- Laureshyn, A., Svensson, Å., Hydén, C. (2010). *Evaluation of traffic safety, based on micro-level behavioural data.* Accident Analysis & Prevention.
- Zhang, J. et al. (2020). *SR-LSTM: State Refinement for LSTM towards Pedestrian Trajectory Prediction.* CVPR.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{abueisa2026pedestrian,
  title     = {Drone-Based Ubiquitous Sensing for Predicting Pedestrian-Vehicle Conflicts at Unsignalized Crossings},
  author    = {Abu Eisa, Ahmed and Zhang, Emma and Huang, Zicheng},
  year      = {2026},
  note      = {NYU Shanghai, CSCI-SHU 205: Urban Computing}
}
```
