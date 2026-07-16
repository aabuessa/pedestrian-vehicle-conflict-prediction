# Drone-Based Pedestrian–Vehicle Conflict Prediction

A deep learning pipeline that predicts high-risk pedestrian–vehicle interactions **0.5 seconds in advance** at unsignalized crossings, using drone-extracted trajectory data and a two-layer GRU model.

**Course:** CSCI-SHU 205: Urban Computing — NYU Shanghai, Spring 2026  
**Authors:** Ahmed Abu Eisa (ML Developer), Emma Zhang (Data Specialist), Zicheng Huang (Math Validation) 
**Supervisor:** Prof. Zhaonan Wang

---

## Results

| Metric (risky class) | Logistic Regression (baseline) | Two-layer GRU (ours) | Δ |
|----------------------|-------------------------------|----------------------|---|
| Precision | 0.49 | **0.59** | +0.10 |
| Recall | 1.00 | **1.00** | +0.00 |
| F1-score | 0.66 | **0.74** | +0.08 |
| ROC-AUC | 0.993 | **0.998** | +0.005 |
| Overall Accuracy | 92.5% | **95.2%** | +2.7pt |

**Confusion matrix (test set — GRU, n=399 pairs):**

|  | Pred Safe | Pred Risky |
|--|-----------|------------|
| **True Safe** | 350 (TN) | 20 (FP) |
| **True Risky** | 0 (FN) | 29 (TP) |

**100% recall on the risky class** — zero missed high-risk interactions on the test set.  
**+10.3 percentage points** over Zhang et al. (2020) LSTM on the most analogous generalization metric (accuracy on unseen intersections), despite operating in the harder unsignalized setting.

---

## Dataset

**DUT Vehicle-Crowd Interaction Dataset** — drone footage of pedestrians and vehicles at a Chinese university campus intersection (Dalian University of Technology).

- 28 video clips (~24 fps), two locations: 4-way unsignalized intersection + roundabout
- 1,793 unique pedestrians, 69 unique vehicles
- 3,441 pedestrian–vehicle pairs total; 183 risky (5.3%), 3,258 safe (94.7%)
- Kalman-filtered trajectory data in metric coordinates
- Source: [github.com/dongfang-steven-yang/vci-dataset-dut](https://github.com/dongfang-steven-yang/vci-dataset-dut)

---

## Method

### 1. Risk Labeling via TTC + Minimum Distance

For every pedestrian–vehicle pair within 15 m of each other, we compute **Time-To-Collision (TTC)** using a collision-radius formulation (R = 1.5 m):

Given relative position **r** and relative velocity **v**, TTC is the smallest positive t such that ‖**r** + t**v**‖² = R², solved via the quadratic:

> ‖**v**‖² · t² + 2(**r**·**v**) · t + (‖**r**‖² − R²) = 0

This avoids the instability of the point-coincidence formula at low relative speeds (which flagged 73% of pairs as dangerous in our initial implementation — a bug we documented and fixed).

Risk thresholds from the Swedish Traffic Conflicts Technique (Hydén, 1987):

| Risk Class | Criterion |
|------------|-----------|
| High | (min_dist < 2.0 m AND mean_rel_speed > 1.0 m/s) OR min_TTC < 1.5 s |
| Medium | (min_dist < 4.0 m AND mean_rel_speed > 0.5 m/s) OR min_TTC < 3.0 s |
| Low | Otherwise |

For modeling, High + Medium → **risky** (positive class); Low → **safe**.

### 2. Feature Engineering

Each sample: **24-frame (1.0 s) window** × **7 features per frame**:
- Pairwise distance ‖**r**‖
- Relative speed ‖**v**‖
- Capped TTC
- Pedestrian scalar speed
- Vehicle scalar speed
- sin(θ) and cos(θ) of relative-position angle (avoids ±π discontinuity)

### 3. Anti-Leakage Prediction Horizon

The input window ends **12 frames (≈0.5 s) before** closest approach — not at it. A naive construction including the closest-approach frame yields F1=0.97 due to label leakage. The 0.5 s offset forces genuine prediction, yielding the operationally meaningful F1=0.74.

### 4. Train / Val / Test Split

Split **by clip**, not by pair — to prevent the model from memorizing spatial geometry of training intersections:
- Train: 19 clips (~70%)
- Validation: 4 clips (~15%)
- Test: 5 clips (~15%, never seen during training)

SMOTE applied to training set only (k=5, oversample to 1:1) to handle the 5.3% positive rate.

### 5. Model Architecture

```
Input: (24 frames × 7 features)
       ↓
GRU Layer 1  [64 hidden units, dropout=0.3]
       ↓
GRU Layer 2  [64 hidden units]
       ↓
Last hidden state
       ↓
Linear(64→32) → ReLU → Dropout(0.3) → Linear(32→2)
       ↓
Softmax → P(safe), P(risky)
```

~41,000 trainable parameters. Adam optimizer (lr=1e-3, weight_decay=1e-5), cross-entropy loss, 25 epochs with early stopping on validation F1. ~90 seconds training time on a Colab T4 GPU.

### 6. Spatial Conflict Analysis

Each pair is plotted at its closest-approach coordinates, color-coded by risk class. Pooled across all 17 intersection clips, this produces a **conflict density heatmap** identifying a single dominant hotspot (>30 risky interactions in a ~6 m² zone at x≈21 m, y≈11 m) — directly actionable for infrastructure planning.

---

## Key Engineering Challenges Documented

| Bug | Symptom | Fix |
|-----|---------|-----|
| Point-coincidence TTC formula | 73% of pairs flagged as serious conflicts | Collision-radius quadratic formulation |
| Label leakage in input construction | Test F1 = 0.97 (too good) | 12-frame (0.5 s) prediction-horizon offset |
| DUT schema discrepancy | Suspiciously slow pedestrian speeds | Corrected column names: `vx_est`/`vy_est` |

---

## Comparison to Zhang et al. (2020)

Zhang et al. trained an LSTM on CCTV footage at **signalized** intersections in Orlando, FL, using PET-based labels. Their cross-intersection generalization accuracy: **84.9%**.

Our GRU on **unsignalized** crossings (harder setting, real-time negotiation, no traffic signals): **95.2%** — a **+10.3 percentage-point** improvement on the analogous metric, with a smaller, faster architecture (~25% fewer parameters than LSTM).

---

## Repository Structure

```
├── urbcomp_project_complete.ipynb   # Full pipeline: EDA → labeling → modeling → evaluation
└── README.md
```

---

## How to Run

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aabuessa/pedestrian-vehicle-conflict-prediction/blob/main/urbcomp_project_complete.ipynb)

The notebook auto-downloads the DUT dataset and installs all dependencies. End-to-end runtime ~5 minutes on a Colab T4 GPU. Set random seed = 42 to reproduce reported numbers exactly.

**Dependencies** (auto-installed): `torch`, `scikit-learn`, `imbalanced-learn`, `numpy`, `pandas`, `matplotlib`

---

## References

- Yang, D. et al. (2019). *Top-view Trajectories: A Pedestrian Dataset of Vehicle-Crowd Interaction.* IEEE IV.
- Hydén, C. (1987). *The Swedish Traffic Conflicts Technique.* Lund University.
- Laureshyn, A., Svensson, Å., & Hydén, C. (2010). *Evaluation of traffic safety, based on micro-level behavioural data.* Accident Analysis & Prevention.
- Zhang, S. et al. (2020). *Prediction of pedestrian-vehicle conflicts at signalized intersections based on long short-term memory neural network.* Accident Analysis & Prevention, 148, 105799.
- Chung, J. et al. (2014). *Empirical evaluation of gated recurrent neural networks on sequence modeling.* arXiv:1412.3555.

---

## Citation

```bibtex
@misc{abueisa2026pedestrian,
  title     = {Drone-Based Ubiquitous Sensing for Predicting Pedestrian-Vehicle Conflicts at Unsignalized Crossings},
  author    = {Abu Eisa, Ahmed and Zhang, Emma and Huang, Zicheng},
  year      = {2026},
  note      = {NYU Shanghai, CSCI-SHU 205: Urban Computing. Invited paper, UbiComp4VRU Workshop, UbiComp/ISWC 2026}
}
```
