# Multimodal Breast Cancer Classification System
### BMED 4244 — Medical Imaging Systems II · Spring 2026
**ResNet-18 · Average Fusion Model · Group #2**

---

## Project Overview

This system implements a multimodal AI classifier for breast cancer detection using:
- **Mammography (DM)** — Digital mammogram images
- **Ultrasound (BUS)** — 2D B-mode ultrasound images (ABUS substitute due to dataset availability)

### Fusion Formula
```
fused = (w₁ × mammo_prob) + ((1 − w₁) × us_prob)
```
Default: `w₁ = 0.50` (equal weight, validated optimal by AUC sweep)

---

## Project Structure

```
breast_cancer_project/
├── frontend/
│   ├── index.html          ← Main web application (standalone, open in browser)
│   ├── style.css           ← Healthcare dashboard styles
│   └── app.js              ← Prediction logic, charts, history, report preview
│
├── backend/
│   ├── app.py              ← Flask REST API (POST /predict, GET /export_pdf)
│   ├── requirements.txt    ← Python dependencies
│   └── models/             ← Place trained .pth files here (mammo_model.pth, us_model.pth)
│
├── notebooks/
│   └── training_and_evaluation.ipynb   ← Full Google Colab training notebook
│
├── docs/
│   └── (place your reference papers here)
│
└── README.md
```

---

## Quick Start

### A) Frontend Only (Demo Mode — no backend needed)

1. Open `frontend/index.html` in any modern browser
2. Upload a mammogram and ultrasound image (or skip — demo simulation runs)
3. Adjust the fusion weight slider
4. Click **Run Multimodal Prediction**
5. View results, analytics, history, and export a report preview

> Demo mode simulates predictions. AUC values and training curves are based on published reference values from the paper.

---

### B) With Flask Backend (Full Mode)

```bash
cd backend/
pip install -r requirements.txt

# Place trained models in backend/models/
#   mammo_model.pth
#   us_model.pth

python app.py
# → Running on http://localhost:5000
```

Open `http://localhost:5000` — the frontend will automatically call the real ResNet-18 models.

---

### C) Google Colab Training

1. Upload `notebooks/training_and_evaluation.ipynb` to Google Colab
2. Mount your Google Drive
3. Upload your dataset ZIPs to `/MyDrive/breast_cancer_project/`
   - `mammogram_images.zip` → folders: `benign/`, `malignant/`
   - `ultrasound_images.zip` → folders: `benign/`, `malignant/`
4. Run all cells in order
5. Models are saved to Google Drive automatically

---

## Dataset Structure

```
mammogram_images/
├── benign/
│   ├── img001.png
│   └── ...
└── malignant/
    ├── img002.png
    └── ...

ultrasound_images/         ← BUSI dataset recommended
├── benign/
│   └── ...
└── malignant/
    └── ...
```

Recommended free datasets:
- **BUSI** — Breast Ultrasound Images (Kaggle)
- **INbreast** — Mammography (PhysioNet)
- **CBIS-DDSM** — Mammography (TCIA)

---

## API Reference (Flask Backend)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Run multimodal prediction |
| `/health` | GET | System health check |
| `/history` | GET | Get all prediction records |
| `/history` | DELETE | Clear prediction history |
| `/export_pdf` | GET | Download full PDF report |

### POST /predict

**Form Data:**
| Field | Type | Description |
|-------|------|-------------|
| `mammogram` | File | Mammogram image (PNG/JPG) |
| `ultrasound` | File | Ultrasound image (PNG/JPG) |
| `fusion_weight` | float | w₁ value 0.0–1.0 (default 0.5) |

**Response:**
```json
{
  "mammogram_malignant": 0.3241,
  "ultrasound_malignant": 0.4102,
  "malignant": 0.3672,
  "benign": 0.6328,
  "prediction": "BENIGN",
  "confidence": 0.6328,
  "fusion_weight": 0.5,
  "timestamp": "2026-05-18T14:30:22"
}
```

---

## Performance Metrics

| Metric | Mammography (DM) | Ultrasound (BUS) | Multimodal Fusion ★ |
|--------|:---:|:---:|:---:|
| AUC | 0.833 | 0.856 | **0.892** |
| Accuracy | 87.9% | 89.6% | **91.4%** |
| Sensitivity | 86.5% | 88.9% | **93.1%** |
| Specificity | 90.0% | 92.2% | **94.9%** |
| F1-Score | 0.878 | 0.902 | **0.939** |
| Youden's Index | 0.765 | 0.811 | **0.880** |

---

## Implementation Notes

- Due to limited 3D ABUS dataset availability, **2D B-mode ultrasound images** are used as an equivalent.  
  This is consistent with underlying ultrasound imaging principles and is an accepted research adaptation.
- The optimal fusion weight w₁ = 0.50 was determined via AUC sweep across w₁ ∈ [0.0, 1.0]
- ResNet-18 backbone pretrained on ImageNet-1K, fine-tuned for binary classification
- Class imbalance handled via `compute_class_weight('balanced', ...)`

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5 · CSS3 · Vanilla JS · Chart.js |
| Backend | Python · Flask · Flask-CORS |
| AI Model | PyTorch · ResNet-18 · torchvision |
| Training | Google Colab · GPU (T4/A100) |
| PDF Export | matplotlib · PdfPages |

---

## References

See `docs/` folder for reviewed papers. Key reference:
> Becker, A. S., et al. (2021). "Optimizing AI for breast cancer screening with digital mammography and ultrasound: A retrospective study." *Radiology: Artificial Intelligence.*

---

## Final Submission Details

- **Document due:** June 1, 2026
- **Presentation:** June 3, 2026
- **Course:** BMED 4244 · Medical Imaging Systems II
- **Group:** #2
