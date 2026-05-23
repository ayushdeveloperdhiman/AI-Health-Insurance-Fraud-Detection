# AI-Based Real-Time Health Insurance Fraud Detection System

**EmpowerTech Solutions, Chennai | Qollabb Industry Project 1726**
**Student:** Ayush Dhiman | ayushdeveloper0711@gmail.com

---

## Project Overview

A complete, production-ready, open-source health insurance fraud detection
platform achieving F1: 0.9996, 25,117 claims/sec throughput, and 38ms
real-time dashboard latency — at zero licensing cost vs $500K/year
commercial alternatives.

---

## Results Summary

| Metric               | Result               |
| -------------------- | -------------------- |
| F1-Score             | 0.9996               |
| AUC-ROC              | 1.0000               |
| Dashboard Latency    | 38ms                 |
| Throughput           | 25,117 claims/sec    |
| SUS Usability        | 84.2 (Grade B)       |
| Production Readiness | 94.1% (16/17 checks) |
| All Objectives       | 24/24 = 100%         |

---

## How to Run

### Milestone 2 — Run the Algorithm

```bash
conda activate base
cd Milestone_2_Algorithm
pip install -r requirements.txt
python3 Milestone2_FraudDetection_Code.py
```

### Milestone 3 — Run the Dashboard

```bash
conda activate base
cd Milestone_3_Dashboard
pip install -r requirements.txt
python3 main.py
# Open Milestone3_Dashboard.html in browser
# Login: analyst@empowertech.in / Analyst@2025
```

### Milestone 4 — Run Tests

```bash
python3 Milestone_4_Test_Optimize/Milestone4_TestOptimize.py
```

### Milestone 5 — Run Evaluation

```bash
python3 Milestone_5_Evaluate/Milestone6_EvaluatePerformance.py
```

### Milestone 6 — Run Comparison

```bash
python3 Milestone_6_Compare/Milestone5_CompareExisting.py
```

---

## Tech Stack

- **ML:** XGBoost 2.x + TensorFlow/Keras 2.16 + SHAP
- **Backend:** FastAPI + uvicorn + SQLite
- **Frontend:** React.js + D3.js + Chart.js
- **Auth:** JWT + RBAC (sha256_crypt)

---

## System Requirements

- Python 3.12.x
- conda (Anaconda base environment)
- macOS / Linux / Windows
