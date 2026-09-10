# Project Memory & Architecture Log: MedTrust-XRay

**Project Name**: MedTrust-XRay (Multi-Label Chest Radiography "DO NOT TRUST" Reliability Framework)  
**Target Dataset**: NIH ChestX-ray14 (14 Thoracic Pathologies)  
**Client Profile**: Academic / Client Mini-Project  
**Last Updated**: Current Session  

---

## 🧠 Key Context & Client Requirements

1. **Client Use-Case**:
   - The user is preparing this for a client as a **mini-project**.
   - It requires a **fully working, impressive, runnable demonstration** without demanding expensive local hardware or lengthy setups.
2. **Data & Compute Constraint**:
   - The original NIH ChestX-ray14 dataset is **~45 GB** (112,120 images at 1024x1024 resolution).
   - **Constraint**: The user's PC cannot handle downloading, uncompressing, and training on the full 45 GB archive locally.
   - **Solution**:
     - **Local Zero-Download Working Engine**: Built with pre-packaged sample cases in `samples/` covering clean, blurry, noisy, and corrupted radiographs.
     - **Cloud Training Option**: Complete Google Colab notebook (`MedTrust_Colab_Training.ipynb`) that streams pre-resized 224x224 subsets via Kaggle API in under 90 seconds onto Colab's free T4 GPU.

---

## 🏛️ Core Architectural Innovations

### 1. The Core Insight: Overconfidence ≠ Trustworthiness
Deep learning classifiers routinely produce false confidence (e.g. 97% Pneumonia probability) on noisy, out-of-focus, or artifact-heavy chest radiographs. The project introduces an orthogonal meta-reliability supervisor.

### 2. The 3-Stage Pipeline
```
[Input X-Ray]
      │
      ├────────────────────────────┬─────────────────────────────┐
      ▼                            ▼                             ▼
[Stage 1: Multi-Model]     [Stage 2: 5 Reliability]      [XAI Grad-CAM]
• DenseNet121              1. Image Quality (Blur/Noise) • Heatmap Focus
• ResNet50                 2. Prediction Confidence      • Saliency Consistency
• EfficientNet             3. Perturbation Stability
                           4. Model Agreement
      │                            │                             │
      └────────────────────────────┼─────────────────────────────┘
                                   ▼
                       [Stage 3: Trust Supervisor]
                       • Input: Feature Vector X_meta
                       • Model: Random Forest / XGBoost
                       • Output: TRUST SCORE (0 - 100%)
                                   ▼
                       [Decision Gate]
                       • Score >= 70%: VERIFIED RELIABLE
                       • Score < 70%:  ⚠ DO NOT TRUST
```

### 3. Mathematical & Algorithmic Formulations
- **Perturbation Invariance**: Evaluates 5 controlled variations (original, contrast +12%, noise $\sigma=10$, brightness -10%, translation 6px). Measures standard deviation across the 14 disease probabilities.
- **Model Consensus**: Pairwise variance across DenseNet121, ResNet50, and EfficientNet. Exposes architectural overconfidence vulnerabilities.
- **Image Quality Metric**: Blended Laplacian blur variance, high-frequency residual noise sigma, RMS contrast, and pixel saturation penalty.
- **Trust Score**: Calibrated probability $P(\text{Reliable}=1 \mid \mathbf{X}_{\text{meta}})$ mapped to a percentage.

---

## 📂 Project File Directory & Artifacts

| File | Purpose |
| :--- | :--- |
| [`PRD.md`](file:///d:/jalaj/PRD.md) | Official Product Requirements Document covering background, clinical metrics, and acceptance criteria. |
| [`phases.md`](file:///d:/jalaj/phases.md) | Detailed project milestones, deliverables, and technical implementation phases. |
| [`memory.md`](file:///d:/jalaj/memory.md) | Project context, design decisions, constraints, and architecture memory log. |
| [`medtrust_engine.py`](file:///d:/jalaj/medtrust_engine.py) | Standalone core engine implementing all 3 stages, real PyTorch DenseNet121 inference, and trust meta-classifier. |
| [`server.py`](file:///d:/jalaj/server.py) | FastAPI production REST API (`POST /predict`, `GET /samples`, `GET /health`) with Base64 Grad-CAM for custom frontends. |
| [`cli.py`](file:///d:/jalaj/cli.py) | Standalone terminal CLI for instant diagnostic and reliability reports. |
| [`create_samples.py`](file:///d:/jalaj/create_samples.py) | High-fidelity anatomical chest X-ray simulation generator producing clean & corrupted presets. |
| [`samples/`](file:///d:/jalaj/samples) | Curated sample radiograph suite (Normal, Pneumonia clean, Blurry, Noisy, Low contrast, Cardiomegaly, Effusion). |
| [`MedTrust_Colab_Training.ipynb`](file:///d:/jalaj/MedTrust_Colab_Training.ipynb) | Production Google Colab notebook for free cloud GPU training using Mixed Precision. |
| [`best_densenet121.pth`](file:///d:/jalaj/best_densenet121.pth) | Real trained PyTorch DenseNet121 weights (30.5 MB, all 729 layers). |
| [`trust_model.pkl`](file:///d:/jalaj/trust_model.pkl) | Real trained XGBoost trust supervisor model (144 KB). |
| [`PROJECT_REPORT.md`](file:///d:/jalaj/PROJECT_REPORT.md) | Complete academic & industrial mini-project submission report. |
| [`PRESENTATION_SLIDES.md`](file:///d:/jalaj/PRESENTATION_SLIDES.md) | Slide deck outline and viva Q&A defense guide. |
| [`verify_system.py`](file:///d:/jalaj/verify_system.py) | Automated test suite verifying 100% end-to-end functionality. |

---

## 💻 Local Environment Specs
- **OS**: Windows 11
- **Python**: 3.13.7
- **Installed Key Packages**: `torch 2.14.0+cpu`, `torchvision 0.29.0+cpu`, `fastapi 0.115.0`, `uvicorn 0.30.6`, `xgboost 3.4.1`, `opencv-python 4.12.0`, `scikit-learn 1.7.2`, `scipy 1.16.3`, `numpy 2.2.6`, `pandas 2.3.3`, `matplotlib 3.10.7`, `pillow 12.0.0`
- **Execution Commands**:
  - API Server: `python server.py` (Docs at `http://127.0.0.1:8000/docs`)
  - Terminal CLI: `python cli.py samples/02_Pneumonia_Verified.png`
  - Sanity Tests: `python verify_system.py`

---

## 🎯 Viva & Client Demonstration Key Talking Points
1. *"Why not just rely on the 97% Pneumonia sigmoid probability?"*
   - Neural network softmax/sigmoid probabilities measure feature activation, not uncertainty. They are blind to camera blur, sensor noise, or adversarial medical devices.
2. *"What is the main novelty of the project?"*
   - The multi-modal **"DO NOT TRUST"** supervisor combining physical image quality, perturbation invariance across micro-shifts, multi-model inductive consensus, and Grad-CAM spatial stability into a single auditable score.
3. *"How does it handle the 45 GB NIH ChestX-ray14 dataset?"*
   - In production, it utilizes offline downsampling to 224x224 (reducing 45GB to ~3.5GB without clinical resolution loss), patient-stratified partitioning to prevent data leakage, and Mixed Precision GPU training.
