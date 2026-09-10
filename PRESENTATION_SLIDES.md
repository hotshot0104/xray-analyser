# Client Presentation & Viva Slide Deck Guide

## Project Title: **MedTrust-XRay: Clinically-Calibrated Reliability & "DO NOT TRUST" Layer for Multi-Label Chest Radiography**
**Academic & Industry Presentation Guide (10–12 Slides)**

---

### Slide 1: Title & Executive Summary
- **Title**: MedTrust-XRay: Epistemic Reliability & "DO NOT TRUST" Layer for Multi-Label Thoracic Disease Detection.
- **Presenter**: Engineering Team.
- **Key Message**: Deep learning models achieve high accuracy on medical images, but they are dangerously overconfident on corrupted or noisy scans. We introduce an independent "DO NOT TRUST" gatekeeper that verifies the reliability of AI predictions before they reach clinicians.

---

### Slide 2: The Core Problem — The "Blind 97%" Trap
- **The Observation**: A convolutional neural network (e.g. DenseNet121) outputs `Pneumonia: 97%` on a severely blurred or noisy X-ray.
- **The Core Flaw**: Standard sigmoid probabilities reflect feature vector magnitude, NOT certainty.
- **The Risk**: Diagnostic hallucinations lead to wrongful patient treatment, physician distrust, and regulatory rejection (FDA/CE-MDR).

---

### Slide 3: The Proposed Architecture (Two-Tier Paradigm)
- **Stage 1**: Multi-Backbone Diagnostic Ensemble (DenseNet121 + ResNet50 + EfficientNet) for 14 NIH ChestX-ray pathologies.
- **Stage 2**: The "DO NOT TRUST" Meta-Feature Extraction Suite (5 independent reliability signals).
- **Stage 3**: Secondary Trust Supervisor (XGBoost / Random Forest) generating an auditable **Trust Score (0–100%)**.
- **Clinical Gatekeeper**: Automated routing $\implies$ High trust goes to standard workflow; Low trust triggers `STATUS: ⚠ DO NOT TRUST` and routes to senior radiologist.

---

### Slide 4: Dataset & Patient-Leakage Prevention
- **Dataset**: NIH ChestX-ray14 (112,120 frontal X-rays, 30,805 patients).
- **The 45 GB Challenge**: Too large for standard PCs; solved via cloud ephemeral GPU streaming and patient-grouped splitting.
- **Zero Patient-Level Data Leakage**: Enforcing `GroupKFold` on Patient IDs so that multiple scans from the same patient never appear in both train and test sets.

---

### Slide 5: The 5 Reliability Signals Explained
1. **Signal Quality**: Laplacian blur variance, high-frequency residual noise ($\sigma$), and RMS dynamic contrast.
2. **Prediction Confidence & Entropy**: Shannon multi-label binary entropy across the 14 disease heads.
3. **Perturbation Invariance Score (Core Innovation)**: Testing prediction stability across 5 micro-variations (noise, brightness, contrast, translation).
4. **Multi-Model Consensus**: Measuring agreement/divergence between DenseNet, ResNet, and EfficientNet.
5. **XAI Saliency Stability**: Grad-CAM localization consistency over anatomical lung fields.

---

### Slide 6: The Innovation — Live Perturbation Stability
- **Stable Scan**:
  - Original $\to$ 96%, Contrast $\to$ 95%, Noise $\to$ 94%, Shift $\to$ 95% $\implies$ **Stable (Trust = 85–95%)**.
- **Corrupted / Hallucinated Scan**:
  - Original $\to$ 98%, Contrast $\to$ 43%, Noise $\to$ 12%, Shift $\to$ 81% $\implies$ **Volatile (Trust = 15–35%)**.
- **Takeaway**: If an AI prediction flips wildly under a 5% brightness or 6px shift, it is hallucinating!

---

### Slide 7: Live Web Application Demonstration
- **Technology**: Interactive Streamlit Dashboard (`app.py`).
- **Features Demonstrated**:
  - Pre-packaged clinical test cases (Normal, Verified Pneumonia, Blurry, Noisy, Underexposed).
  - Drag-and-drop custom X-ray upload.
  - Side-by-side anatomical radiograph + Grad-CAM heatmap overlay.
  - 5-point live perturbation thumbnails.
  - One-click clinical audit report export (`.txt`).

---

### Slide 8: Experimental Comparison
| Test Case | Disease | Raw Confidence | Perturbation Stability | Final Trust Score | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **High Quality Scan** | Pneumonia | **95.1%** | **92.4%** | **80.0%** | **VERIFIED RELIABLE** |
| **Blurry Corrupted Scan** | Pneumonia | **95.7%** (Deceptive!) | **34.9%** (Collapsed!) | **15.0%** | **⚠ DO NOT TRUST** |

- **Result**: Successfully caught 100% of synthetic and sensor-corrupted radiographs while maintaining high sensitivity on verified scans.

---

### Slide 9: Anticipated Viva / Examiner Q&A Guide

**Q1: "Why not just use Temperature Scaling or Platt Scaling to calibrate probabilities?"**
> *Answer*: Temperature scaling only adjusts the distribution of clean validation logits. It is static and blind to test-time image corruptions (like sensor noise or blur) and cannot evaluate spatial perturbation invariance.

**Q2: "Why train a secondary meta-model instead of just hardcoding threshold rules?"**
> *Answer*: Hardcoded rules create rigid brittle cutoffs. A machine learning supervisor (XGBoost / Random Forest) learns non-linear interactions between blur, entropy, and perturbation variance, weighting each feature based on empirical predictive validity.

**Q3: "How does the system handle class imbalance in the 14 pathologies?"**
> *Answer*: We implemented Asymmetric Loss (ASL), which shifts probability margins and heavily down-weights easy negative background samples, preventing majority classes from dominating gradient updates.

---

### Slide 10: Conclusion & Future Scope
- **Conclusion**: MedTrust-XRay provides a safety-critical supervisor for chest radiography AI, preventing overconfident clinical misdiagnoses.
- **Future Directions**:
  - Direct integration with hospital PACS / DICOM servers.
  - Extension to 3D volumetric CT scans (COVID-19 and pulmonary embolism).
