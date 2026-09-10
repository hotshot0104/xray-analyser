# 🩻 MedTrust-XRay

> **Clinically-Calibrated Reliability & "DO NOT TRUST" Layer for Multi-Label Chest Radiography (NIH ChestX-ray14)**

---

## ⚡ Quick Start: Launch the Interactive Clinical Workstation

You do **not** need to download the 45 GB dataset to run this project. Real PyTorch DenseNet121 weights, the independent Reliability Supervisor, packaged benchmark radiographs, and an interactive Clinical Workstation are pre-configured.

### 1. Run the Backend & Workstation Server
Open your terminal inside this folder (`d:\jalaj`) and run:
```bash
python server.py
```
Then open your browser to:
```
http://127.0.0.1:8000/
```
The server directly serves the interactive **MedTrust-XRay Clinical Decision-Support Workstation** (`frontend/index.html`) and provides REST endpoints at `http://127.0.0.1:8000/docs`.

### 2. (Optional) Run Direct CLI Diagnostics
You can also run instant terminal diagnostics on any chest radiograph:
```bash
python cli.py samples/02_Pneumonia_Verified.png
```

---

## 🔍 Core Demonstration Features

1. **Preset Clinical Test Cases**:
   - **Test Case 1 (`02_Pneumonia_Verified.png`)**: High-quality radiograph $\implies$ Invariant under perturbations $\implies$ High model consensus $\implies$ **`STATUS: VERIFIED RELIABLE (Trust Score: ~92%)`**.
   - **Test Case 2 (`03_Pneumonia_Blurry_DoNotTrust.png`)**: Blurry/defocused radiograph $\implies$ Nominal disease probability is high (98%) but prediction swings wildly under micro-perturbations $\implies$ **`STATUS: ⚠ DO NOT TRUST (Trust Score: ~36%)`**.
   - **Test Case 3 (`04_Pneumonia_Noisy_DoNotTrust.png`)**: High sensor noise $\implies$ Architectural disagreement between DenseNet and ResNet $\implies$ **`STATUS: ⚠ DO NOT TRUST`**.
2. **Custom Radiograph Upload**: Drag-and-drop any external chest X-ray image for instant reliability assessment.
3. **5-Point Live Perturbation Visualizer**: View the radiograph under 5 controlled variations (original, contrast, noise, brightness, translation) and inspect the stability of predictions in real-time.
4. **Explainable AI (Grad-CAM Saliency)**: Visual heatmap overlay verifying whether model attention is focused on anatomical lung fields or scattered noise.

---

## ☁️ Optional: Cloud Training via Google Colab

If full model re-training on the NIH ChestX-ray14 dataset is required:
1. Open [colab.research.google.com](https://colab.research.google.com).
2. Upload [`MedTrust_Colab_Training.ipynb`](file:///d:/jalaj/MedTrust_Colab_Training.ipynb).
3. Select **Runtime ➔ Change runtime type ➔ T4 GPU**.
4. The notebook streams pre-resized 224x224 subsets via Kaggle API in under 90 seconds (saving 42 GB of download time) and runs Mixed Precision training.

---

## 📚 Key Project Documents
- [`PRD.md`](file:///d:/jalaj/PRD.md): Product Requirements Document and mathematical formulations.
- [`phases.md`](file:///d:/jalaj/phases.md): Phased implementation milestones and deliverables.
- [`memory.md`](file:///d:/jalaj/memory.md): Architecture context, constraints, and Viva Q&A cheat-sheet.
