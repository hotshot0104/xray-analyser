import os
import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from medtrust_engine import MedTrustPipeline, DISEASE_CLASSES

app = FastAPI(
    title="MedTrust-XRay Diagnostic & Reliability API",
    description="Backend API for Multi-Label Chest Radiography and 'DO NOT TRUST' Reliability Supervisor",
    version="1.0.0"
)

# Enable CORS for all local web clients (React, HTML/JS, Vue, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline once at server startup
pipeline = MedTrustPipeline()

@app.get("/")
def serve_frontend():
    """Serves the MedTrust-XRay Clinical Decision Support Workstation."""
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "MedTrust-XRay API online. Frontend available at /frontend/index.html"}

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "model_loaded": pipeline.diagnostic_engine.torch_wrapper.is_ready,
        "device": pipeline.diagnostic_engine.torch_wrapper.device,
        "supported_pathologies": DISEASE_CLASSES
    }

@app.get("/samples")
def list_sample_images():
    """List pre-packaged benchmark chest radiographs."""
    samples_dir = os.path.join(os.path.dirname(__file__), "samples")
    if not os.path.exists(samples_dir):
        return {"samples": []}
    files = sorted([f for f in os.listdir(samples_dir) if f.endswith(".png")])
    return {"samples": files}

@app.get("/samples/{filename}")
def get_sample_image(filename: str):
    """Serve specific packaged sample radiograph image."""
    samples_dir = os.path.join(os.path.dirname(__file__), "samples")
    filepath = os.path.join(samples_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(filepath, media_type="image/png")

@app.post("/predict")
async def predict_radiograph(file: UploadFile = File(...)):
    """
    Core Inference & Reliability Endpoint:
    Receives chest X-ray image -> Runs Stage 1 Diagnosis -> Stage 2 Stress Tests -> Stage 3 Trust Model.
    Returns complete JSON breakdown and Base64 Grad-CAM heatmap overlay.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")
            
        # Run 3-Stage Pipeline
        results = pipeline.run(image_bgr)
        
        top_disease, top_prob = results['top_pathology']
        iq = results['iq']
        stability = results['stability']
        trust = results['trust']
        
        # Convert Grad-CAM overlay and raw heatmap to Base64 PNG
        _, buffer = cv2.imencode('.png', results['xai']['overlay'])
        gradcam_b64 = base64.b64encode(buffer).decode('utf-8')
        
        _, raw_buf = cv2.imencode('.png', results['xai']['heatmap'])
        raw_heatmap_b64 = base64.b64encode(raw_buf).decode('utf-8')

        # Multi-model consensus comparison
        models_data = results['models']
        model_breakdown = {
            "densenet": {k: round(float(v), 4) for k, v in models_data['densenet_preds'].items()},
            "resnet": {k: round(float(v), 4) for k, v in models_data['resnet_preds'].items()},
            "efficientnet": {k: round(float(v), 4) for k, v in models_data['efficientnet_preds'].items()},
            "primary_disease_scores": {
                "densenet": round(float(models_data['densenet_preds'].get(top_disease, 0.0)), 4),
                "resnet": round(float(models_data['resnet_preds'].get(top_disease, 0.0)), 4),
                "efficientnet": round(float(models_data['efficientnet_preds'].get(top_disease, 0.0)), 4),
            },
            "agreement_score": round(float(results['agreement_score']), 4)
        }

        # 5-Point Perturbation stress-test details
        pert_specs = [
            ("Baseline (Original)", "Reference unperturbed scan"),
            ("Contrast +12%", "Dynamic range micro-variation"),
            ("Noise σ=10", "High-frequency sensor perturbation"),
            ("Brightness -10%", "Sub-optimal penetration test"),
            ("Shift +6px", "Patient anatomical micro-displacement")
        ]
        perturbations_data = []
        variation_imgs = stability.get('variation_images', [])
        variation_preds = stability.get('variations', [])
        
        for i, (p_name, p_desc) in enumerate(pert_specs):
            prob = float(variation_preds[i].get(top_disease, top_prob)) if i < len(variation_preds) else float(top_prob)
            delta = float(prob - top_prob)
            
            thumb_b64 = ""
            if i < len(variation_imgs):
                v_img = variation_imgs[i]
                v_thumb = cv2.resize(v_img, (128, 128))
                _, v_buf = cv2.imencode('.png', v_thumb)
                thumb_b64 = f"data:image/png;base64,{base64.b64encode(v_buf).decode('utf-8')}"
                
            perturbations_data.append({
                "name": p_name,
                "description": p_desc,
                "disease_probability": round(prob, 4),
                "delta": round(delta, 4),
                "delta_percentage": f"{delta * 100:+.1f}%",
                "thumbnail_base64": thumb_b64
            })
        
        return {
            "success": True,
            "filename": file.filename,
            "primary_finding": {
                "disease": top_disease,
                "nominal_probability": round(float(top_prob), 4),
                "nominal_percentage": f"{top_prob * 100:.1f}%"
            },
            "pathologies": {
                k: round(float(v), 4) for k, v in results['models']['ensemble_preds'].items()
            },
            "reliability": {
                "image_quality_score": round(float(iq['overall_quality']), 4),
                "blur_metric": round(float(iq['blur_metric']), 2),
                "noise_sigma": round(float(iq['noise_sigma']), 2),
                "contrast_metric": round(float(iq['contrast_metric']), 2),
                "perturbation_stability_score": round(float(stability['stability_score']), 4),
                "instability_std": round(float(stability['mean_instability_std']), 4),
                "model_agreement_score": round(float(results['agreement_score']), 4),
                "saliency_xai_stability": round(float(results['xai']['xai_stability']), 4)
            },
            "trust_supervisor": {
                "trust_score": round(float(trust['trust_score']), 4),
                "trust_percentage": f"{trust['trust_score'] * 100:.1f}%",
                "is_trusted": trust['is_trusted'],
                "status": trust['status_label'],
                "recommendation": (
                    "Safe and reliable. The scan is clear and models agree."
                    if trust['is_trusted'] else
                    "Warning: Unreliable AI prediction. A doctor must review this X-ray manually."
                )
            },
            "xai_heatmap_base64": f"data:image/png;base64,{gradcam_b64}",
            "raw_heatmap_base64": f"data:image/png;base64,{raw_heatmap_b64}",
            "model_breakdown": model_breakdown,
            "perturbations": perturbations_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
