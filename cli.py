import sys
import os

# Ensure clean UTF-8 handling on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
from medtrust_engine import MedTrustPipeline

def main():
    if len(sys.argv) < 2:
        print("Usage: python cli.py <path_to_xray_image.png>")
        print("Example: python cli.py samples/02_Pneumonia_Verified.png")
        sys.exit(1)
        
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        sys.exit(1)
        
    print("\n" + "="*65)
    print("           MEDTRUST-XRAY CLINICAL DIAGNOSTIC CLI           ")
    print("="*65)
    print(f"Active Image: {image_path}")
    print("Running 3-Stage Pipeline (Diagnosis -> Stress-Test -> Trust Model)...")
    
    img = cv2.imread(image_path)
    if img is None:
        print("Error: OpenCV failed to decode image.")
        sys.exit(1)
        
    pipeline = MedTrustPipeline()
    results = pipeline.run(img)
    
    top_disease, top_prob = results['top_pathology']
    iq = results['iq']
    stab = results['stability']
    agree = results['agreement_score']
    trust = results['trust']
    
    print("\n" + "-"*65)
    print(" 1. STAGE 1: TOP SUSPECTED PATHOLOGIES (14 NIH DISEASES)")
    print("-"*65)
    sorted_preds = sorted(results['models']['ensemble_preds'].items(), key=lambda x: x[1], reverse=True)
    for disease, prob in sorted_preds[:4]:
        bar_len = int(prob * 25)
        bar = "#" * bar_len + "-" * (25 - bar_len)
        print(f"   * {disease:<20} : {prob*100:5.1f}%  [{bar}]")
        
    print("\n" + "-"*65)
    print(" 2. STAGE 2: 'DO NOT TRUST' RELIABILITY METRICS")
    print("-"*65)
    print(f"   * Image Quality Score   : {iq['overall_quality']*100:5.1f}%  (Blur: {iq['blur_metric']:.1f}, Noise: {iq['noise_sigma']:.1f})")
    print(f"   * Perturbation Stability: {stab['stability_score']*100:5.1f}%  (Instability Std: {stab['mean_instability_std']:.4f})")
    print(f"   * Multi-Model Agreement : {agree*100:5.1f}%  (DenseNet, ResNet, EfficientNet)")
    print(f"   * Saliency XAI Stability: {results['xai']['xai_stability']*100:5.1f}%  (Grad-CAM Anatomical Focus)")
    
    print("\n" + "="*65)
    print(f" 3. STAGE 3: TRUST SUPERVISOR VERDICT")
    print("="*65)
    print(f"   >> FINAL TRUST SCORE    : {trust['trust_score']*100:5.1f}%")
    print(f"   >> CLINICAL STATUS      : {trust['status_label']}")
    if trust['is_trusted']:
        print("   >> Recommendation       : Verified reliable. Safe for clinical review.")
    else:
        print("   >> Recommendation       : [DO NOT TRUST] Overconfident or unstable finding. Manual review required.")
    print("="*65 + "\n")

if __name__ == '__main__':
    main()
