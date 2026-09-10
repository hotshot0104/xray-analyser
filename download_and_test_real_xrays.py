import os
import urllib.request
import cv2
from medtrust_engine import MedTrustPipeline

def download_real_xrays():
    os.makedirs('d:/jalaj/real_test_images', exist_ok=True)
    
    # Real clinical chest radiographs from open medical repositories
    real_images = {
        'real_patient_01_pneumonia.jpg': 'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/000001-1.jpg',
        'real_patient_02_infiltration.jpg': 'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/000001-10.jpg',
        'real_patient_03_consolidation.jpg': 'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/000001-12.jpg',
        'real_patient_04_bilateral.jpg': 'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/000001-20.jpg',
    }
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    downloaded_paths = []
    
    print("="*70)
    print("      DOWNLOADING REAL CLINICAL CHEST X-RAYS FROM INTERNET        ")
    print("="*70)
    
    for filename, url in real_images.items():
        save_path = os.path.join('d:/jalaj/real_test_images', filename)
        try:
            print(f"Fetching {filename}...")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp, open(save_path, 'wb') as out_f:
                out_f.write(resp.read())
            print(f"  [OK] Saved to {save_path} ({os.path.getsize(save_path):,} bytes)")
            downloaded_paths.append((filename, save_path))
        except Exception as e:
            print(f"  [Error] Failed to fetch {filename}: {e}")
            
    return downloaded_paths

def test_pipeline_on_real_images(image_list):
    print("\n" + "="*70)
    print("    RUNNING MEDTRUST 'DO NOT TRUST' PIPELINE ON REAL PATIENT SCANS   ")
    print("="*70)
    
    pipeline = MedTrustPipeline()
    
    for filename, img_path in image_list:
        print("\n" + "-"*70)
        print(f"Evaluating Real Scan: {filename}")
        print("-"*70)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image at {img_path}")
            continue
            
        results = pipeline.run(img)
        
        top_d, top_p = results['top_pathology']
        iq = results['iq']
        stab = results['stability']
        agree = results['agreement_score']
        trust = results['trust']
        
        # Display top 3 predicted pathologies
        sorted_preds = sorted(results['models']['ensemble_preds'].items(), key=lambda x: x[1], reverse=True)
        print("  Top Detected Pathologies (14 NIH Diseases):")
        for disease, prob in sorted_preds[:3]:
            bar_len = int(prob * 20)
            bar = "#" * bar_len + "-" * (20 - bar_len)
            print(f"    * {disease:<20} : {prob*100:5.1f}%  [{bar}]")
            
        # Display Stage 2 Reliability Metrics
        print(f"\n  Stage 2 Reliability & Stress-Tests:")
        print(f"    * Image Quality Score   : {iq['overall_quality']*100:5.1f}%  (Laplacian Blur: {iq['blur_metric']:.1f}, Noise: {iq['noise_sigma']:.1f})")
        print(f"    * Perturbation Stability: {stab['stability_score']*100:5.1f}%  (Instability Std: {stab['mean_instability_std']:.4f})")
        print(f"    * Model Consensus       : {agree*100:5.1f}%  (DenseNet, ResNet, EfficientNet)")
        print(f"    * XAI Grad-CAM Focus    : {results['xai']['xai_stability']*100:5.1f}%")
        
        # Display Stage 3 Final Trust Verdict
        print(f"\n  Stage 3 Trust Supervisor Gate:")
        print(f"    >> FINAL TRUST SCORE    : {trust['trust_score']*100:5.1f}%")
        print(f"    >> CLINICAL VERDICT     : {trust['status_label']}")
        print(f"    >> ACTION               : {trust['is_trusted'] and 'Verified Reliable for Diagnosis' or 'DO NOT TRUST - Flagged for Radiologist Review'}")

if __name__ == '__main__':
    images = download_real_xrays()
    if images:
        test_pipeline_on_real_images(images)
