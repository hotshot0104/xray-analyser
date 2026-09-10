import os
import cv2
import numpy as np
from medtrust_engine import MedTrustPipeline, DISEASE_CLASSES

def run_tests():
    print("="*60)
    print("      MEDTRUST-XRAY SYSTEM VERIFICATION TEST RUNNER       ")
    print("="*60)
    
    # 1. Check Samples
    samples_dir = 'd:/jalaj/samples'
    sample_files = [f for f in os.listdir(samples_dir) if f.endswith('.png')]
    print(f"[1/4] Checking sample radiographs in {samples_dir}...")
    print(f"      Found {len(sample_files)} sample files.")
    assert len(sample_files) >= 5, "Missing required sample images."
    
    # 2. Check Pipeline Initialization
    print("[2/4] Initializing MedTrust 3-Stage Pipeline...")
    pipeline = MedTrustPipeline()
    print("      Pipeline successfully initialized.")
    
    # 3. Test Clean Sample (Should Pass with High Trust)
    print("[3/4] Testing Clean Verified Case (02_Pneumonia_Verified.png)...")
    clean_img = cv2.imread(os.path.join(samples_dir, '02_Pneumonia_Verified.png'))
    res_clean = pipeline.run(clean_img)
    trust_clean = res_clean['trust']['trust_score'] * 100
    status_clean = res_clean['trust']['status_label']
    print(f"      Disease: {res_clean['top_pathology'][0]} ({res_clean['top_pathology'][1]*100:.1f}%)")
    print(f"      Trust Score: {trust_clean:.1f}% | Verdict: {status_clean}")
    assert trust_clean >= 70.0, f"Clean sample should be verified reliable, got {trust_clean}%"
    
    # 4. Test Blurry Corrupted Sample (Should Trigger "DO NOT TRUST")
    print("[4/4] Testing Corrupted Case (03_Pneumonia_Blurry_DoNotTrust.png)...")
    corrupt_img = cv2.imread(os.path.join(samples_dir, '03_Pneumonia_Blurry_DoNotTrust.png'))
    res_corrupt = pipeline.run(corrupt_img)
    trust_corrupt = res_corrupt['trust']['trust_score'] * 100
    status_corrupt = res_corrupt['trust']['status_label']
    print(f"      Nominal Prob: {res_corrupt['top_pathology'][1]*100:.1f}%")
    print(f"      Stability: {res_corrupt['stability']['stability_score']*100:.1f}%")
    print(f"      Trust Score: {trust_corrupt:.1f}% | Verdict: {status_corrupt}")
    assert trust_corrupt < 70.0, f"Corrupted sample should be DO NOT TRUST, got {trust_corrupt}%"
    
    print("\n" + "="*60)
    print("   [SUCCESS] ALL VERIFICATION TESTS PASSED! (100% READY)   ")
    print("="*60)

if __name__ == '__main__':
    run_tests()
