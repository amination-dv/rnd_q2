"""Pre-compute model predictions for all samples."""

import sys
from tqdm import tqdm

from config import MODEL_LOCAL_PATH, MODEL_S3_URI
from data_manager import DataManager
from inference import load_model, run_inference, download_model_from_s3


def main():
    print("=" * 60)
    print("Pre-computing Model Predictions for All Samples")
    print("=" * 60)
    
    # Initialize data manager and scan samples
    print("\n[1/4] Scanning samples...")
    dm = DataManager()
    dm.scan_samples()
    
    samples = dm.get_all_samples()
    print(f"Found {len(samples)} total samples")
    
    # Check how many already have predictions
    samples_with_predictions = [s for s in samples if s.model_predictions is not None]
    samples_without_predictions = [s for s in samples if s.model_predictions is None]
    
    print(f"  - Already have predictions: {len(samples_with_predictions)}")
    print(f"  - Need predictions: {len(samples_without_predictions)}")
    
    if not samples_without_predictions:
        print("\nAll samples already have predictions. Nothing to do.")
        dm.save_results()
        return
    
    # Download model if needed
    print("\n[2/4] Loading model...")
    try:
        download_model_from_s3(MODEL_S3_URI, MODEL_LOCAL_PATH)
        model, transform = load_model(MODEL_LOCAL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure you have AWS credentials configured.")
        sys.exit(1)
    
    # Run inference on all samples without predictions
    print(f"\n[3/4] Running inference on {len(samples_without_predictions)} samples...")
    
    success = 0
    errors = 0
    
    for sample in tqdm(samples_without_predictions, desc="Processing"):
        data = dm.load_sample_data(sample)
        if data is None:
            errors += 1
            continue
        
        img_std, img_patch = data
        
        try:
            probs = run_inference(model, transform, img_std, img_patch)
            dm.update_sample(
                sample.sample_id,
                model_predictions=probs.tolist(),
            )
            success += 1
        except Exception as e:
            print(f"\nError processing {sample.sample_id}: {e}")
            errors += 1
    
    # Save results
    print(f"\n[4/4] Saving results...")
    dm.save_results()
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Successfully processed: {success}")
    print(f"  Errors: {errors}")
    print(f"  Total with predictions: {len(samples_with_predictions) + success}")
    print(f"\nResults saved to: {dm.results_file}")
    print("\nYou can now run the UI with: ./run.sh")


if __name__ == "__main__":
    main()
