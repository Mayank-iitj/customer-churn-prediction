# Health Check Endpoint
# Simple health check script for monitoring

import os
import sys
import requests

def check_health():
    """Check if the Streamlit app is healthy."""
    try:
        response = requests.get('http://localhost:8501/_stcore/health', timeout=5)
        if response.status_code == 200:
            print("✓ Application is healthy")
            return 0
        else:
            print(f"✗ Application returned status code: {response.status_code}")
            return 1
    except requests.exceptions.RequestException as e:
        print(f"✗ Health check failed: {e}")
        return 1

def check_model_exists():
    """Check if trained model exists."""
    model_paths = [
        'models/best_model.pkl',
        'results/*/best_model.pkl'
    ]
    
    for path_pattern in model_paths:
        if '*' in path_pattern:
            import glob
            paths = glob.glob(path_pattern)
            if paths and os.path.exists(paths[0]):
                print(f"✓ Model found at: {paths[0]}")
                return 0
        elif os.path.exists(path_pattern):
            print(f"✓ Model found at: {path_pattern}")
            return 0
    
    print("✗ No trained model found")
    return 1

def main():
    """Run all health checks."""
    print("Running health checks...")
    print("-" * 40)
    
    checks = [
        ("Application Health", check_health),
        ("Model Availability", check_model_exists)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        result = check_func()
        results.append(result)
    
    print("\n" + "=" * 40)
    if all(r == 0 for r in results):
        print("All health checks passed ✓")
        return 0
    else:
        print("Some health checks failed ✗")
        return 1

if __name__ == "__main__":
    sys.exit(main())
