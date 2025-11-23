"""
Test Streamlit Compatibility

This script tests if all Streamlit components work correctly.
Run this before deploying to Streamlit Cloud.
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    packages = [
        'streamlit',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'sklearn',
        'xgboost',
        'joblib',
    ]
    
    failed = []
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True


def test_matplotlib_backend():
    """Test matplotlib backend configuration."""
    print("\nTesting matplotlib backend...")
    
    import matplotlib
    backend = matplotlib.get_backend()
    print(f"Current backend: {backend}")
    
    if backend == 'agg':
        print("✓ Backend is 'agg' (Streamlit compatible)")
    else:
        print("⚠ Backend is not 'agg', setting to 'agg'")
        matplotlib.use('Agg')
        print("✓ Backend set to 'agg'")
    
    return True


def test_file_structure():
    """Test if required files exist."""
    print("\nTesting file structure...")
    
    import os
    
    required_files = [
        'app.py',
        'requirements.txt',
        '.streamlit/config.toml',
        'packages.txt',
    ]
    
    optional_files = [
        'models/best_model.pkl',
        'data/customer_churn.csv',
        'data/sample_customer_churn.csv',
    ]
    
    missing_required = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (REQUIRED)")
            missing_required.append(file)
    
    print("\nOptional files:")
    for file in optional_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"⚠ {file} (optional)")
    
    if missing_required:
        print(f"\n❌ Missing required files: {', '.join(missing_required)}")
        return False
    else:
        print("\n✅ All required files present!")
        return True


def test_streamlit_config():
    """Test Streamlit configuration."""
    print("\nTesting Streamlit configuration...")
    
    import os
    import toml
    
    config_path = '.streamlit/config.toml'
    if not os.path.exists(config_path):
        print(f"✗ {config_path} not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = toml.load(f)
        
        print(f"✓ Configuration loaded")
        
        # Check important settings
        if 'server' in config:
            print(f"  Server settings: {list(config['server'].keys())}")
        if 'theme' in config:
            print(f"  Theme configured")
        
        return True
    except Exception as e:
        print(f"✗ Error reading config: {e}")
        return False


def test_model_loading():
    """Test if model can be loaded."""
    print("\nTesting model loading...")
    
    import os
    import joblib
    
    model_paths = [
        'models/best_model.pkl',
    ]
    
    # Check results directory
    if os.path.exists('results'):
        result_dirs = [d for d in os.listdir('results') if os.path.isdir(os.path.join('results', d))]
        if result_dirs:
            latest = max(result_dirs)
            model_paths.append(f'results/{latest}/best_model.pkl')
    
    model_found = False
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                print(f"✓ Model loaded from {path}")
                print(f"  Model type: {type(model).__name__}")
                model_found = True
                break
            except Exception as e:
                print(f"✗ Error loading {path}: {e}")
    
    if not model_found:
        print("⚠ No trained model found")
        print("  Run 'python main.py' to train a model")
        print("  Or the app will prompt users to upload data")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Streamlit Compatibility Test Suite")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Matplotlib Backend", test_matplotlib_backend),
        ("File Structure", test_file_structure),
        ("Streamlit Config", test_streamlit_config),
        ("Model Loading", test_model_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your app is Streamlit-ready!")
        print("\nNext steps:")
        print("1. Run locally: streamlit run app.py")
        print("2. Deploy to Streamlit Cloud: See STREAMLIT_DEPLOYMENT.md")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    try:
        import toml
    except ImportError:
        print("Installing toml package for config testing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "toml"])
        import toml
    
    sys.exit(main())
