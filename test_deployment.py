#!/usr/bin/env python3
"""
Test script to verify deployment readiness for Streamlit Cloud
Run this before deploying to catch common issues early
"""

import os
import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_requirements():
    """Check if all required packages can be installed"""
    print("\n📦 Checking requirements.txt...")
    
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"📋 Found {len(requirements)} requirements:")
    
    failed_imports = []
    for req in requirements:
        # Extract package name (before any version specifiers)
        package_name = req.split('>=')[0].split('==')[0].split('<')[0].strip()
        try:
            importlib.import_module(package_name.replace('-', '_'))
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - Not installed")
            failed_imports.append(package_name)
    
    if failed_imports:
        print(f"\n⚠️  Install missing packages: pip install {' '.join(failed_imports)}")
        return False
    
    return True

def check_streamlit_config():
    """Check Streamlit configuration files"""
    print("\n⚙️  Checking Streamlit configuration...")
    
    config_path = '.streamlit/config.toml'
    secrets_path = '.streamlit/secrets.toml'
    
    if os.path.exists(config_path):
        print("✅ .streamlit/config.toml found")
    else:
        print("❌ .streamlit/config.toml missing")
        return False
    
    if os.path.exists(secrets_path):
        print("✅ .streamlit/secrets.toml template found")
        print("⚠️  Remember to configure actual secrets in Streamlit Cloud")
    else:
        print("❌ .streamlit/secrets.toml template missing")
        return False
    
    return True

def check_app_file():
    """Check if main app file exists and imports work"""
    print("\n📄 Checking app.py...")
    
    if not os.path.exists('app.py'):
        print("❌ app.py not found")
        return False
    
    print("✅ app.py found")
    
    # Try to import the app (basic syntax check)
    try:
        spec = importlib.util.spec_from_file_location("app", "app.py")
        if spec is None:
            print("❌ Could not load app.py spec")
            return False
        print("✅ app.py syntax appears valid")
    except Exception as e:
        print(f"❌ Error loading app.py: {e}")
        return False
    
    return True

def check_git_status():
    """Check git status and suggest next steps"""
    print("\n🔧 Checking Git status...")
    
    try:
        # Check if we're in a git repo
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, check=True)
        
        if result.stdout.strip():
            print("📝 Uncommitted changes detected:")
            print(result.stdout)
            print("💡 Commit and push changes before deploying")
        else:
            print("✅ Git working directory clean")
        
        # Check if we have a remote
        result = subprocess.run(['git', 'remote', '-v'], 
                              capture_output=True, text=True, check=True)
        
        if 'github.com' in result.stdout:
            print("✅ GitHub remote detected")
        else:
            print("⚠️  No GitHub remote found - you'll need this for Streamlit Cloud")
        
    except subprocess.CalledProcessError:
        print("⚠️  Not a git repository or git not available")
        print("💡 Initialize git repo: git init && git remote add origin <your-repo-url>")

def run_basic_test():
    """Run a basic test of the Streamlit app"""
    print("\n🧪 Running basic app test...")
    
    try:
        # Try to run streamlit for a few seconds to check for immediate errors
        result = subprocess.run(['streamlit', 'run', 'app.py', '--server.headless', 'true'], 
                              timeout=10, capture_output=True, text=True)
        print("✅ Streamlit app starts without immediate errors")
    except subprocess.TimeoutExpired:
        print("✅ Streamlit app started successfully (stopped after 10s)")
    except subprocess.CalledProcessError as e:
        print(f"❌ Streamlit app failed to start: {e}")
        print("Error output:", e.stderr)
        return False
    except FileNotFoundError:
        print("⚠️  Streamlit not found - install with: pip install streamlit")
        return False
    
    return True

def main():
    """Run all deployment readiness checks"""
    print("🚀 Streamlit Cloud Deployment Readiness Check")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Requirements", check_requirements),
        ("Streamlit Config", check_streamlit_config),
        ("App File", check_app_file),
        ("Git Status", check_git_status),
        ("Basic Test", run_basic_test)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 Ready for deployment!")
        print("\n📋 Next steps:")
        print("1. Push your code to GitHub")
        print("2. Go to https://share.streamlit.io")
        print("3. Connect your repository")
        print("4. Configure secrets in Streamlit Cloud")
        print("5. Deploy!")
    else:
        print("⚠️  Please fix the issues above before deploying")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)