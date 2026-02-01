# Cell 1: Smart Environment Detection & Setup
import os
import sys
import torch

def detect_environment():
    is_colab = 'google.colab' in sys.modules or os.path.exists('/content')
    has_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else "None"

    config = {
        "IS_COLAB": is_colab,
        "HAS_GPU": has_gpu,
        "GPU_NAME": gpu_name,
    }

    if is_colab:
        print(f"☁️ Cloud Mode Detected (Colab). GPU: {gpu_name}")
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            config["SAVE_DIR"] = '/content/drive/MyDrive/AlgorithmArena/saves/'
            config["LOG_DIR"] = '/content/drive/MyDrive/AlgorithmArena/logs/'
        except:
            print("⚠️ Could not mount Google Drive, using local Colab storage.")
            config["SAVE_DIR"] = '/content/saves/'
            config["LOG_DIR"] = '/content/logs/'

        config["BATCH_SIZE"] = 64
        config["NUM_WORKERS"] = 2
    else:
        print(f"🏠 Local Mode Detected. GPU: {gpu_name}")
        config["SAVE_DIR"] = './saves/'
        config["LOG_DIR"] = './logs/'

        # Optimized for high-end local hardware (RTX 5070)
        config["BATCH_SIZE"] = 512
        config["NUM_WORKERS"] = 8
        torch.backends.cudnn.benchmark = True
        print("🚀 CUDNN Benchmark enabled for maximum speed.")

    os.makedirs(config["SAVE_DIR"], exist_ok=True)
    os.makedirs(config["LOG_DIR"], exist_ok=True)

    return config

# Global Configuration
ENV_CONFIG = detect_environment()

# Install dependencies if in Colab (optional, usually done via !pip in cell)
if ENV_CONFIG["IS_COLAB"]:
    print("Installing dependencies for Colab...")
    # !pip install flask flask-socketio eventlet numpy torch pyngrok psutil --quiet
    # Note: In a python script we'd use subprocess, but this is intended for a Colab cell.

print(f"✓ Environment setup complete. Config: {ENV_CONFIG}")
