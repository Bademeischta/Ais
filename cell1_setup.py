# Cell 1: Setup and Dependencies

# Mount Google Drive for persistence
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install flask flask-socketio eventlet numpy torch pyngrok psutil

# Create directories
import os
os.makedirs('/content/drive/MyDrive/AlgorithmArena/saves', exist_ok=True)
os.makedirs('/content/drive/MyDrive/AlgorithmArena/logs', exist_ok=True)
os.makedirs('/content/templates', exist_ok=True)

# IMPORTANT: Set your ngrok auth token
# Get yours at https://ngrok.com (free account)
NGROK_AUTH_TOKEN = ""  # <-- PASTE YOUR TOKEN HERE

print("✓ Setup complete")
