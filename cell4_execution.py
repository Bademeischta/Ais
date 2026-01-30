# Cell 4: Run the Server

from pyngrok import ngrok
import subprocess
import threading

# Set ngrok auth
if NGROK_AUTH_TOKEN:
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
else:
    print("⚠️ Warning: NGROK_AUTH_TOKEN not set. Tunnel may fail.")

# Start tunnel
try:
    public_url = ngrok.connect(5000)
    print("\n" + "="*60)
    print("🎮 ALGORITHM ARENA IS LIVE!")
    print("="*60)
    print(f"\n🌐 Open in browser: {public_url}\n")
    print("="*60 + "\n")
except Exception as e:
    print(f"❌ Ngrok failed: {e}")

# Run the app
exec(open('app.py').read())
