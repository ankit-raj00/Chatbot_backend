import subprocess
import time
import requests
import sys

# Start the server
server_process = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])

# Wait for server to be up
max_retries = 30
server_up = False
for i in range(max_retries):
    try:
        resp = requests.get("http://localhost:8000/health")
        if resp.status_code in [200, 503]:
            server_up = True
            print(f"Server is up after {i+1} seconds!")
            break
    except:
        pass
    time.sleep(1)

if not server_up:
    print("Server failed to start in time.")
    server_process.terminate()
    sys.exit(1)

# Run e2e test
try:
    subprocess.run([sys.executable, "e2e_test.py"], check=True)
except subprocess.CalledProcessError as e:
    print(f"E2E test failed with code {e.returncode}")
finally:
    server_process.terminate()
    server_process.wait()
