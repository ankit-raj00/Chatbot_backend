import subprocess
import time
import requests
import sys

# Start the server in background
server_process = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])

# Wait for server to be up
max_retries = 30
server_up = False
for i in range(max_retries):
    try:
        resp = requests.get("http://localhost:8000/")
        if resp.status_code == 200:
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

print("--- Test 2: Validation Error ---")
try:
    resp = requests.post("http://localhost:8000/auth/login", json={"email": "not-an-email", "password": ""})
    print(f"Status: {resp.status_code}")
    print(resp.json())
except Exception as e:
    print(f"Failed: {e}")

print("--- Test 3: Health Check ---")
try:
    resp = requests.get("http://localhost:8000/health")
    print(f"Status: {resp.status_code}")
    print(resp.json())
except Exception as e:
    print(f"Failed: {e}")

print("--- Test 4: CORS Headers ---")
try:
    resp = requests.options("http://localhost:8000/health", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"})
    print("Allowed headers (valid origin):", resp.headers.get("access-control-allow-origin"))
    
    resp_bad = requests.options("http://localhost:8000/health", headers={"Origin": "http://evil.com", "Access-Control-Request-Method": "GET"})
    print("Allowed headers (invalid origin):", resp_bad.headers.get("access-control-allow-origin"))
except Exception as e:
    print(f"Failed: {e}")

# Kill server
server_process.terminate()
server_process.wait()
