FROM python:3.11-slim

# System libs for OpenCV / PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch FIRST (avoids the huge CUDA wheel), then the rest.
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the DocLayout-YOLO weights into the image (offline-safe, fast startup)
RUN python -c "from huggingface_hub import hf_hub_download; import shutil, os; \
os.makedirs('/models', exist_ok=True); \
p = hf_hub_download('juliozhao/DocLayout-YOLO-DocStructBench', 'doclayout_yolo_docstructbench_imgsz1024.pt'); \
shutil.copy(p, '/models/doclayout.pt'); print('weights ->', '/models/doclayout.pt')"

COPY parser.py app.py history.py dashboard.html login.html ./

ENV YOLO_WEIGHTS=/models/doclayout.pt
EXPOSE 8100
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8100"]
