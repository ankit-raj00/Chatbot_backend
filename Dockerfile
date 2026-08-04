FROM python:3.11-slim

# System deps: build tools (for packages needing compilation), poppler-utils
# (pdftoppm, used by the create-pdf skill), and weasyprint's rendering stack
# (cairo/pango/gdk-pixbuf) — weasyprint is in requirements.txt for PDF/doc
# generation and fails at import time without these.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
