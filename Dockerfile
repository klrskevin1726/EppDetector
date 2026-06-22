# ── Base image ────────────────────────────────────────────────────────────────
# We use a slim Python 3.11 image. "slim" means it has only the essentials,
# which keeps the final image smaller.
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# OpenCV needs these system libraries to work inside a Linux container.
# libgl1 and libglib2.0-0 handle image rendering under the hood.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
# All subsequent commands will run from /app inside the container.
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# We copy requirements.txt first (before the rest of the code) so Docker
# can cache this layer — if you only change code, it won't reinstall packages.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────────────────────
COPY main.py .
COPY best.pt .

# ── Expose port ───────────────────────────────────────────────────────────────
# This documents which port the app listens on (doesn't actually open it).
EXPOSE 8000

# ── Start command ─────────────────────────────────────────────────────────────
# This is what runs when the container starts.
# --host 0.0.0.0 is critical: it tells uvicorn to accept connections from
# outside the container (not just localhost).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]