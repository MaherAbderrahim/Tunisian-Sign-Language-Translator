# Use Python 3.12 slim image (matches the tested interpreter; setup.py requires >=3.10)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (see .dockerignore: .env is NEVER copied into the image —
# provide it at runtime via compose `environment:`/`env_file` or `-e` vars)
COPY . .

# Expose ports
EXPOSE 8000 8501

# Default command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
