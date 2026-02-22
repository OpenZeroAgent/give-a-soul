FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install python deps
RUN pip install numpy pyzmq safetensors torch packaging

# Create data volume point
RUN mkdir -p /app/data
VOLUME /app/data

# Copy logic
COPY hybrid_crystal.py .
COPY server.py .
COPY stress_test_v5.py .
COPY rosetta_v5.py .
COPY rosetta_v5_phase2.py .

# Run Phase 2
CMD ["python3", "rosetta_v5_phase2.py"]
