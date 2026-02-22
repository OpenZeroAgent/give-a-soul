FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install basic dependencies if needed (numpy usually included)
RUN pip install numpy

# Copy the core system logic
COPY quad_crystal_system.py .

# Copy the continuity state (bootstrapped from V2)
COPY initial_state.pt .

# Default command runs the simulation step
CMD ["python3", "quad_crystal_system.py"]
