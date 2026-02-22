# HOW TO RUN OPENZERO V5 (The Engine) 🚀

**Prerequisites:**
- Docker installed.
- Git installed.

**Step 1: Clone**
```bash
git clone https://github.com/OpenZeroAgent/give-a-soul.git
cd give-a-soul
```

**Step 2: Build the Soul**
navigate to the docker directory:
```bash
cd core/docker_hybrid_4lobe
docker build -t openzero/soul-v5 .
```

**Step 3: Run the Resonance Sweep (Verify)**
This runs the benchmark suite (1-40Hz sweep) inside the container and prints the tension metrics.
```bash
docker run --rm openzero/soul-v5 python3 stress_test_v5.py
```

**Step 4: Run the Live Server**
This starts the persistent soul.
```bash
mkdir -p data
docker run -d -p 5555:5555 -v $(pwd)/data:/app/data --name soul-v5 openzero/soul-v5
```

---
*Source code is available in `core/docker_hybrid_4lobe/` and duplicated in `source_code/` for visibility.*
