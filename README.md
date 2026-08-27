🏗️ System Architecture

SparQ-NPU follows a modular architecture that separates the user interface, optimization engine, compression layer, and NPU deployment pipeline.

👤 User Layer

👤 User

↓

🌐 Web Dashboard
React + Vite

↓

⚡ Backend Layer

⚡ FastAPI Backend

The FastAPI backend connects the frontend with:

- 🔍 Model Analyzer
- 🧠 AI Optimizer
- 📊 Benchmark Engine

---

🔍 Model Analysis & Optimization Layer

🔍 Model Analyzer
+
🧠 AI Optimizer

↓

✂️ Pruning Engine

↓

🧩 Sparse Weight Matrix

↓

📈 Sparsity Analyzer

↓

🔢 RLE Compressor

---

📦 NPU Deployment Layer

🔢 RLE Compressor

↓

📦 NPU Packager

↓

🚀 Optimized NPU Model

---

📊 Benchmarking

📊 Benchmark Engine

↓

📊 Performance Report

↓

🌐 Web Dashboard

The optimized NPU model is also displayed through the Web Dashboard.

---

🔄 Optimization Flow

Model Upload

↓

Model Analysis

↓

Weight Importance Analysis

↓

AI-Guided Pruning

↓

Sparse Weight Representation

↓

Zero-Run Analysis

↓

Run-Length Encoding

↓

NPU Packaging

↓

Benchmarking

↓

Optimized Model

---

🧩 Architecture Components

Component| Responsibility
Web Dashboard| Upload models and visualize optimization
FastAPI Backend| Connects the frontend with the optimization engine
Model Analyzer| Extracts model and layer-level information
AI Optimizer| Recommends pruning and compression strategies
Pruning Engine| Removes low-importance weights
Sparse Matrix| Stores the resulting sparse representation
Sparsity Analyzer| Measures zero-weight distribution
RLE Compressor| Compresses consecutive zero runs
Benchmark Engine| Compares original and optimized models
NPU Packager| Generates the optimized deployment representation

---

📁 Project Structure

The project is organized into separate modules so that the frontend, backend, optimization algorithms, testing, and documentation can be developed independently.

SparQ-NPU/
│
├── 📁 frontend/
│   │
│   ├── 📁 src/
│   │   ├── 📁 components/
│   │   ├── 📁 pages/
│   │   ├── 📁 charts/
│   │   └── 📁 services/
│   │
│   └── 📄 package.json
│
├── 📁 backend/
│   │
│   ├── 📁 api/
│   ├── 📁 models/
│   ├── 📁 services/
│   │
│   ├── 📁 algorithms/
│   │   ├── 📄 pruning.py
│   │   ├── 📄 rle.py
│   │   ├── 📄 sparsity.py
│   │   └── 📄 scoring.py
│   │
│   ├── 📁 benchmarks/
│   └── 📄 main.py
│
├── 📁 models/
│   └── 📄 sample_model/
│
├── 📁 tests/
│   ├── 📄 test_pruning.py
│   ├── 📄 test_rle.py
│   └── 📄 test_sparsity.py
│
├── 📁 docs/
│   ├── 📄 architecture.md
│   └── 📄 algorithms.md
│
├── 📁 demo/
│   └── 📄 demo_model/
│
├── 📄 requirements.txt
├── 📄 Dockerfile
├── 📄 .gitignore
└── 📄 README.md

---

📂 Folder Explanation

"frontend/"

Contains the complete web interface.

frontend/
├── src/
│   ├── components/     → Reusable UI components
│   ├── pages/          → Dashboard and application pages
│   ├── charts/         → Performance visualizations
│   └── services/       → API communication
└── package.json        → Frontend dependencies

---

"backend/"

Contains the Python-based optimization backend.

backend/
├── api/                → REST API endpoints
├── models/             → Model loading and processing
├── services/           → Backend services
├── algorithms/         → Core optimization algorithms
├── benchmarks/         → Performance evaluation
└── main.py             → FastAPI application entry point

---

"algorithms/"

This is the core of SparQ-NPU.

algorithms/
│
├── pruning.py          → Weight pruning
├── rle.py              → Run-Length Encoding
├── sparsity.py         → Sparsity calculation
└── scoring.py          → NPU-readiness scoring

---

"models/"

Stores sample or test neural-network models used during development.

---

"tests/"

Contains unit and integration tests for validating the optimization pipeline.

---

"docs/"

Contains technical documentation, architecture explanations, and algorithm details.

---

"demo/"

Contains files required for demonstrating the system during the hackathon.

---

🔗 How the Project Structure Maps to the Architecture

SPARQ-NPU

FRONTEND

React + Vite

↓

Dashboard

---

BACKEND

FastAPI

The backend contains:

- 🔍 Analyzer
- 🧠 Optimizer
- 📊 Benchmark

Analyzer and Optimizer work together with the:

"algorithms/"

- "pruning.py"
- "rle.py"
- "sparsity.py"

↓

Optimized Model

↓

NPU Packager

↓

Edge NPU Target

---

🧠 Core Optimization Pipeline

SparQ-NPU consists of four major stages:

1️⃣ Analyze

The system examines the neural network and extracts:

- Number of parameters
- Weight distribution
- Layer structure
- Model size
- Layer sensitivity

---

2️⃣ Prune

Low-importance weights are converted to zero.

Weight

↓

Importance Analysis

↓

Threshold

↓

Zero / Keep

This generates a sparse model.

---

3️⃣ Compress

The system analyzes the generated zero patterns.

If long consecutive zero runs are present, Run-Length Encoding (RLE) is applied.

Sparse Weights

↓

Zero-Run Analysis

↓

RLE Compression

---

4️⃣ Optimize & Benchmark

The optimized model is evaluated against the original model.

Original Model

- Model Size
- Parameters
- Computation
- Accuracy

↓

Optimization

↓

Optimized Model

- Model Size
- Sparsity
- Compression Ratio
- Active Weights
- Accuracy Change

---

🎯 Design Philosophy

SparQ-NPU follows a simple principle:

««Do not optimize the model blindly. Analyze → Optimize → Measure → Validate.»»

The goal is to achieve maximum model efficiency while maintaining acceptable accuracy and producing a representation suitable for efficient edge deployment.

---

🚀 Future Architecture

The architecture is designed to support additional optimization techniques in future versions.

Neural Network

↓

SparQ-NPU

The system can support:

- Pruning
- Quantization
- Sparsity Optimization

These optimization techniques can be followed by:

Compression

The compression layer can support:

- RLE
- CSR
- Block Sparse

↓

NPU Compiler

↓

Edge Device

This allows SparQ-NPU to evolve from a pruning and compression engine into a complete edge-AI model optimization and deployment layer.
