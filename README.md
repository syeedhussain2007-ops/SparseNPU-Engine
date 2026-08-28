PROJECT TITLE:
"SpareQ-NPU: Sparse Weight Pruning and Run-Length Encoding Engine for Edge NPU"

PROBLEM STATEMENT:
Develop a highly optimized, automated weight-pruning pipeline for a Multi-Layer Perceptron (MLP). The architecture must reduce the neural network's memory footprint and computational load through unstructured sparsity while preserving baseline classification accuracy for deployment on memory-constrained Edge Neural Processing Units (NPUs).

PROJECT DOMAIN:
Edge AI / TinyML / Neural Network Optimization / NPU Inference

CORE REQUIREMENTS:
1. Aggressive Unstructured Pruning
   - Automatically prune MLP weight matrices.
   - Achieve at least 60% unstructured sparsity.
   - Preserve classification accuracy as much as possible.

2. Zero-Skipping Compression
   - Remove unnecessary zero values from the model representation.
   - Support a sparse compression approach such as Run-Length Encoding (RLE) or CSR.
   - Reduce the physical memory required to store model weights.

3. Sparse Execution Engine
   - Implement an optimized C++ sparse inference engine.
   - Execute compressed sparse weights directly.
   - Avoid fully decompressing the model into dense RAM during inference.
   - Use zero-skipping during computation.

4. Performance Targets
   - Demonstrate approximately 2× reduction in physical memory footprint.
   - Demonstrate approximately 1.5× computational/inference speedup compared with the uncompressed dense baseline.
   - Verify that classification accuracy remains acceptable.

PROJECT TEAM STRUCTURE:
Member 1:
- Baseline MLP training
- Weight pruning
- Fine-tuning

Member 2:
- RLE-based weight compression

Member 3:
- Sparse representation
- Pipeline integration

Member 4:
- Sparse inference
- Performance benchmarking

MY MODULE — MEMBER 4:
- Develop the sparse inference component.
- Read and process compressed/RLE sparse weights.
- Perform inference without fully converting the model back into a dense representation.
- Implement zero-skipping computation.
- Compare sparse inference against the dense baseline.
- Measure inference time, memory usage, speedup, and accuracy.
- Generate benchmark results and save them in the results/ directory.
- Integrate the sparse inference module with the outputs produced by the previous pipeline stages.

CURRENT PROJECT STRUCTURE:

SpareQ-NPU/
│
├── member1/
│   └── train.py
│
├── member2/
│   └── prune.py
│
├── member3/
│   └── rle.py
│
├── member4/
│   └── sparse_inference.py
│
├── models/
│
├── results/
│   └── inference.json
│
└── README.md

EXECUTION FLOW:
1. Train the baseline MLP.
2. Prune unnecessary weights.
3. Fine-tune the pruned model.
4. Convert sparse weights into a compressed representation using RLE/CSR.
5. Pass the compressed representation to the sparse inference engine.
6. Perform inference using zero-skipping.
7. Compare sparse inference with dense baseline inference.
8. Measure:
   - Accuracy
   - Model size
   - Memory footprint
   - Inference time
   - Computational speedup
   - Sparsity percentage
9. Save benchmark results to results/inference.json.
10. Verify whether the project meets the target of 60% sparsity, 2× memory reduction, and 1.5× speedup.
