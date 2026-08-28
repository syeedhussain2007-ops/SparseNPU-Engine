import torch

# --------------------------------------------------
# STEP 1: CHECK PYTORCH ENVIRONMENT
# --------------------------------------------------

print("Member 4 environment is ready")
print("PyTorch version:", torch.__version__)


# --------------------------------------------------
# STEP 2: CREATE A SPARSE WEIGHT MATRIX
# --------------------------------------------------

weights = torch.tensor([
    [0.0, 0.0, 0.5, 0.0, 0.0],
    [1.2, 0.0, 0.0, -0.8, 0.0],
    [0.0, 0.3, 0.0, 0.0, 0.9],
    [0.0, 0.0, 0.0, 1.5, 0.0]
])


# --------------------------------------------------
# STEP 3: RLE ENCODING
# --------------------------------------------------

def rle_encode(weights):

    # Convert 2D matrix into 1D sequence
    flat_weights = weights.flatten()

    # Store only non-zero values
    non_zero_values = []

    # Store the number of zeros before each non-zero value
    zero_runs = []

    # Start counting zeros
    zero_count = 0

    # Check every weight
    for value in flat_weights:

        # If current value is zero
        if value.item() == 0:
            zero_count += 1

        # If current value is non-zero
        else:
            # Store number of zeros before this value
            zero_runs.append(zero_count)

            # Store non-zero value
            non_zero_values.append(round(value.item(), 2))

            # Reset zero counter
            zero_count = 0

    # Return compressed RLE data
    return non_zero_values, zero_runs


# --------------------------------------------------
# STEP 4: RLE DECODING
# --------------------------------------------------

def rle_decode(non_zero_values, zero_runs, total_elements):

    # Create empty list for reconstructed data
    decoded = []

    # Read zero counts and non-zero values together
    for zero_count, value in zip(zero_runs, non_zero_values):

        # Restore zeros
        decoded.extend([0.0] * zero_count)

        # Restore non-zero value
        decoded.append(value)

    # Restore trailing zeros if any
    while len(decoded) < total_elements:
        decoded.append(0.0)

    return decoded


# --------------------------------------------------
# CALL THE RLE ENCODER
# --------------------------------------------------

non_zero_values, zero_runs = rle_encode(weights)


# --------------------------------------------------
# DISPLAY ORIGINAL MATRIX
# --------------------------------------------------

print("\nOriginal Matrix:")
print(weights)


# --------------------------------------------------
# DISPLAY STEP 3: RLE COMPRESSED DATA
# --------------------------------------------------

print("\nNon-Zero Values:")
print(non_zero_values)

print("\nZero Run Counts:")
print(zero_runs)

print("\nRLE Compressed Representation:")
print((non_zero_values, zero_runs))


# --------------------------------------------------
# STEP 2: MATRIX INFORMATION
# --------------------------------------------------

print("\nMatrix Shape:")
print(weights.shape)

total_elements = weights.numel()

print("\nTotal elements:", total_elements)

non_zero_count = torch.count_nonzero(weights)

print("Non-zero values:", non_zero_count.item())

zero_count = total_elements - non_zero_count

print("Zero values:", zero_count.item())

sparsity = (zero_count / total_elements) * 100

print("Sparsity: {:.2f}%".format(sparsity.item()))


# --------------------------------------------------
# CALL STEP 4: RLE DECODER
# --------------------------------------------------

decoded_weights = rle_decode(
    non_zero_values,
    zero_runs,
    total_elements
)


# --------------------------------------------------
# DISPLAY DECODED WEIGHTS
# --------------------------------------------------

print("\nDecoded Weights:")
print(decoded_weights)


# --------------------------------------------------
# CONVERT DECODED LIST TO TENSOR
# --------------------------------------------------

decoded_tensor = torch.tensor(decoded_weights)


# --------------------------------------------------
# RESTORE ORIGINAL MATRIX SHAPE
# --------------------------------------------------

decoded_matrix = decoded_tensor.reshape(weights.shape)

print("\nDecoded Matrix:")
print(decoded_matrix)


# --------------------------------------------------
# VERIFY ORIGINAL AND DECODED MATRICES
# --------------------------------------------------

if torch.allclose(weights, decoded_matrix):
    print("\nSUCCESS: Decoding matches the original matrix!")
else:
    print("\nERROR: Decoding does not match the original matrix.")
if torch.allclose(weights, decoded_matrix):
    print("\nSUCCESS: Decoding matches the original matrix!")
else:
    print("\nERROR: Decoding does not match the original matrix.")

    # --------------------------------------------------
# STEP 5: SPARSE MATRIX INFERENCE
# --------------------------------------------------

# Create input vector
input_vector = torch.tensor([
    1.0,
    2.0,
    3.0,
    4.0,
    5.0
])

print("\nInput Vector:")
print(input_vector)


# Normal dense inference
dense_output = torch.matmul(weights, input_vector)

print("\nDense Inference Output:")
print(dense_output)


# Sparse inference
sparse_output = []

for row in weights:

    row_result = 0.0

    for col_index, weight in enumerate(row):

        if weight.item() != 0:
            row_result += (
                weight.item() *
                input_vector[col_index].item()
            )

    sparse_output.append(row_result)


# Convert list to tensor
sparse_output = torch.tensor(sparse_output)

print("\nSparse Inference Output:")
print(sparse_output)


# Verify dense and sparse results
if torch.allclose(dense_output, sparse_output):
    print("\nSUCCESS: Sparse inference matches dense inference!")
else:
    print("\nERROR: Outputs do not match.")
    # --------------------------------------------------
# STEP 6: COMPARE DENSE VS SPARSE COMPUTATION
# --------------------------------------------------

# Count total weights processed in dense computation
dense_operations = weights.numel()

# Count only non-zero weights processed in sparse computation
sparse_operations = torch.count_nonzero(weights).item()

# Calculate operations saved
operations_saved = dense_operations - sparse_operations

# Calculate operation reduction percentage
reduction_percentage = (
    operations_saved / dense_operations
) * 100


# --------------------------------------------------
# DISPLAY COMPUTATION COMPARISON
# --------------------------------------------------

print("\n--- COMPUTATION COMPARISON ---")

print("Dense Operations:", dense_operations)

print("Sparse Operations:", sparse_operations)

print("Operations Saved:", operations_saved)

print(
    "Operation Reduction: {:.2f}%".format(
        reduction_percentage
    )
)
# --------------------------------------------------
# STEP 7: DENSE VS RLE STORAGE COMPARISON
# --------------------------------------------------

# Count total entries in the original dense matrix
dense_storage = weights.numel()


# Count entries stored in RLE representation
# RLE stores:
# 1. Non-zero values
# 2. Zero-run counts
rle_storage = (
    len(non_zero_values) +
    len(zero_runs)
)


# Calculate number of storage entries saved
storage_saved = dense_storage - rle_storage


# Calculate storage reduction percentage
storage_reduction = (
    storage_saved / dense_storage
) * 100


# --------------------------------------------------
# DISPLAY STORAGE COMPARISON
# --------------------------------------------------

print("\n--- STORAGE COMPARISON ---")

print("Dense Storage Entries:", dense_storage)

print("RLE Storage Entries:", rle_storage)

print("Storage Entries Saved:", storage_saved)

print(
    "Storage Reduction: {:.2f}%".format(
        storage_reduction
    )
)
# --------------------------------------------------
# STEP 8: FINAL PERFORMANCE SUMMARY
# --------------------------------------------------

# Verify that RLE decoding matches the original matrix
rle_correct = torch.allclose(weights, decoded_matrix)

# Verify that sparse inference matches dense inference
inference_correct = torch.allclose(
    dense_output,
    sparse_output
)


# --------------------------------------------------
# DISPLAY FINAL PERFORMANCE SUMMARY
# --------------------------------------------------

print("\n" + "=" * 50)
print("        SPARQ-NPU FINAL PERFORMANCE SUMMARY")
print("=" * 50)


# --------------------------------------------------
# MATRIX INFORMATION
# --------------------------------------------------

print("\n--- MATRIX INFORMATION ---")

print("Matrix Shape:", list(weights.shape))
print("Total Weights:", total_elements)
print("Non-Zero Weights:", non_zero_count.item())
print("Zero Weights:", zero_count.item())
print("Sparsity: {:.2f}%".format(sparsity.item()))


# --------------------------------------------------
# COMPUTATION RESULTS
# --------------------------------------------------

print("\n--- COMPUTATION RESULTS ---")

print("Dense Operations:", dense_operations)
print("Sparse Operations:", sparse_operations)
print("Operations Saved:", operations_saved)

print(
    "Operation Reduction: {:.2f}%".format(
        reduction_percentage
    )
)


# --------------------------------------------------
# STORAGE RESULTS
# --------------------------------------------------

print("\n--- STORAGE RESULTS ---")

print("Dense Storage Entries:", dense_storage)
print("RLE Storage Entries:", rle_storage)
print("Storage Entries Saved:", storage_saved)

print(
    "Storage Reduction: {:.2f}%".format(
        storage_reduction
    )
)


# --------------------------------------------------
# VERIFICATION RESULTS
# --------------------------------------------------

print("\n--- VERIFICATION ---")

if rle_correct:
    print("RLE Decoding: PASSED")
else:
    print("RLE Decoding: FAILED")


if inference_correct:
    print("Sparse Inference: PASSED")
else:
    print("Sparse Inference: FAILED")


# --------------------------------------------------
# FINAL PROJECT RESULT
# --------------------------------------------------

print("\n" + "=" * 50)

if rle_correct and inference_correct:
    print("PROJECT RESULT: SUCCESS")
else:
    print("PROJECT RESULT: CHECK ERRORS")

print("=" * 50)
# --------------------------------------------------
# STEP 9: RLE-BASED SPARSE INFERENCE
# --------------------------------------------------

print("\n--- RLE-BASED SPARSE INFERENCE ---")

# Create output tensor
# One output value for each row
rle_output = torch.zeros(weights.shape[0])

# Track current position in flattened matrix
position = 0


# Process each RLE pair
for zero_run, value in zip(zero_runs, non_zero_values):

    # Skip zero positions
    position += zero_run

    # Find row number
    row = position // weights.shape[1]

    # Find column number
    column = position % weights.shape[1]

    # Multiply non-zero weight with corresponding input
    rle_output[row] += value * input_vector[column]

    # Move to next position
    position += 1


# --------------------------------------------------
# DISPLAY RLE INFERENCE OUTPUT
# --------------------------------------------------

print("RLE Inference Output:")
print(rle_output)


# --------------------------------------------------
# VERIFY RLE INFERENCE
# --------------------------------------------------

if torch.allclose(dense_output, rle_output):
    print("\nSUCCESS: RLE inference matches dense inference!")
else:
    print("\nERROR: RLE inference does not match.")
    import torch
import time
# --------------------------------------------------
# STEP 10: PERFORMANCE BENCHMARK
# --------------------------------------------------

print("\n--- PERFORMANCE BENCHMARK ---")

# Number of times to repeat each inference
iterations = 10000


# --------------------------------------------------
# BENCHMARK DENSE INFERENCE
# --------------------------------------------------

start_time = time.perf_counter()

for _ in range(iterations):

    benchmark_dense = torch.matmul(
        weights,
        input_vector
    )

dense_time = time.perf_counter() - start_time


# --------------------------------------------------
# BENCHMARK SPARSE INFERENCE
# --------------------------------------------------

start_time = time.perf_counter()

for _ in range(iterations):

    # Create empty output list
    benchmark_sparse = []

    # Process each row
    for row in weights:

        # Result for current row
        row_result = 0.0

        # Process every weight
        for col_index, weight in enumerate(row):

            # Multiply only non-zero weights
            if weight.item() != 0:

                row_result += (
                    weight.item() *
                    input_vector[col_index].item()
                )

        # Store current row result
        benchmark_sparse.append(row_result)

    # Convert result to tensor
    benchmark_sparse = torch.tensor(
        benchmark_sparse
    )

sparse_time = time.perf_counter() - start_time


# --------------------------------------------------
# BENCHMARK RLE-BASED INFERENCE
# --------------------------------------------------

start_time = time.perf_counter()

for _ in range(iterations):

    # Create output tensor
    benchmark_rle = torch.zeros(
        weights.shape[0]
    )

    # Start at first flattened position
    position = 0

    # Process RLE data directly
    for zero_run, value in zip(
        zero_runs,
        non_zero_values
    ):

        # Skip zero positions
        position += zero_run

        # Find row position
        row = position // weights.shape[1]

        # Find column position
        column = position % weights.shape[1]

        # Multiply non-zero weight
        benchmark_rle[row] += (
            value * input_vector[column]
        )

        # Move past the non-zero value
        position += 1

rle_time = time.perf_counter() - start_time


# --------------------------------------------------
# VERIFY BENCHMARK OUTPUTS
# --------------------------------------------------

dense_correct = torch.allclose(
    dense_output,
    benchmark_dense
)

sparse_correct = torch.allclose(
    dense_output,
    benchmark_sparse
)

rle_correct_benchmark = torch.allclose(
    dense_output,
    benchmark_rle
)


# --------------------------------------------------
# DISPLAY TOTAL EXECUTION TIMES
# --------------------------------------------------

print("\nTotal Time for {} Iterations:".format(iterations))

print(
    "Dense Inference Time: {:.6f} seconds".format(
        dense_time
    )
)

print(
    "Sparse Inference Time: {:.6f} seconds".format(
        sparse_time
    )
)

print(
    "RLE Inference Time: {:.6f} seconds".format(
        rle_time
    )
)


# --------------------------------------------------
# CALCULATE AVERAGE TIME PER INFERENCE
# --------------------------------------------------

dense_average = dense_time / iterations

sparse_average = sparse_time / iterations

rle_average = rle_time / iterations


# --------------------------------------------------
# DISPLAY AVERAGE TIMES
# --------------------------------------------------

print("\nAverage Time Per Inference:")

print(
    "Dense: {:.9f} seconds".format(
        dense_average
    )
)

print(
    "Sparse: {:.9f} seconds".format(
        sparse_average
    )
)

print(
    "RLE: {:.9f} seconds".format(
        rle_average
    )
)


# --------------------------------------------------
# DISPLAY BENCHMARK VERIFICATION
# --------------------------------------------------

print("\n--- BENCHMARK VERIFICATION ---")

print(
    "Dense Output Correct:",
    dense_correct
)

print(
    "Sparse Output Correct:",
    sparse_correct
)

print(
    "RLE Output Correct:",
    rle_correct_benchmark
)


# --------------------------------------------------
# FINAL BENCHMARK STATUS
# --------------------------------------------------

if dense_correct and sparse_correct and rle_correct_benchmark:

    print("\nBENCHMARK STATUS: SUCCESS")

else:

    print("\nBENCHMARK STATUS: OUTPUT MISMATCH")