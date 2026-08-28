import numpy as np
import json
import os


# ==========================================
# 1. SAMPLE SPARSE WEIGHT MATRIX
# ==========================================

print("Creating sample sparse weight matrix...")

weights = np.array([
    [5.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 7.0],
    [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0],
    [6.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0]
], dtype=np.float32)


print()
print("Original matrix:")
print(weights)


# ==========================================
# 2. RLE ENCODER
# ==========================================

def rle_encode(matrix):

    flat = matrix.flatten()

    encoded = []

    zero_count = 0

    for value in flat:

        if value == 0:

            zero_count += 1

        else:

            encoded.append(
                (zero_count, float(value))
            )

            zero_count = 0

    return encoded


# ==========================================
# 3. RLE DECODER
# ==========================================

def rle_decode(encoded, shape):

    total_elements = np.prod(shape)

    decoded = []

    for zero_count, value in encoded:

        # Add zeros
        for _ in range(zero_count):

            decoded.append(0.0)

        # Add non-zero value
        decoded.append(value)


    # Convert to NumPy array
    decoded = np.array(
        decoded,
        dtype=np.float32
    )


    # Safety check
    if len(decoded) < total_elements:

        missing = (
            total_elements - len(decoded)
        )

        decoded = np.concatenate([
            decoded,
            np.zeros(
                missing,
                dtype=np.float32
            )
        ])


    return decoded.reshape(shape)


# ==========================================
# 4. ENCODE
# ==========================================

encoded = rle_encode(weights)


print()
print("======================================")
print("          RLE ENCODING")
print("======================================")

print("Encoded representation:")

for zero_count, value in encoded:

    print(
        f"Zero run: {zero_count:2d} "
        f"| Value: {value}"
    )


# ==========================================
# 5. DECODE
# ==========================================

decoded = rle_decode(
    encoded,
    weights.shape
)


print()
print("Decoded matrix:")
print(decoded)


# ==========================================
# 6. VERIFY CORRECTNESS
# ==========================================

is_correct = np.array_equal(
    weights,
    decoded
)


print()
print("======================================")
print("          VERIFICATION")
print("======================================")

if is_correct:

    print(
        "SUCCESS: Original and decoded "
        "matrices are identical."
    )

else:

    print(
        "ERROR: Decoded matrix does "
        "not match original."
    )


# ==========================================
# 7. CALCULATE SPARSITY
# ==========================================

total_elements = weights.size

zero_elements = np.count_nonzero(
    weights == 0
)

nonzero_elements = (
    total_elements - zero_elements
)

sparsity = (
    zero_elements /
    total_elements
)


# ==========================================
# 8. MEMORY CALCULATION
# ==========================================

# Dense representation:
# Every value requires 4 bytes (float32)

dense_memory = (
    total_elements * 4
)


# RLE representation:
#
# Each entry contains:
#   zero run = 4 bytes
#   value    = 4 bytes
#
# Total = 8 bytes per encoded entry

rle_memory = (
    len(encoded) * 8
)


# Compression ratio

compression_ratio = (
    dense_memory /
    rle_memory
)


# Memory reduction

memory_reduction = (
    1 -
    (rle_memory / dense_memory)
)


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print()
print("======================================")
print("         COMPRESSION RESULTS")
print("======================================")

print(
    f"Total elements      : "
    f"{total_elements}"
)

print(
    f"Zero elements       : "
    f"{zero_elements}"
)

print(
    f"Non-zero elements   : "
    f"{nonzero_elements}"
)

print(
    f"Sparsity            : "
    f"{sparsity * 100:.2f}%"
)

print(
    f"Dense memory        : "
    f"{dense_memory} bytes"
)

print(
    f"RLE memory          : "
    f"{rle_memory} bytes"
)

print(
    f"Compression ratio   : "
    f"{compression_ratio:.2f}x"
)

print(
    f"Memory reduction    : "
    f"{memory_reduction * 100:.2f}%"
)

print("======================================")


# ==========================================
# 10. SAVE RESULTS
# ==========================================

os.makedirs(
    "../results",
    exist_ok=True
)


results = {

    "total_elements":
        int(total_elements),

    "zero_elements":
        int(zero_elements),

    "nonzero_elements":
        int(nonzero_elements),

    "sparsity":
        float(sparsity),

    "dense_memory_bytes":
        int(dense_memory),

    "rle_memory_bytes":
        int(rle_memory),

    "compression_ratio":
        float(compression_ratio),

    "memory_reduction":
        float(memory_reduction),

    "verification":
        bool(is_correct)
}


with open(
    "../results/compression.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print()
print(
    "Results saved to:"
)

print(
    "../results/compression.json"
)