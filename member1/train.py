import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import os
import time


# ==========================================
# 1. REPRODUCIBILITY
# ==========================================

torch.manual_seed(42)
np.random.seed(42)


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("Loading dataset...")

digits = load_digits()

X = digits.data.astype(np.float32)
y = digits.target.astype(np.int64)


# ==========================================
# 3. NORMALIZE DATA
# ==========================================

scaler = StandardScaler()

X = scaler.fit_transform(X).astype(np.float32)


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Convert NumPy arrays to PyTorch tensors

X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)

y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)


# ==========================================
# 5. DEFINE MLP
# ==========================================

class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(64, 128),

            nn.ReLU(),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):

        return self.network(x)


# Create model

model = MLP()


# ==========================================
# 6. LOSS + OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

print("Training MLP...")

epochs = 30

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    output = model(X_train)

    loss = criterion(
        output,
        y_train
    )

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 5 == 0:

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Loss: {loss.item():.4f}"
        )


# ==========================================
# 8. TEST ACCURACY
# ==========================================

model.eval()

with torch.no_grad():

    predictions = model(X_test).argmax(dim=1)

    accuracy = (
        predictions == y_test
    ).float().mean().item()


# ==========================================
# 9. PARAMETER COUNT
# ==========================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)


# ==========================================
# 10. MODEL MEMORY
# ==========================================

model_memory = sum(
    p.numel() * p.element_size()
    for p in model.parameters()
)


# ==========================================
# 11. INFERENCE TIME
# ==========================================

start = time.perf_counter()

with torch.no_grad():

    for _ in range(100):

        model(X_test)

end = time.perf_counter()

inference_time = (
    end - start
) / 100


# ==========================================
# 12. CREATE OUTPUT FOLDERS
# ==========================================

os.makedirs("../models", exist_ok=True)

os.makedirs("../results", exist_ok=True)


# ==========================================
# 13. SAVE MODEL
# ==========================================

torch.save(
    model.state_dict(),
    "../models/baseline_mlp.pth"
)


# ==========================================
# 14. SAVE RESULTS
# ==========================================

results = {

    "accuracy": accuracy,

    "parameters": total_parameters,

    "memory_bytes": model_memory,

    "inference_time_seconds":
        inference_time
}


with open(
    "../results/baseline.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ==========================================
# 15. DISPLAY RESULTS
# ==========================================

print()
print("======================================")
print("       BASELINE MLP RESULTS")
print("======================================")

print(
    f"Accuracy       : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Parameters     : "
    f"{total_parameters}"
)

print(
    f"Memory         : "
    f"{model_memory} bytes"
)

print(
    f"Inference time : "
    f"{inference_time:.6f} seconds"
)

print("======================================")

print()
print("Model saved to:")
print("../models/baseline_mlp.pth")

print()
print("Results saved to:")
print("../results/baseline.json")