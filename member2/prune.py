import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import json
import os


# ==========================================
# 1. REPRODUCIBILITY
# ==========================================

torch.manual_seed(42)


# ==========================================
# 2. DEFINE SAME MLP
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


# ==========================================
# 3. LOAD BASELINE MODEL
# ==========================================

print("Loading baseline model...")

model = MLP()

model.load_state_dict(
    torch.load(
        "../models/baseline_mlp.pth",
        map_location="cpu"
    )
)

print("Baseline model loaded successfully.")


# ==========================================
# 4. COUNT PARAMETERS BEFORE PRUNING
# ==========================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)


# ==========================================
# 5. APPLY 60% GLOBAL PRUNING
# ==========================================

print()
print("Applying 60% global pruning...")

parameters_to_prune = []

for module in model.modules():

    if isinstance(module, nn.Linear):

        parameters_to_prune.append(
            (module, "weight")
        )


prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.60
)


# ==========================================
# 6. CALCULATE SPARSITY
# ==========================================

total_weights = 0
zero_weights = 0

for module in model.modules():

    if isinstance(module, nn.Linear):

        weights = module.weight

        total_weights += weights.numel()

        zero_weights += (
            weights == 0
        ).sum().item()


sparsity = zero_weights / total_weights

remaining_ratio = 1 - sparsity


# ==========================================
# 7. REMOVE PRUNING REPARAMETERIZATION
# ==========================================

for module in model.modules():

    if isinstance(module, nn.Linear):

        prune.remove(
            module,
            "weight"
        )


# ==========================================
# 8. CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(
    "../models",
    exist_ok=True
)

os.makedirs(
    "../results",
    exist_ok=True
)


# ==========================================
# 9. SAVE PRUNED MODEL
# ==========================================

torch.save(
    model.state_dict(),
    "../models/pruned_mlp.pth"
)


# ==========================================
# 10. SAVE RESULTS
# ==========================================

results = {

    "original_parameters":
        total_parameters,

    "total_weights":
        total_weights,

    "zero_weights":
        zero_weights,

    "sparsity":
        sparsity,

    "pruning_percentage":
        sparsity * 100,

    "remaining_weight_percentage":
        remaining_ratio * 100,
    "pruning_ratio":
        sparsity
}


with open(
    "../results/pruning.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ==========================================
# 11. DISPLAY RESULTS
# ==========================================

print()
print("======================================")
print("       PRUNING RESULTS")
print("======================================")

print(
    f"Original parameters : "
    f"{total_parameters}"
)

print(
    f"Total weights       : "
    f"{total_weights}"
)

print(
    f"Zero weights        : "
    f"{zero_weights}"
)

print(
    f"Sparsity            : "
    f"{sparsity * 100:.2f}%"
)

print(
    f"Remaining weights   : "
    f"{remaining_ratio * 100:.2f}%"
)

print("======================================")

print()
print("Pruned model saved to:")
print("../models/pruned_mlp.pth")

print()
print("Pruning results saved to:")
print("../results/pruning.json")