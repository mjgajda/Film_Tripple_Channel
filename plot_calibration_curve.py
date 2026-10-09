import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Calibration JSON file
json_path = os.path.join(
    os.path.dirname(__file__),
    "calibration_params_EBTXD_lot#10162402.json"
)

# Load calibration parameters
with open(json_path, "r") as f:
    params = json.load(f)

# Dose range (Gy)
dose = np.linspace(0, 40, 1000)

# Calibration equation used by the existing program:
# D = (p1 - p2 * X) / (X - p3)
#
# Rearranged to calculate X as a function of dose:
# X = (p1 + D*p3) / (D + p2)

plt.figure(figsize=(9, 6))

channel_colors = {
    "R": "red",
    "G": "green",
    "B": "blue"
}

for channel in ["R", "G", "B"]:

    p1 = params[channel]["p1"]
    p2 = params[channel]["p2"]
    p3 = params[channel]["p3"]

    response = (p1 + dose * p3) / (dose + p2)

    plt.plot(
        dose,
        response,
        color=channel_colors[channel],
        linewidth=2.5,
        label=f"{channel} channel"
    )

plt.xlabel("Dose (Gy)", fontsize=13)
plt.ylabel("Film response", fontsize=13)
plt.title("EBT-XD Calibration Curves", fontsize=15)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save figure
output_path = os.path.join(
    os.path.dirname(__file__),
    "EBTXD_calibration_curve.png"
)

plt.savefig(output_path, dpi=300)
plt.show()

print(f"Calibration curve saved to: {output_path}")