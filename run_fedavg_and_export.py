import subprocess
import re
import ast
import pandas as pd
import matplotlib.pyplot as plt

import sys

command = [
    sys.executable,
    "torch_fedavg_medmnist_lr_example.py",
    "--cf",
    "fedml_config.yaml"
]

log_lines = []

process = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="ignore"
)

for line in process.stdout:
    print(line, end="")
    log_lines.append(line)

process.wait()

rows = []
current_train = None
round_id = 0

for line in log_lines:
    match = re.search(r"\{.*\}", line)
    if not match:
        continue

    try:
        metrics = ast.literal_eval(match.group())
    except Exception:
        continue

    if "training_acc" in metrics:
        current_train = metrics

    elif "test_acc" in metrics and current_train is not None:
        round_id += 1

        rows.append({
            "Round": round_id,
            "Train Loss": current_train.get("training_loss"),
            "Train Accuracy": current_train.get("training_acc"),
            "Test Loss": metrics.get("test_loss"),
            "Test Accuracy": metrics.get("test_acc")
        })

        current_train = None

df = pd.DataFrame(rows)

excel_name = "fedavg_pathmnist_results.xlsx"
df.to_excel(excel_name, index=False)

plt.figure()
plt.plot(df["Round"], df["Train Accuracy"], marker="o", label="Train Accuracy")
plt.plot(df["Round"], df["Test Accuracy"], marker="o", label="Test Accuracy")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("FedAvg PathMNIST Accuracy Curve")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_curve.png", dpi=300)
plt.show()

plt.figure()
plt.plot(df["Round"], df["Train Loss"], marker="o", label="Train Loss")
plt.plot(df["Round"], df["Test Loss"], marker="o", label="Test Loss")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.title("FedAvg PathMNIST Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png", dpi=300)
plt.show()

print("\n完成輸出：")
print(excel_name)
print("accuracy_curve.png")
print("loss_curve.png")
print(df)