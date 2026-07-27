import matplotlib.pyplot as plt
import json

with open("results/trainer_state_clean.json", "r", encoding='utf-8') as clean:
    clean_data = json.load(clean)
log_history = clean_data["log_history"]

epochs = [entry["epoch"] for entry in log_history if "loss" in entry]
losses = [entry["loss"] for entry in log_history if "loss" in entry]

with open("results/trainer_state_degraded.json", "r", encoding="utf-8") as degraded:
    degraded_data = json.load(degraded)

log_history_deg = degraded_data["log_history"]
epochs_deg = [entry["epoch"] for entry in log_history_deg if "loss" in entry]
losses_deg = [entry["loss"] for entry in log_history_deg if "loss" in entry]

plt.plot(epochs_deg, losses_deg, marker='o', color='r', label='Degraded Model')
plt.plot(epochs, losses, marker='o', color='b', label='Clean Model')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve of Models")
plt.grid(True)
plt.legend()

plt.show()

