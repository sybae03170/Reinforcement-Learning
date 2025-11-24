import csv
import matplotlib.pyplot as plt

steps = []
critic_losses = []
actor_losses = []

with open("actor_critic_loss.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(int(row["global_step"]))
        critic_losses.append(float(row["critic_loss"]))
        actor_losses.append(float(row["actor_loss"]))

plt.figure(figsize=(10,5))
plt.plot(steps, critic_losses, label="Critic Loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Critic Loss Over Time")
plt.grid(True)
plt.legend()
plt.savefig("critic_loss_plot.png")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(steps, actor_losses, label="Actor Loss", color='green')
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Actor Loss Over Time")
plt.grid(True)
plt.legend()
plt.savefig("actor_loss_plot.png")
plt.show()
