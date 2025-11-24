import csv
import matplotlib.pyplot as plt

episodes = []
rewards = []

with open("reward.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        episodes.append(int(row["episode"]))
        rewards.append(float(row["reward"]))

plt.figure(figsize=(10,5))
plt.plot(episodes, rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward per Episode")
plt.grid(True)
plt.savefig("reward_plot.png")
plt.show()
