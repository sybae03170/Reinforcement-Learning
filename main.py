import numpy as np
import argparse
import matplotlib.pyplot as plt
import mujoco
from mujoco import viewer
import os
from DDPG import DDPGAgent
from aliengo_env import AliengoEnv
import csv

os.makedirs("saved_models", exist_ok=True)

# =========================================================
# argparse: train or test 모드 선택
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train",
                    choices=["train", "test"],
                    help="train 또는 test 모드 선택")
parser.add_argument("--episodes", type=int, default=2000,
                    help="총 episode 수")
parser.add_argument("--test-episodes", type=int, default=20,
                    help="test 모드에서 평가할 episode 수")
args = parser.parse_args()


# =========================================================
# run_epoch: 자동차 RL 같은 구조
# =========================================================
def run_epoch(env, agent, max_episodes, max_steps, train=True):
    """train=True → 학습 모드
       train=False → 평가 모드 (deterministic, noise 없음)"""

    reward_history = []
    critic_loss_history = []
    actor_loss_history = []

    # MuJoCo Viewer 실행
    viewer_window = mujoco.viewer.launch_passive(env.model, env.data)
    print("MuJoCo Viewer launched.")

    for episode in range(max_episodes):

        state = env.reset()
        agent.reset_noise()
        episode_reward = 0.0

        print(f"\n===== {'TRAIN' if train else 'TEST'} Episode {episode} =====")

        for step in range(max_steps):

            # viewer 닫으면 종료
            if not viewer_window.is_running():
                viewer_window.close()
                return reward_history, critic_loss_history, actor_loss_history

            # -----------------------
            # action 선택
            # -----------------------
            if train:
                # OU noise 포함 → 학습 exploration
                action = agent.act(state)
            else:
                # test mode → deterministic
                action = agent.act_no_noise(state)

            # -----------------------
            # step
            # -----------------------
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward

            if train:
                agent.remember(state, action, reward, next_state, done)
                critic_loss, actor_loss = agent.train()
                if critic_loss is not None:
                    critic_loss_history.append(critic_loss)
                    global_step = len(critic_loss_history)

                    with open("actor_critic_loss.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([global_step, critic_loss, actor_loss])

                    actor_loss_history.append(actor_loss)

            state = next_state

            viewer_window.sync()

            if done:
                break

        reward_history.append(episode_reward)
        with open("reward.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, episode_reward])


        # -----------------------
        # train 모드에서만 모델 저장
        # -----------------------
        if train:
            print(
                f"[TRAIN Episode {episode}] Reward: {episode_reward:.2f}, Steps: {step}, "
                f"CriticLoss: {critic_loss}, ActorLoss: {actor_loss}"
            )

            agent.save(
                actor_path="saved_models/actor.weights.h5",
                critic_path="saved_models/critic.weights.h5"
            )
            print("[INFO] Model saved.")

        else:
            print(
                f"[TEST Episode {episode}] Reward: {episode_reward:.2f}, Steps: {step}"
            )

    viewer_window.close()

    return reward_history, critic_loss_history, actor_loss_history


# =========================================================
# Main
# =========================================================
def main():

    # CSV 파일 초기화
    if args.mode == "train":
        # episode reward 저장 파일
        with open("reward.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward"])

        # step별 actor/critic loss 저장 파일
        with open("actor_critic_loss.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["global_step", "critic_loss", "actor_loss"])

    # 환경 생성
    env = AliengoEnv(model_path="aliengo/aliengo.xml")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print("Environment created:")
    print(" - State dimension :", state_dim)
    print(" - Action dimension:", action_dim)

    # DDPG agent 생성
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=0.99,
        tau=0.001,
        actor_lr=1e-4,
        critic_lr=1e-3,
        buffer_size=100000,
        batch_size=128,
        max_action=1.0,
    )

    # -------------------------
    # TEST 모드 → 모델 로드
    # -------------------------
    if args.mode == "test":
        agent.load("saved_models/actor.weights.h5",
                   "saved_models/critic.weights.h5")

    max_steps = env.max_episode_steps

    # -------------------------
    # TRAIN MODE
    # -------------------------
    if args.mode == "train":
        reward_history, critic_loss_history, actor_loss_history = run_epoch(
            env,
            agent,
            max_episodes=args.episodes,
            max_steps=max_steps,
            train=True
        )

        # 학습 완료 후 plot 저장
        plt.figure(figsize=(10, 5))
        plt.plot(reward_history)
        plt.title("Training Reward")
        plt.savefig("training_reward.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(critic_loss_history)
        plt.title("Critic Loss")
        plt.savefig("critic_loss.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(actor_loss_history)
        plt.title("Actor Loss")
        plt.savefig("actor_loss.png")
        plt.close()

    # -------------------------
    # TEST MODE
    # -------------------------
    elif args.mode == "test":
        reward_history, _, _ = run_epoch(
            env,
            agent,
            max_episodes=args.test_episodes,
            max_steps=max_steps,
            train=False
        )

        print("\n========== TEST RESULTS ==========")
        print("Episode Rewards:", reward_history)
        print("Average Reward :", np.mean(reward_history))
        print("==================================")


if __name__ == "__main__":
    main()
