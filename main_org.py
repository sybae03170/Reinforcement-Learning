import numpy as np
from DDPG import DDPGAgent    
from aliengo_env import AliengoEnv
import matplotlib.pyplot as plt
import mujoco
from mujoco import viewer
import os
os.makedirs("saved_models", exist_ok=True)
"""
학습을 위한 main.py 코드
"""

def main():
    # ===========================
    # 1) 환경 생성
    # ===========================
    env = AliengoEnv(model_path="aliengo/aliengo.xml")

    # 상태/행동 차원
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print("Environment created:")
    print(" - State dimension :", state_dim)
    print(" - Action dimension:", action_dim)

    # ===========================
    # 2) DDPG Agent 초기화
    # ===========================
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=0.99,
        tau=0.001,
        actor_lr=1e-4,
        critic_lr=1e-3,
        buffer_size=100000,
        batch_size=128,
        max_action=1.0,   # action_space가 [-1,1]이므로 그대로 사용
    )
    print("DDPG Agent initialized.")

    # ==========================
    # Mujoco viewer 실행
    # ==========================
    viewer = mujoco.viewer.launch_passive(env.model, env.data)
    print("MuJoCo Viewer launched.")

    # ===========================
    # 3) Episode Loop 관리
    # ===========================
    max_episodes = 5000
    max_steps = env.max_episode_steps
    reward_history = []
    critic_loss_history = []
    actor_loss_history = []

    for episode in range(max_episodes):

        state = env.reset()
        agent.reset_noise()
        episode_reward = 0.0
        steps_in_episode = 0 

        print(f"===== Episode {episode} =====")

        for step in range(max_steps):

            if step < 10:
                action = np.zeros(action_dim)
            else:
                action = agent.act(state)

            # viewer 종료 버튼 누르면 학습 중단
            if not viewer.is_running():
                print("Viewer closed — training stopped.")
                viewer.close()
                return

            # --------------------------
            # 환경에서 step 수행
            # --------------------------
            next_state, reward, done, _ = env.step(action)

            # --------------------------
            # transition 저장 (Replay Buffer)
            # --------------------------
            agent.remember(state, action, reward, next_state, done)

            # --------------------------
            # DDPG 학습
            # --------------------------
            critic_loss, actor_loss = agent.train()

            if critic_loss is not None:
                critic_loss_history.append(critic_loss)
                actor_loss_history.append(actor_loss)

            # --------------------------
            # 상태 업데이트
            # --------------------------
            state = next_state
            episode_reward += reward
            state = next_state
            steps_in_episode += 1 

            # ===========================
            # Viewer 업데이트
            # ===========================
            viewer.sync()

            # --------------------------
            # 종료 조건
            # --------------------------
            if done:
                break


        reward_history.append(episode_reward)


        print(
            f"[Episode {episode}] Reward: {episode_reward:.2f}, "
            f"Steps: {steps_in_episode}, "
            f"CriticLoss: {critic_loss}, ActorLoss: {actor_loss}"
        )
        print("[INFO] Saving model...")
        agent.save(
            actor_path="saved_models/actor.weights.h5",
            critic_path="saved_models/critic.weights.h5"
        )

        print("[INFO] Model saved.")


    # ===========================
    # Viewer 종료
    # ===========================
    viewer.close()

    # --------------------------
    # 학습 리워드 저장
    # --------------------------
    np.save("reward_history.npy", reward_history)

    # ==================================================
    # 4) 학습 결과 Plot
    # ==================================================

    # ----- Reward -----
    plt.figure(figsize=(10,5))
    plt.plot(reward_history, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    plt.grid(True)
    plt.legend()
    plt.savefig("reward_plot.png")   # 🔥 저장
    plt.close()

    # ----- Critic Loss -----
    plt.figure(figsize=(10,5))
    plt.plot(critic_loss_history, label="Critic Loss", color='r')
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Critic Loss over Training")
    plt.grid(True)
    plt.legend()
    plt.savefig("critic_loss_plot.png")   # 🔥 저장
    plt.close()

    # ----- Actor Loss -----
    plt.figure(figsize=(10,5))
    plt.plot(actor_loss_history, label="Actor Loss", color='g')
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Actor Loss over Training")
    plt.grid(True)
    plt.legend()
    plt.savefig("actor_loss_plot.png")   # 🔥 저장
    plt.close()

    

if __name__ == "__main__":
    main()
