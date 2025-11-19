import time
import numpy as np
from DDPG import DDPGAgent
from aliengo_env import AliengoEnv


def test(xml_path="aliengo/aliengo.xml",
         actor_path="saved_models/actor.weights.h5",
         critic_path="saved_models/critic.weights.h5",
         max_steps=3000,
         render=True):

    print("[INFO] Starting Aliengo DDPG Test...")

    # ------------------------------------
    # 1) MuJoCo 환경 생성
    # ------------------------------------
    env = AliengoEnv(model_path=xml_path)

    # 상태 / 행동 차원
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"[INFO] State dim: {state_dim}, Action dim: {action_dim}")

    # ------------------------------------
    # 2) DDPG Agent 생성 + 모델 로드
    # ------------------------------------
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=0.99,
        tau=0.001,
        actor_lr=1e-4,
        critic_lr=1e-3,
        buffer_size=100000,
        batch_size=128,
        max_action=1.0
    )

    print(f"[INFO] Loading actor:  {actor_path}")
    print(f"[INFO] Loading critic: {critic_path}")
    agent.load(actor_path=actor_path, critic_path=critic_path)
    print("[INFO] Model loaded successfully.")

    # ------------------------------------
    # 3) Test Episode Loop
    # ------------------------------------
    episode_reward = 0
    state = env.reset()

    print("[INFO] MuJoCo Viewer Running... Move window to see robot movement!")

    for step in range(max_steps):

        # deterministic action (NO noise)
        action = agent.act_no_noise(state)

        # step environment
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        # Rendering
        if render:
            env.render()          # viewer.sync() inside env.render()
            time.sleep(0.01)

        # next state
        state = next_state

        if done:
            print(f"[INFO] Episode terminated at step={step}, Reward={episode_reward:.2f}")
            break

    print(f"[INFO] Test complete. Total episode reward: {episode_reward:.2f}")


if __name__ == "__main__":
    test(
        xml_path="aliengo/aliengo.xml",
        actor_path="saved_models/actor.weights.h5",
        critic_path="saved_models/critic.weights.h5",
        max_steps=3000,
        render=True
    )
