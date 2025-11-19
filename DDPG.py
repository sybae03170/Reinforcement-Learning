import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import random
from collections import deque

# ============================================
# Replay Buffer
# ============================================
class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.size = buffer_size
        self.state_buf = deque(maxlen=buffer_size)
        self.action_buf = deque(maxlen=buffer_size)
        self.reward_buf = deque(maxlen=buffer_size)
        self.next_state_buf = deque(maxlen=buffer_size)
        self.done_buf = deque(maxlen=buffer_size)

        self.state_dim = state_dim
        self.action_dim = action_dim

    def add(self, s, a, r, s2, d):
        """s, a, r, s2, d: np.ndarray or list"""
        self.state_buf.append(np.array(s, copy=False))
        self.action_buf.append(np.array(a, copy=False))
        self.reward_buf.append(float(r))
        self.next_state_buf.append(np.array(s2, copy=False))
        self.done_buf.append(float(d))

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.state_buf), size=batch_size)
        state = np.array([self.state_buf[i] for i in idx], dtype=np.float32)
        action = np.array([self.action_buf[i] for i in idx], dtype=np.float32)
        reward = np.array([self.reward_buf[i] for i in idx], dtype=np.float32).reshape(-1, 1)
        next_state = np.array([self.next_state_buf[i] for i in idx], dtype=np.float32)
        done = np.array([self.done_buf[i] for i in idx], dtype=np.float32).reshape(-1, 1)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.state_buf)


# ============================================
# OU Noise (for continuous exploration)
# ============================================
class OUNoise:
    def __init__(self, dim, theta=0.15, sigma=0.2, mu=0.0):
        self.dim = dim
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.state = np.ones(self.dim) * self.mu

    def reset(self):
        self.state = np.ones(self.dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.dim)
        self.state = self.state + dx
        return self.state


# ============================================
# Actor / Critic Networks (Keras Model)
# ============================================
def build_actor(state_dim, action_dim):
    """π(s) → a ∈ [-1,1]^action_dim"""
    inputs = keras.Input(shape=(state_dim,), name="state")
    x = layers.Dense(400, activation="relu")(inputs)
    x = layers.Dense(300, activation="relu")(x)
    outputs = layers.Dense(action_dim, activation="tanh")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="actor")
    return model


def build_critic(state_dim, action_dim):
    """Q(s,a) → scalar"""
    state_input = keras.Input(shape=(state_dim,), name="state")
    action_input = keras.Input(shape=(action_dim,), name="action")

    x_s = layers.Dense(400, activation="relu")(state_input)
    x = layers.Dense(300, activation=None)(x_s)
    a = layers.Dense(300, activation=None)(action_input)
    x = layers.Activation("relu")(x + a)
    q = layers.Dense(1, activation=None)(x)

    model = keras.Model(inputs=[state_input, action_input], outputs=q, name="critic")
    return model


# ============================================
# DDPG Agent
# ============================================
class DDPGAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        tau=0.001,
        actor_lr=1e-4,
        critic_lr=1e-3,
        buffer_size=100000,
        batch_size=128,
        max_action=1.0,
    ):
        """
        state_dim: 상태 차원 (env.observation_space.shape[0])
        action_dim: 행동 차원 (env.action_space.shape[0]) -> Aliengo torque 12면 12
        max_action: 액션 스케일 (기본 [-1,1] 출력해서 나중에 torque로 곱해줌)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size, state_dim, action_dim)

        # OU Noise
        self.noise = OUNoise(action_dim)

        # Networks
        self.actor = build_actor(state_dim, action_dim)
        self.actor_target = build_actor(state_dim, action_dim)
        self.critic = build_critic(state_dim, action_dim)
        self.critic_target = build_critic(state_dim, action_dim)

        # Optimizers
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=critic_lr)

        # Hard-copy weights: θ' ← θ
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

        # For loss printing/debugging
        self.mse_loss = keras.losses.MeanSquaredError()

    # ----------------------------------------
    # Action selection (for interaction with env)
    # ----------------------------------------
    def act(self, state, add_noise=True):
        """state: np.array shape (state_dim,)"""
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        action = self.actor(state).numpy()[0]  # [-1,1]
        if add_noise:
            noise = self.noise.sample()
            action = np.clip(action + noise, -1.0, 1.0)
        # scale to [−max_action, max_action], but 보통 env가 [-1,1]이면 여기서 바로 반환
        return action

    def act_no_noise(self, state):
        return self.act(state, add_noise=False)

    # ----------------------------------------
    # Store experience
    # ----------------------------------------
    def remember(self, s, a, r, s2, d):
        self.memory.add(s, a, r, s2, d)

    # ----------------------------------------
    # Soft target update
    # ----------------------------------------
    def _soft_update(self, main_model, target_model):
        main_weights = main_model.get_weights()
        target_weights = target_model.get_weights()
        new_weights = []
        for mw, tw in zip(main_weights, target_weights):
            nw = self.tau * mw + (1.0 - self.tau) * tw
            new_weights.append(nw)
        target_model.set_weights(new_weights)

    # ----------------------------------------
    # One training step (DDPG update)
    # ----------------------------------------
    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        # Critic update
        with tf.GradientTape() as tape_c:
            next_actions = self.actor_target(next_states, training=True)
            target_q = self.critic_target([next_states, next_actions], training=True)
            # y = r + γ(1-d)Q'(s', μ'(s'))
            y = rewards + self.gamma * (1.0 - dones) * target_q
            q = self.critic([states, actions], training=True)
            critic_loss = self.mse_loss(y, q)

        critic_grads = tape_c.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Actor update: maximize Q(s, μ(s)) == minimize -Q
        with tf.GradientTape() as tape_a:
            actions_pred = self.actor(states, training=True)
            q_pred = self.critic([states, actions_pred], training=True)
            actor_loss = -tf.reduce_mean(q_pred)

        actor_grads = tape_a.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        return critic_loss, actor_loss

    def train(self):
        if len(self.memory) < self.batch_size:
            return None, None

        s, a, r, s2, d = self.memory.sample(self.batch_size)
        critic_loss, actor_loss = self._train_step(
            tf.convert_to_tensor(s, dtype=tf.float32),
            tf.convert_to_tensor(a, dtype=tf.float32),
            tf.convert_to_tensor(r, dtype=tf.float32),
            tf.convert_to_tensor(s2, dtype=tf.float32),
            tf.convert_to_tensor(d, dtype=tf.float32),
        )

        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        # eager tensor → float
        return float(critic_loss.numpy()), float(actor_loss.numpy())

    def reset_noise(self):
        self.noise.reset()

    # ----------------------------------------
    # Save actor / critic
    # ----------------------------------------
    def save(self, actor_path="actor.weights.h5", critic_path="critic.weights.h5"):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        print("[INFO] Actor/Critic saved.")

    # ----------------------------------------
    # Load actor / critic
    # ----------------------------------------
    def load(self, actor_path="actor.weights.h5", critic_path="critic.weights.h5"):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)

        # Target networks sync
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

        print("[INFO] Actor/Critic loaded.")