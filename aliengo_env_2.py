import numpy as np
import mujoco
import gym
from gym import spaces
from mujoco import viewer


class AliengoEnv(gym.Env):
    """
    Pure RL Locomotion Environment for Aliengo
    - PD 완전 제거
    - Actor(action) → torques directly applied
    - 초기자세만 안정적으로 세팅
    """

    def __init__(self, model_path="aliengo/aliengo.xml"):
        super().__init__()

        # -------------------------
        # Load MuJoCo model
        # -------------------------
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.num_joints = self.model.nu
        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk"
        )

        # Foot site & geom
        self.foot_site_names = ["fl_tc", "fr_tc", "rl_tc", "rr_tc"]
        self.foot_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
            for n in self.foot_site_names
        ]

        self.foot_geom_names = ["fl_foot", "fr_foot", "rl_foot", "rr_foot"]
        self.foot_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in self.foot_geom_names
        ]

        # Episode length
        self.max_episode_steps = 2000
        self.step_counter = 0

        # -------------------------
        # Observation space
        # -------------------------
        # base_quat(4), base vel(3), base ang vel(3)
        # joint pos(12), joint vel(12)
        # foot contacts(4)
        # + stability info: base height(1), roll, pitch(2)
        self.obs_dim = 4 + 3 + 3 + self.num_joints + self.num_joints + 4 + 3

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # -------------------------
        # Action space (RL 토크 직접 적용)
        # -------------------------
        self.max_torque = 100.0
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )

        print("[INFO] Pure RL AliengoEnv Loaded.")

    # ============================================================
    # (1) Observation
    # ============================================================
    def _get_obs(self):

        # Base orientation quaternion
        base_quat = self.data.xquat[self.base_body_id]

        # Base velocities (world frame)
        base_lin_vel = self.data.qvel[0:3]
        base_ang_vel = self.data.qvel[3:6]

        # Joint states
        joint_pos = self.data.qpos[7:7 + self.num_joints]
        joint_vel = self.data.qvel[6:6 + self.num_joints]

        # Foot contact
        foot_contacts = []
        for geom_id in self.foot_geom_ids:
            contact_flag = 0
            for c in range(self.data.ncon):
                contact = self.data.contact[c]
                if contact.geom1 == geom_id or contact.geom2 == geom_id:
                    contact_flag = 1
                    break
            foot_contacts.append(contact_flag)
        foot_contacts = np.array(foot_contacts, dtype=np.float32)

        # Stability info
        base_z = np.array([self.data.qpos[2]], dtype=np.float32)
        roll, pitch, yaw = self._quat_to_euler(base_quat)
        stability = np.array([roll, pitch], dtype=np.float32)

        obs = np.concatenate([
            base_quat,
            base_lin_vel,
            base_ang_vel,
            joint_pos,
            joint_vel,
            foot_contacts,
            base_z,
            stability
        ])

        return obs

    # ============================================================
    # (2) Reset (PD warm-up 제거 → 초기 자세만 안정적으로 세팅)
    # ============================================================
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

        # base 위치 설정
        self.data.qpos[0:3] = [0.0, 0.0, 0.38]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        # 안정적인 초기 다리 각도 (standing)
        init_joint = np.array([
            0.0, 0.8, -1.6,
            0.0, 0.8, -1.6,
            0.0, 0.8, -1.6,
            0.0, 0.8, -1.6
        ])
        self.data.qpos[7:7+self.num_joints] = init_joint

        self.data.qvel[:] = 0.0

        # forward dynamics consistency
        mujoco.mj_forward(self.model, self.data)

        self.step_counter = 0
        return self._get_obs()

    # ============================================================
    # (3) Step: RL → Torque Direct Application
    # ============================================================
    def step(self, action):

        # ------------------------------
        # RL torque mapping
        # ------------------------------
        torques = np.clip(action * self.max_torque,
                          -self.max_torque, self.max_torque)
        self.data.ctrl[:] = torques

        # ------------------------------
        # Simulate physics
        # ------------------------------
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward(action)
        self.step_counter += 1

        # ------------------------------
        # Termination conditions
        # ------------------------------
        done = False
        base_z = self.data.qpos[2]
        roll, pitch, yaw = self._quat_to_euler(self.data.xquat[self.base_body_id])

        if base_z < 0.22:     # 무너짐
            done = True
            reward -= 10.0

        if abs(roll) > 0.5 or abs(pitch) > 0.5:   # 전복
            done = True
            reward -= 10.0

        if self.step_counter >= self.max_episode_steps:
            done = True

        return obs, reward, done, {}

    # ============================================================
    # (4) Reward (Pure RL Locomotion용)
    # ============================================================
    def _compute_reward(self, action):

        # Forward speed
        vx = self.data.qvel[0]
        reward_forward = vx * 2.0     # forward incentive

        # Stability penalty
        roll, pitch, yaw = self._quat_to_euler(
            self.data.xquat[self.base_body_id])
        penalty_orientation = -2.0 * (abs(roll) + abs(pitch))

        # Height stability
        base_z = self.data.qpos[2]
        penalty_height = -5.0 * abs(base_z - 0.38)

        # Torque penalty
        penalty_torque = -0.001 * np.sum(action ** 2)

        # Keep agent alive
        reward_alive = 1.0

        total_reward = (
            reward_forward +
            penalty_orientation +
            penalty_height +
            penalty_torque +
            reward_alive
        )

        return float(total_reward)

    # ============================================================
    # (5) Quaternion → Euler
    # ============================================================
    def _quat_to_euler(self, quat):
        w, x, y, z = quat

        # roll
        sinr = 2.0 * (w*x + y*z)
        cosr = 1.0 - 2.0 * (x*x + y*y)
        roll = np.arctan2(sinr, cosr)

        # pitch
        sinp = 2.0 * (w*y - z*x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

        # yaw
        siny = 2.0 * (w*z + x*y)
        cosy = 1.0 - 2.0 * (y*y + z*z)
        yaw = np.arctan2(siny, cosy)

        return roll, pitch, yaw

    # ============================================================
    # (6) Rendering
    # ============================================================
    def render(self):
        if not hasattr(self, "viewer") or self.viewer is None:
            self.viewer = viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()
            self.viewer = None
