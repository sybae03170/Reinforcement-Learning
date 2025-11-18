import numpy as np
import mujoco
import gym
from gym import spaces

class AliengoEnv(gym.Env):
    def __init__(self, model_path="aliengo.xml"):
        # load Mujoco model& data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # joint(actuator) 수 + base foot&body index 정의
        self.num_joints = self.model.nu
        self.base_body_id = self.model.body_name2id("trunk")    # Base body index
        self.foot_site_names = ["fl_tc", "fr_tc", "rl_tc", "rr_tc"]
        self.foot_site_ids = [self.model.site_name2id(n) for n in self.foot_site_names]

        # -------------- Observation 차원 -------------- #
        # base state (SRBM)
        base_orientation_dim = 4          # quaternion
        base_lin_vel_dim = 3
        base_ang_vel_dim = 3

        # Joint states
        joint_pos_dim = self.num_joints   # 12
        joint_vel_dim = self.num_joints   # 12

        # Foot contacts
        foot_contact_dim = 4

        # Total observation dimension
        self.obs_dim = (
            base_orientation_dim +
            base_lin_vel_dim +
            base_ang_vel_dim +
            joint_pos_dim +
            joint_vel_dim +
            foot_contact_dim
        )
        # ---------------------------------------------- #

        # gym spaces (observation& action의 range와 shape 정의)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )        # observation state vector는 길이 obs_dim인 실수 벡터!

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )        # actor output -1.0 ~ 1.0으로 정의 후 나중에 실제 torque scaling

    # ------------------------------------------------
    # (1) Observation Function (state 구성)
    # ------------------------------------------------
    def _get_obs(self):
        """
        SRBM + joint state + foot contact 기반의 observation 구성 
        """
        # Base(trunk) Orientation - quaternion(w, x, y, z) + joint pos & vel
        base_quat = self.data.xquat[self.base_body_id]       # shape (4,)
        base_lin_vel = self.data.qvel[0:3]                   # [vx, vy, vz]
        base_ang_vel = self.data.qvel[3:6]                   # [wx, wy, wz]

        # Joint pos & vel
        joint_pos = self.data.qpos[7:7 + self.num_joints]  # qpos 7개의 값 = (x,y,z) 위치 + 쿼터니언(w,x,y,z)
        joint_vel = self.data.qvel[6:6 + self.num_joints]  # qvel 6개의 값 = 선속도(vx, vy, vz) + 각속도(wx, wy, wz)

        # foot contact (0 OR 1)
        foot_contacts = []
        for site_id in self.foot_site_ids:
            # site contact detection: check if any geom is in contact
            is_contact = 0
            for c in range(self.data.ncon):
                contact = self.data.contact[c]
                if contact.site1 == site_id or contact.site2 == site_id:
                    is_contact = 1
                    break
            foot_contacts.append(is_contact)

        foot_contacts = np.array(foot_contacts, dtype=np.float32)

        # Concatenate
        obs = np.concatenate([
            base_quat,          # 4
            base_lin_vel,       # 3
            base_ang_vel,       # 3
            joint_pos,          # 12
            joint_vel,          # 12
            foot_contacts       # 4
        ], dtype=np.float32)

        return obs
    
        # ------------------------------------------------
    # (2) Reset Function
    # ------------------------------------------------
    def reset(self):
        """
        환경 초기화:
        - qpos, qvel을 초기화
        - base를 정해진 높이에 배치
        - 마지막 상태/보상 초기화
        """
        mujoco.mj_resetData(self.model, self.data)

        # 적당한 초기 높이 (Aliengo는 대략 0.3m 근처)
        self.data.qpos[2] = 0.35   

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()

        # for reward (slip 계산 시 사용)
        self.last_foot_positions = self._get_foot_positions()  
        self.last_action = np.zeros(self.num_joints)

        return obs

    # ------------------------------------------------
    # (3) Step Function (action → physics → reward)
    # ------------------------------------------------
    def step(self, action):
        """
        행동(action)을 torque로 변환하여 MuJoCo simulation step을 진행하고
        reward를 계산.
        """
        # ---- Action Scaling ----
        # action: [-1, 1] → 실제 torque 범위로 확장
        max_torque = 33.5   # Aliengo hip/thigh/calf torque approx
        torques = action * max_torque
        self.data.ctrl[:] = torques

        # ---- MuJoCo simulate ----
        mujoco.mj_step(self.model, self.data)

        # ---- Observation ----
        obs = self._get_obs()

        # ---- Reward ----
        reward = self._compute_reward(action)

        # ---- Termination Conditions ----
        done = False

        base_z = self.data.qpos[2]
        quat = self.data.xquat[self.base_body_id]
        w, x, y, z = quat

        # 1) 너무 낮아짐 (무너짐)
        if base_z < 0.18:
            done = True
            reward -= 10.0

        # 2) pitch/roll 너무 큼 (전복)
        roll, pitch, yaw = self._quat_to_euler(quat)
        if abs(roll) > 0.8 or abs(pitch) > 0.8:
            done = True
            reward -= 10.0

        # for next slip calculation
        self.last_action = action

        return obs, reward, done, {}

    # ================================================================
    # (4) Reward Function — 6개 항목 준수
    # ================================================================
    def _compute_reward(self, action):
        """
        Reward Terms:
        1) forward velocity reward
        2) roll/pitch penalty
        3) base height penalty
        4) torque penalty
        5) joint velocity penalty
        6) foot slip penalty
        """

        # ------------ 1) Forward Velocity ------------
        vx = self.data.qvel[0]     # base linear x velocity
        reward_forward = 1.5 * vx

        # ------------ 2) Upright (roll/pitch) penalty ------------
        quat = self.data.xquat[self.base_body_id]
        roll, pitch, yaw = self._quat_to_euler(quat)

        reward_upright = -1.0 * (abs(roll) + abs(pitch))

        # ------------ 3) Base height 유지 ------------
        base_z = self.data.qpos[2]
        nominal_h = 0.35
        reward_height = -1.0 * abs(base_z - nominal_h)

        # ------------ 4) Torque penalty ------------
        reward_torque = -0.001 * np.sum((action)**2)

        # ------------ 5) Joint velocity penalty ------------
        joint_vel = self.data.qvel[6 : 6 + self.num_joints]
        reward_joint_vel = -0.0001 * np.sum(joint_vel**2)

        # ------------ 6) Foot slip penalty ------------
        slip_penalty = self._compute_slip_penalty()
        reward_slip = -2.0 * slip_penalty

        # ------------ Total reward ------------
        total_reward = (
            reward_forward +
            reward_upright +
            reward_height +
            reward_torque +
            reward_joint_vel +
            reward_slip
        )

        # optional clipping
        total_reward = np.clip(total_reward, -10.0, 10.0)

        return total_reward

    # ------------------------------------------------
    # (5) Foot slip penalty helper
    # ------------------------------------------------
    def _compute_slip_penalty(self):
        """
        stance 상태에서 foot velocity 크기를 penalty로 사용.
        slip = Σ ||foot_vel_xy||^2 (contact 중인 발)
        """
        slip = 0.0

        for i, site_id in enumerate(self.foot_site_ids):
            contact_flag = self._is_contacting(site_id)

            # 현재 foot velocity
            foot_vel = self.data.site_xvelp[site_id]  # 3D velocity

            # stance 상태일 때 XY-plane slip 측정
            if contact_flag == 1:
                slip += foot_vel[0]**2 + foot_vel[1]**2

        return slip

    # ------------------------------------------------
    # (6) foot contact 확인
    # ------------------------------------------------
    def _is_contacting(self, site_id):
        for c in range(self.data.ncon):
            contact = self.data.contact[c]
            if contact.site1 == site_id or contact.site2 == site_id:
                return 1
        return 0

    # ------------------------------------------------
    # (7) foot positions (slip 계산용)
    # ------------------------------------------------
    def _get_foot_positions(self):
        return np.array([self.data.site_xpos[i] for i in self.foot_site_ids])

    # ------------------------------------------------
    # (8) Quaternion → Euler 변환
    # ------------------------------------------------
    def _quat_to_euler(self, quat):
        """
        MuJoCo quaternion(w,x,y,z) → roll, pitch, yaw
        """
        w, x, y, z = quat
        # roll (x-axis)
        sinr = 2.0 * (w*x + y*z)
        cosr = 1.0 - 2.0 * (x*x + y*y)
        roll = np.arctan2(sinr, cosr)

        # pitch (y-axis)
        sinp = 2.0 * (w*y - z*x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

        # yaw (z-axis)
        siny = 2.0 * (w*z + x*y)
        cosy = 1.0 - 2.0 * (y*y + z*z)
        yaw = np.arctan2(siny, cosy)

        return roll, pitch, yaw
