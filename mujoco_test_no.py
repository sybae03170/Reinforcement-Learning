import mujoco
from mujoco import viewer
from aliengo_env import AliengoEnv

# env init 값 테스트
env = AliengoEnv("aliengo/aliengo.xml")
obs = env.reset()

for i in range(5000):
    env.data.ctrl[:] = 0.0
    mujoco.mj_step(env.model, env.data)
    env.render()


# (1) 모델 로드
# model = mujoco.MjModel.from_xml_path(r"C:\aliengo\aliengo.xml")
# data = mujoco.MjData(model)

# # (2)뷰어 실행 및 시뮬레이션 루프
# with viewer.launch_passive(model, data) as v:
#     print("[INFO] MuJoCo viewer launched. Close the window to end simulation.")
#     while v.is_running():
#         mujoco.mj_step(model, data)  # 한 스텝 시뮬레이션 진행
#         v.sync()                      # 화면 갱신