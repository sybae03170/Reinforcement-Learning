import mujoco
from mujoco import viewer

# (1) 모델 로드
model = mujoco.MjModel.from_xml_path(r"C:\aliengo\aliengo.xml")
data = mujoco.MjData(model)

# (2)뷰어 실행 및 시뮬레이션 루프
with viewer.launch_passive(model, data) as v:
    print("[INFO] MuJoCo viewer launched. Close the window to end simulation.")
    while v.is_running():
        mujoco.mj_step(model, data)  # 한 스텝 시뮬레이션 진행
        v.sync()                      # 화면 갱신
