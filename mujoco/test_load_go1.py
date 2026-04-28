import mujoco 
import mujoco.viewer
import time 
# loading go1 model 
model = mujoco.MjModel.from_xml_path("c:\\Learning code\\mujoco_menagerie\\unitree_go1\\scene.xml")
data = mujoco.MjData(model)
print("number of position values: ", model.nq)
print("number of velocity values: ", model.nv)
print("number of actuator control values: ", model.nu)
#printing actuator names 
for i in range(model.nu):
    name = mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(i, name)
for i in range(model.njnt):
    name = mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT, i)
    print(i, name)
print("qpos:", data.qpos)
print("qvel:", data.qvel)
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        print("qpos:", data.qpos)
        data.ctrl[:] = 0
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)


#mujoco.viewer.launch(model, data)
