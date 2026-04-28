import mujoco


model = mujoco.MjModel.from_xml_path(
    r"c:\Learning code\mujoco_menagerie\unitree_go1\scene.xml"
)

home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

if home_key_id == -1:
    raise ValueError("Could not find a keyframe named 'home' in the model.")

home_qpos = model.key_qpos[home_key_id]
home_height = home_qpos[2]
fall_height = 0.65 * home_height

print("home key id:", home_key_id)
print("home height:", home_height)
print("suggested fall height:", fall_height)
