import mujoco
import imageio
import os
from config.xml_cfg import XMLParams
from config.vis_cfg import VisualizationParams

def load_model(xml_string: str) -> tuple[mujoco.MjModel, mujoco.MjData]:    
    m = mujoco.MjModel.from_xml_string(xml_string)
    d = mujoco.MjData(m)
    return m, d

def reset_model(m: mujoco.MjModel, 
                d: mujoco.MjData, 
                keyframe: str = "init") -> None:
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    mujoco.mj_forward(m, d)
    
def set_viewer(viewer: mujoco.viewer,
               track: bool,
               presets: dict,
               show_world_csys: bool,
               show_body_csys: bool) -> None:
    
    if track:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
    else:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    viewer.cam.lookat[:] = presets["lookat"]        # Point the camera is looking at
    viewer.cam.distance = presets["distance"]       # Distance from lookat point
    viewer.cam.azimuth = presets["azimuth"]         # Horizontal rotation angle [deg]
    viewer.cam.elevation = presets["elevation"]     # Vertical rotation angle [deg]
    
    if show_body_csys:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    if show_world_csys:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
        
def set_video_cam(video_cam,
            track: bool,
            presets: dict,
            show_world_csys: bool,
            show_body_csys: bool) -> None:
    
    video_cam.lookat[:] = presets["lookat"]        # Point the camera is looking at
    video_cam.distance = presets["distance"]       # Distance from lookat point
    video_cam.azimuth = presets["azimuth"]         # Horizontal rotation angle [deg]
    video_cam.elevation = presets["elevation"]     # Vertical rotation angle [deg]
        

def generate_xml(xml_cfg: XMLParams) -> str:
    
    obs_xml = """                
            <body name="obs_o" pos="0 0 0">
                <geom type="sphere" size="0" rgba="1 1 1 0.5" />
            </body>
    """
    

    XML_HEADER = f"""
    <mujoco model="{xml_cfg.name}">
        <compiler angle="radian"/>
        <option gravity="0 0 {-xml_cfg.gravity}" integrator="RK4" timestep="{xml_cfg.dt}">
        </option>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />
            <texture type="2d" name="floor" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
            <material name="floor" texture="floor" texuniform="true" texrepeat="5 5" reflectance="0.5"/>
        </asset>

        <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
            <global offwidth="1280" offheight="720"/>
        </visual>
    """
    
    XML_WORLDBODY = """
        <worldbody>
            <geom name="floor" type="plane" material="floor" size="10 10 0.1" pos="0 0 0" conaffinity="0"/>
    """  
    
    XML_SPOTLIGHT = """
            <light name="spotlight_m" mode="targetbodycom" target="agent_m" pos="0 0 10" cutoff="60" />
    """
    for m in range(xml_cfg.M):
        spotlight_xml_m = XML_SPOTLIGHT.replace("_m", f"_{m}")
        XML_WORLDBODY += spotlight_xml_m
        

    for m in range(xml_cfg.M):
        agent_xml_m = xml_cfg.agent_xml.replace("_m", f"_{m}")
        XML_WORLDBODY += agent_xml_m
        
    for o, obs in enumerate(xml_cfg.obs):
        obs_xml_o = obs_xml.replace("_o", f"_{o}").replace("pos=\"0 0 0\"", f"pos=\"{obs[0]} {obs[1]} {obs[2]}\"").replace("size=\"0\"", f"size=\"{obs[3]}\"")
        XML_WORLDBODY += obs_xml_o
        
    XML_WORLDBODY +=  "    </worldbody>\n"
    
    
    XML_ACUATAORS = """
        <actuator>
    """
    
    for m in range(xml_cfg.M):
        actuator_xml_m = xml_cfg.actuator_xml.replace("_m", f"_{m}")
        XML_ACUATAORS += actuator_xml_m

    XML_ACUATAORS +=  """
        </actuator>
    """
    

    qpos = " ".join(map(str, xml_cfg.q0_val[:, :xml_cfg.nq].flatten()))
    qvel = " ".join(map(str, xml_cfg.q0_val[:, xml_cfg.nq:].flatten()))

    XML_KEYFRAME = f"""
        <keyframe>
            <key name="init" 
                qpos="{qpos}"
                qvel="{qvel}"
            />
        </keyframe>
    """


    XML_FOOTER = "</mujoco>"
    

    XML =  XML_HEADER + XML_WORLDBODY + XML_ACUATAORS + XML_KEYFRAME +  XML_FOOTER
    
    return XML

       
def mj_cleanup(vis_cfg: VisualizationParams,
               name: str,
               fname: str,
               qualifier: str = "") -> None:
    
    if vis_cfg.enable_viewer:
        vis_cfg.viewer.close()

    os.makedirs(f"figures/{name}/videos", exist_ok=True)

    imageio.mimsave(
        f"figures/{name}/videos/{fname}{('_' + qualifier) if qualifier else ''}.mp4",
        vis_cfg.frames,
        fps=vis_cfg.vid_fps,
        codec="libx264",
        quality=8
    )
    

def init_vis(
        m: mujoco.MjModel, 
        d: mujoco.MjData, 
        vis_cfg: VisualizationParams) -> None:
     
    vis_cfg.presets["lookat"] = d.qpos[:3]
    
    if vis_cfg.enable_viewer:
        viewer = mujoco.viewer.launch_passive(m, d)
        set_viewer(
                viewer=viewer, 
                track=vis_cfg.track, 
                presets=vis_cfg.presets, 
                show_world_csys=vis_cfg.show_world_csys, 
                show_body_csys=vis_cfg.show_body_csys)
        vis_cfg.viewer = viewer

    video_cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(video_cam)
    set_video_cam(
                video_cam=video_cam, 
                presets=vis_cfg.presets, 
                track=vis_cfg.track, 
                show_world_csys=vis_cfg.show_world_csys, 
                show_body_csys=vis_cfg.show_body_csys)
    
    video_renderer = mujoco.Renderer(m, width=vis_cfg.vid_width, height=vis_cfg.vid_height)
    
    vis_cfg.video_cam = video_cam
    vis_cfg.video_renderer = video_renderer
    vis_cfg.frames = []


def mj_vis_step(d: mujoco.MjData,
                vis_cfg: VisualizationParams) -> None:

    vis_cfg.video_cam.lookat[:] = d.qpos[:3]
    vis_cfg.video_renderer.update_scene(d, camera=vis_cfg.video_cam)
    frame = vis_cfg.video_renderer.render()
    vis_cfg.frames.append(frame.copy())
    vis_cfg.next_frame_time += 1/vis_cfg.vid_fps

    if vis_cfg.enable_viewer:
        if vis_cfg.viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
            vis_cfg.viewer.cam.lookat[:] = d.qpos[:3]
        vis_cfg.viewer.sync()

