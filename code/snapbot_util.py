import numpy as np
from util import quaternion_to_euler_angle,r2rpy

def get_snapbot_q(env):
    """
        Get joint position from Snapbot env
    """
    q = env.sim.data.qpos.flat
    q = np.asarray([q[9],q[10],q[13],q[14],q[17],q[18],q[21],q[22]])
    return q

def wait_until_snapbot_on_ground(env,PID,wait_sec=2.0):
    """
        Wait until Snapbot is on the ground
    """
    env.reset()
    PID.reset()
    while (env.get_sec_sim()<=wait_sec):
        q = get_snapbot_q(env)
        PID.update(x_trgt=np.zeros(env.n_torque),t_curr=env.get_sec_sim(),x_curr=q,VERBOSE=False)
        env.step(torque=PID.out(),TORQUE_TO_REV_JOINT=False)

def get_snapbot_p_torso(env,body_name='torso'):
    """
        Get torso position of a body
    """
    p_torso = env.sim.data.get_body_xpos(body_name)
    return p_torso

def get_snapbot_heading(env,body_name='torso'):
    """
        Get heading of a body
    """
    q = env.sim.data.get_body_xquat(body_name)
    _, _, z_deg = quaternion_to_euler_angle(q[0], q[1], q[2], q[3])
    return z_deg

def snapbot_rollout(env,PID,traj_joints,n_traj_repeat=5,DO_RENDER=True):
    """
        Rollout of Snapbot
    """
    if DO_RENDER:
        env.init_viewer(TERMINATE_GLFW=True,window_width=0.5,window_height=0.5,cam_distance=2.5,
                        cam_elevation=-20,cam_lookat=[1.0,0,-0.2])
    wait_until_snapbot_on_ground(env,PID)
    PID.reset()
    L,cnt = traj_joints.shape[0],0
    sec_list    = np.zeros(shape=(int(L*n_traj_repeat)))
    q_curr_list = np.zeros(shape=(int(L*n_traj_repeat),env.n_torque))
    q_trgt_list = np.zeros(shape=(int(L*n_traj_repeat),env.n_torque))
    torque_list = np.zeros(shape=(int(L*n_traj_repeat),env.n_torque))
    xyrad_list  = np.zeros(shape=(int(L*n_traj_repeat),3))
    for r_idx in range(n_traj_repeat): # repeat
        for tick in range(L): # for each tick in trajectory
            sec,q_curr,q_trgt=cnt*env.dt,get_snapbot_q(env),traj_joints[tick,:]
            PID.update(x_trgt=q_trgt,t_curr=sec,x_curr=get_snapbot_q(env),VERBOSE=False)
            torque = PID.out()
            if DO_RENDER:
                env.step_and_render(torque=torque,TORQUE_TO_REV_JOINT=False,render_speedup=1.0)
            else:
                env.step(torque=torque,TORQUE_TO_REV_JOINT=False)
            p_torso = env.get_p_body(body_name='torso')
            heading_rad = r2rpy(env.get_R_body(body_name='torso'),unit='rad')[2]
            # Append
            sec_list[cnt],q_curr_list[cnt,:],q_trgt_list[cnt,:] = sec,q_curr,q_trgt
            torque_list[cnt,:],xyrad_list[cnt,:] = torque,np.concatenate((p_torso[:2],[heading_rad]))
            cnt = cnt + 1 # tick
    res = {'sec_list':sec_list,'q_curr_list':q_curr_list,'q_trgt_list':q_trgt_list,
           'torque_list':torque_list,'xyrad_list':xyrad_list}
    return res