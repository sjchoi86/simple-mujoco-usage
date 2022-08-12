import numpy as np

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