import cv2,os,mujoco_py,glfw,time
import numpy as np
import matplotlib.pyplot as plt
from screeninfo import get_monitors
from util import r2w,trim_scale

class MuJoCoParserClass():
    def __init__(self,
                 name     = 'Robot',
                 rel_path = '../asset/panda/franka_panda.xml',
                 ee_name  = 'panda_eef',
                 VERBOSE  = False):
        """
            Init
        """
        self.name     = name
        self.rel_path = rel_path
        self.ee_name  = ee_name
        self.VERBOSE  = VERBOSE
        self.tick     = 0
        self.t_init   = time.time()
        self.max_sec  = np.inf
        self.max_tick = 50000
        self.SIM_MODE = 'Idle' # Idle / Dynamics / Kinematics
        # Parse basic info
        self._parse()
        # Terminate viewer
        self.terminate_viewer()
        glfw.init()
        # Reset 
        self.reset()
        
    def _parse(self):
        """
            Parse basic info
        """
        cwd                  = os.getcwd() 
        self.xml_path        = os.path.join(cwd,self.rel_path)
        self.mj_model        = mujoco_py.load_model_from_path(self.xml_path)
        self.sim             = mujoco_py.MjSim(self.mj_model)
        # Parse
        self.dt              = self.sim.model.opt.timestep
        self.HZ              = int(1/self.dt)
        self.body_names      = list(self.sim.model.body_names)
        self.n_joint         = self.sim.model.njnt
        self.joint_names     = [self.sim.model.joint_id2name(x) for x in range(self.n_joint)]
        self.joint_types     = self.sim.model.jnt_type # 0:free, 1:ball, 2:slide, 3:hinge
        self.rev_joint_idxs  = np.where(self.joint_types==3)[0].astype(np.int32) # revolute joint
        self.rev_joint_names = [self.joint_names[x] for x in self.rev_joint_idxs]
        self.rev_qvel_idxs   = [self.sim.model.get_joint_qvel_addr(x) for x in self.rev_joint_names]
        self.n_rev_joint     = len(self.rev_joint_idxs)
        self.pri_joint_idxs  = np.where(self.joint_types==2)[0].astype(np.int32) # prismatic joint
        self.pri_joint_names = [self.joint_names[x] for x in self.pri_joint_idxs]
        self.joint_range     = self.sim.model.jnt_range
        self.torque_range    = self.sim.model.actuator_ctrlrange
        self.n_torque        = self.torque_range.shape[0]
        if self.VERBOSE:
            print ("[%s] parsed."%(self.xml_path))
            print ("")
            print ("dt:[%.3f] HZ:[%d]"%(self.dt,self.HZ))
            print ("body_names:\n %s"%(self.body_names))
            print ("ee_name: [%s]"%(self.ee_name))
            print ("")
            print ("n_joint: [%d]"%(self.n_joint))
            print ("joint_names:\n %s"%(self.joint_names))
            print ("joint_types:\n %s"%(self.joint_types))
            print (" (0:free, 1:ball, 2:slide, 3:hinge) ")
            print ("n_rev_joint: [%d]"%(self.n_rev_joint))
            print ("rev_joint_idxs:\n %s"%(self.rev_joint_idxs))
            print ("rev_joint_names:\n %s"%(self.rev_joint_names))
            print ("pri_joint_idxs:\n %s"%(self.pri_joint_idxs))
            print ("pri_joint_names:\n %s"%(self.pri_joint_names))
            print ("")
            print ("joint_range:\n %s"%(self.joint_range))
            print ("torque_range:\n %s"%(self.torque_range))
            print ("n_torque:[%d]"%(self.n_torque))

        # Save initial q
        self.q_init = self.get_q_rev()

    def init_viewer(self,TERMINATE_GLFW=True,window_width=0.5,window_height=0.5,
                    cam_distance=None,cam_elevation=None,cam_lookat=None):
        """
            Init viewer
        """
        if TERMINATE_GLFW:
            glfw.terminate()
        glfw.init()
        # Init viewer
        self.viewer = mujoco_py.MjViewer(self.sim)
        glfw.set_window_size(
            window = self.viewer.window,
            width  = int(window_width*get_monitors()[0].width),
            height = int(window_height*get_monitors()[0].height))
        # Viewer setting
        if cam_distance is not None:
            self.viewer.cam.distance = cam_distance
        if cam_elevation is not None:
            self.viewer.cam.elevation = cam_elevation
        if cam_lookat is not None:
            self.viewer.cam.lookat[0] = cam_lookat[0]
            self.viewer.cam.lookat[1] = cam_lookat[1]
            self.viewer.cam.lookat[2] = cam_lookat[2]
    
    def terminate_viewer(self):
        """
            Terminate viewer
        """
        glfw.terminate()

    def set_max_sec(self,max_sec=10.0):
        """
            Set maximum second
        """
        self.max_sec  = max_sec
        self.max_tick = int(self.max_sec*self.HZ)+1

    def set_max_tick(self,max_tick=1000):
        """
            Set maximum tick
        """
        self.max_tick = max_tick + 1
        self.max_sec  = self.max_tick*self.dt

    def plot_scene(self,figsize=(12,8),render_w=1200,render_h=800,title_str=None,title_fs=11,
                   cam_distance=None,cam_elevation=None,cam_lookat=None,NO_PLOT=False):
        """
            Plot current scnene
        """
        for _ in range(5):
            if cam_distance is not None:
                for r_idx in range(len(self.sim.render_contexts)):
                    self.sim.render_contexts[r_idx].cam.distance  = cam_distance
            if cam_elevation is not None:
                for r_idx in range(len(self.sim.render_contexts)):
                    self.sim.render_contexts[r_idx].cam.elevation = cam_elevation
            if cam_lookat is not None:
                for r_idx in range(len(self.sim.render_contexts)):
                    self.sim.render_contexts[r_idx].cam.lookat[0] = cam_lookat[0]
                    self.sim.render_contexts[r_idx].cam.lookat[1] = cam_lookat[1]
                    self.sim.render_contexts[r_idx].cam.lookat[2] = cam_lookat[2]
            img = self.sim.render(width=render_w,height=render_h)

        img = cv2.flip(cv2.rotate(img,cv2.ROTATE_180),1) # 0:up<->down, 1:left<->right
        if NO_PLOT:
            return img
        else:
            plt.figure(figsize=figsize)
            plt.imshow(img)
            if title_str is not None:
                plt.title(title_str,fontsize=title_fs)
            plt.show()

    def IS_ALIVE(self):
        """
            Is alive
        """
        return (self.tick < self.max_tick)

    def get_sec_sim(self):
        """
            Get simulation time
        """
        self.sim_state = self.sim.get_state()
        self.sec_sim   = self.sim_state.time
        if self.SIM_MODE=='Kinematics':
            self.sec_sim = self.tick*self.dt # forward() does not increase 'sim_state.time'
        return self.sec_sim

    def get_sec_wall(self):
        """
            Get wall-clock time
        """
        self.sec_wall = time.time() - self.t_init
        return self.sec_wall

    def get_q_rev(self):
        """
            Get current revolute joint position
        """
        self.sim_state = self.sim.get_state()
        self.q_rev = self.sim_state.qpos[self.rev_joint_idxs]
        return self.q_rev

    def get_p_body(self,body_name):
        """
            Get body position
        """
        self.sim_state = self.sim.get_state()
        p = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(body_name)])
        return p

    def get_R_body(self,body_name):
        """
            Get body rotation
        """
        R = np.array(self.sim.data.body_xmat[self.sim.model.body_name2id(self.ee_name)].reshape([3, 3]))
        return R

    def get_J_body(self,body_name):
        """
            Get body Jacobian
        """
        J_p    = np.array(self.sim.data.get_body_jacp(body_name).reshape((3, -1))[:,self.rev_qvel_idxs])
        J_R    = np.array(self.sim.data.get_body_jacr(body_name).reshape((3, -1))[:,self.rev_qvel_idxs])
        J_full = np.array(np.vstack([J_p,J_R]))
        return J_p,J_R,J_full

    def step(self,torque=None,TORQUE_TO_REV_JOINT=True):
        """
            Forward dynamics
        """
        # Set mode
        self.SIM_MODE = 'Dynamics' # Idle / Dynamics / Kinematics
        # Action
        if torque is not None:
            if TORQUE_TO_REV_JOINT:
                self.sim.data.ctrl[self.rev_joint_idxs] = np.copy(torque)
            else:
                self.sim.data.ctrl[:] = np.copy(torque)
        # And then make a step
        self.sim.step()
        # Counter
        self.tick = self.tick + 1

    def forward(self,q_rev=None):
        """
            Forward kinematics
        """
        # Set mode
        self.SIM_MODE = 'Kinematics' # Idle / Dynamics / Kinematics
        # Forward kinematics
        if q_rev is not None:
            self.sim.data.qpos[self.rev_joint_idxs] = q_rev
        self.sim.forward()
        # Counter
        self.tick = self.tick + 1

    def render(self,render_speedup=1.0,RENDER_ALWAYS=False):
        """
            Render
        """
        if RENDER_ALWAYS:
            self.viewer._render_every_frame = True
        else:
            self.viewer._render_every_frame = False

        if (self.get_sec_sim() >= render_speedup*self.get_sec_wall()) or RENDER_ALWAYS:
            self.viewer.render()

    def step_and_render(self,torque=None,TORQUE_TO_REV_JOINT=True,render_speedup=1.0,RENDER_ALWAYS=False):
        """
            Step and Render
        """
        self.step(torque=torque,TORQUE_TO_REV_JOINT=TORQUE_TO_REV_JOINT)
        self.render(render_speedup=render_speedup,RENDER_ALWAYS=RENDER_ALWAYS)

    def forward_and_render(self,q_rev=None,render_speedup=1.0,RENDER_ALWAYS=False):
        """
            FK and Render
        """
        self.forward(q_rev=q_rev)
        self.render(render_speedup=render_speedup,RENDER_ALWAYS=RENDER_ALWAYS)

    def add_marker(self,pos,radius=0.02,color=np.array([0.0,1.0,0.0,1.0]),label=None):
        """
            Add a maker to renderer
        """
        self.viewer.add_marker(
            pos   = pos,
            type  = 2, # mjtGeom: 2:sphere, 3:capsule, 6:box, 9:arrow
            size  = radius*np.ones(3),
            rgba  = color,
            label = label)
        
    def reset(self):
        """
            Reset
        """
        # Reset simulation
        self.sim.reset()
        # Revert to the initial position
        self.forward(q_rev=self.q_init)
        # Reset tick and timer
        self.tick    = 0
        self.t_init  = time.time()

    def print(self,print_every_sec=None,print_every_tick=None,VERBOSE=1):
        """
            Print
        """
        if print_every_sec is not None:
            if (((self.tick-1)%int(print_every_sec*self.HZ))==0):
                if (VERBOSE>=1):
                    print ("tick:[%d/%d], sec_wall:[%.3f]sec, sec_sim:[%.3f]sec"%
                    (self.tick,self.max_tick,self.get_sec_wall(),self.get_sec_sim()))
        if print_every_tick is not None:
            if (((self.tick-1)%print_every_tick)==0):
                if (VERBOSE>=1):
                    print ("tick:[%d/%d], sec_wall:[%.3f]sec, sec_sim:[%.3f]sec"%
                    (self.tick,self.max_tick,self.get_sec_wall(),self.get_sec_sim()))

    def one_step_ik(self,body_name,p_trgt=None,R_trgt=None,th=1.0*np.pi/180.0):
        """
            One-step inverse kinematics
        """
        J_p,J_R,J_full = self.get_J_body(body_name=body_name)
        p_curr = self.get_p_body(body_name=body_name)
        R_curr = self.get_R_body(body_name=body_name)
        if (p_trgt is not None) and (R_trgt is not None): # both p and R targets are given
            p_err = (p_trgt-p_curr)
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J,err = J_full,np.concatenate((p_err,w_err))
        elif (p_trgt is not None) and (R_trgt is None): # only p target is given
            p_err = (p_trgt-p_curr)
            J,err = J_p,p_err
        elif (p_trgt is None) and (R_trgt is not None): # only R target is given
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J,err = J_R,w_err
        else:
            raise Exception('At least one IK target is required!')
        dq = np.linalg.solve(a=(J.T@J)+1e-6*np.eye(J.shape[1]),b=J.T@err)
        dq = trim_scale(x=dq,th=th)
        return dq,err

    def sleep(self,sec=0.0):
        """
            Sleep
        """
        if sec > 0.0:
            time.sleep(sec)

