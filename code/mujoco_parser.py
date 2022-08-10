import cv2,os,mujoco_py,glfw,time
import numpy as np
import matplotlib.pyplot as plt
from screeninfo import get_monitors
from util import pr2T

class MuJoCoManipulatorParserClass():
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
        self.ee_name  = 'panda_eef'
        self.VERBOSE  = VERBOSE
        self.tick     = 0
        self.t_init   = time.time()
        self.max_sec  = np.inf
        self.max_tick = np.inf
        # Parse basic info
        self._parse()
        
    def _parse(self):
        """
            Parse basic info
        """
        cwd                  = os.getcwd() 
        self.xml_path        = os.path.join(cwd,self.rel_path)
        self.mj_model        = mujoco_py.load_model_from_path(self.xml_path)
        self.sim             = mujoco_py.MjSim(self.mj_model)
        if self.VERBOSE:
            print ("[%s] parsed."%(self.xml_path))
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
        if self.VERBOSE:
            print ("dt:[%.3f] HZ:[%d]"%(self.dt,self.HZ))
            print ("body_names:\n%s"%(self.body_names))
            print ("ee_name         :%s"%(self.ee_name))
            print ("")
            print ("n_joint         : [%d]"%(self.n_joint))
            print ("joint_names     : %s"%(self.joint_names))
            print ("joint_types     : %s"%(self.joint_types))
            print ("n_rev_joint     : [%d]"%(self.n_rev_joint))
            print ("rev_joint_idxs  : %s"%(self.rev_joint_idxs))
            print ("rev_joint_names : %s"%(self.rev_joint_names))
            print ("pri_joint_idxs  : %s"%(self.pri_joint_idxs))
            print ("pri_joint_names : %s"%(self.pri_joint_names))
            print ("")
            print ("joint_range:\n%s"%(self.joint_range))
            print ("torque_range:\n%s"%(self.torque_range))
            print ("torque_range:\n%s"%(self.torque_range))

    def init_viewer(self,TERMINATE_GLFW=True,window_width=0.5,window_height=0.5):
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

    def plot_scene(
        self,figsize=(12,8),render_w=1200,render_h=800,title_str=None,title_fs=11):
        """
            Plot current scnene
        """
        img = self.sim.render(width=render_w,height=render_h)
        img = self.sim.render(width=render_w,height=render_h)
        img = cv2.flip(cv2.rotate(img,cv2.ROTATE_180),1) # 0:up<->down, 1:left<->right

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

    def step(self,torque=None):
        """
            Step
        """
        # Action
        if torque is not None:
            self.sim.data.ctrl[self.rev_joint_idxs] = np.copy(torque)
        # And then make a step
        self.sim.step()
        # Counter
        self.tick = self.tick + 1

    def forward(self):
        """
            Forward kinematics
        """
        # Forward kinematics
        self.sim.forward()
        # Counter
        self.tick = self.tick + 1

    def render(self,render_speedup=1.0):
        """
            Render
        """
        if (self.get_sec_sim() >= render_speedup*self.get_sec_wall()):
            self.viewer.render()

    def step_and_render(self,torque=None,render_speedup=1.0):
        """
            Step and Render
        """
        self.step(torque=torque)
        self.render(render_speedup=render_speedup)

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
        # Reset tick and timer
        self.tick    = 0
        self.t_init  = time.time()

    def print(self,print_every_sec=1.0,VERBOSE=1):
        """
            Print
        """
        if (((self.tick-1)%int(print_every_sec*self.HZ))==0):
            if (VERBOSE>=1):
                print ("tick:[%d], sec_wall:[%.3f]sec, sec_sim:[%.3f]sec"%
                (self.tick,self.get_sec_wall(),self.get_sec_sim()))

