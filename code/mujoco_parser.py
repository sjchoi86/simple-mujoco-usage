import os,mujoco_py,glfw,time
import numpy as np
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
        

    def _update(self):
        """
            Update
        """
        self.tick      = self.tick + 1
        self.sim_state = self.sim.get_state()
        self.q_curr    = self.sim_state.qpos[self.rev_joint_idxs]
        self.sec_sim   = self.sim_state.time
        self.sec_wall  = time.time() - self.t_init
        if self.ee_name is not None:
            self.p_EE_curr = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(self.ee_name)])
            self.R_EE_curr = np.array(self.sim.data.body_xmat[self.sim.model.body_name2id(self.ee_name)].reshape([3, 3]))
            self.T_EE_curr = pr2T(p=self.p_EE_curr,R=self.R_EE_curr)
            self.J_p_EE    = np.array(self.sim.data.get_body_jacp(self.ee_name).reshape((3, -1))[:,self.rev_qvel_idxs])
            self.J_R_EE    = np.array(self.sim.data.get_body_jacr(self.ee_name).reshape((3, -1))[:,self.rev_qvel_idxs])
            self.J_full_EE = np.array(np.vstack([self.J_p_EE,self.J_R_EE]))
        else:
            self.p_EE_curr = None
            self.R_EE_curr = None
            self.T_EE_curr = None
            self.J_p_EE    = None
            self.J_R_EE    = None
            self.J_full_EE = None

    def step(self):
        """
            Step
        """
        # Update state first
        self._update()
        # And then make a step
        self.sim.step()

    def render(self,render_speedup=1.0):
        """
            Render
        """
        if (self.sec_sim >= render_speedup*self.sec_wall):
            self.viewer.render()

    def step_and_render(self,render_speedup=1.0):
        """
            Step and Render
        """
        self.step()
        self.render(render_speedup=render_speedup)
        
    def reset(self):
        """
            Reset
        """
        # Reset simulation
        self.sim.reset()
        # Update once
        self._update()
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
                (self.tick,self.sec_wall,self.sec_sim))

