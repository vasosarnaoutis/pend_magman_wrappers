import gym
from gym import spaces, error, utils
from gym.utils import seeding
from os import path
# Import simulation_benchmark
import cor_control_benchmarks as cb
import numpy as np


class TriangularBFFeature2D(object):
    """ Converts a 2D location to a vector with the membership degrees of a gird of triangular basis functions

    This class assumes that input states are 2 dimensional and in the normalized domain [-1,1].
    It will fail (possibly silently) if these assumptions do not hold.

    """

    def __init__(self, basis_functions_per_dimension: int):
        """ Initialize the grid with basis_functions_per_dimension * basis_functions_per_dimension basis functions"""
        self.member_width = 2 / (basis_functions_per_dimension - 1)
        self.sparse_feature_size = int(basis_functions_per_dimension ** 2)
        x = y = np.linspace(-1, 1, basis_functions_per_dimension)
        self.xy = np.array(np.meshgrid(x, y)).T.reshape(1, self.sparse_feature_size, 2)

    def __call__(self, dense_state: np.ndarray) -> np.ndarray:
        """ Convert the dense (x,y) state to the sparse membership function state """
        dense_state[0]= dense_state[0]/np.pi
        dense_state[1]= dense_state[1]/30
        ds = dense_state.reshape((-1, 2))
        batch_distance = np.abs(np.expand_dims(ds, axis=1) - self.xy) / self.member_width
        membership = np.clip(np.min(1 - batch_distance, axis=-1), 0, 1)
        return np.squeeze(membership)


class TimPendWrap(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }
    def __init__(self):
        ### Run Simulation
        self.dynamics_check_pendulum = cb.PendulumBenchmark(max_voltage=2., do_not_normalize=True)
        self.state_mapping = TriangularBFFeature2D(basis_functions_per_dimension=21)
        self.basis_fun_states = False
        self.reset_to_0 = False
        self.max_speed=30
        self.max_torque=2.
        self.viewer = None
        high = np.array([np.pi, self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        if self.basis_fun_states == True:
            self.observation_space = spaces.Box(low = np.zeros(21*21), high = np.ones(21*21), dtype=np.float32)
        else :
            self.observation_space = spaces.Box(low =-high, high=high, dtype=np.float32)
        self.termination_with_max_steps = True
        self.step_counter = 0
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        self.step_counter += 1         
        state = self.state # th := theta , th_dot := velocity
        self.dynamics_check_pendulum.reset_to_specific_state(state)
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        state, _, _, _, = self.dynamics_check_pendulum.step(u)
        costs = np.absolute(state[0]) - np.pi - .1 * np.absolute(state[1]) - .01 * np.absolute(u) + self._healthy()
        self.state = np.array(state)
        terminate = self._is_healthy()
        return self._get_obs(), costs, terminate, {}

    def reset(self):
        self.step_counter = 0
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        if self.reset_to_0 == True :
            self.state= np.array([0,0])
        self.last_u = None
        return self._get_obs()

    def _is_healthy(self):
        state = self.state
        terminate = False
        if np.abs(state[0]) > np.pi*3/4 and np.abs(state[1]) > 20:
            terminate = True

        if self.termination_with_max_steps == True and self.step_counter > 199:
            terminate = True
        return terminate

    def _healthy(self):
        state = self.state
        health= 0
        if np.abs(state[0]) > np.pi*3/4 and np.abs(state[1]) > 20:
            health = -2
        '''
        if np.abs(state[0]) > np.pi*19/20 and np.abs(state[1]) < 2:
            health = 2
        '''
        return health

    def reset_to_specific_state(self, state):
        high = np.array([np.pi, 1])
        self.state = state
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        obs = np.array([theta, thetadot])
        if self.basis_fun_states == True:   
            obs= self.state_mapping(obs)
        return obs

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join("/home/vasos/tenv/lib/python3.6/site-packages/gym/envs/classic_control/assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] - np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def reset_model_to_specific_state(self, state):
        high = np.array([np.pi, 1])
        self.state = state
        self.last_u = None
        return self._get_obs()

    def get_all_states(self):
        return self.state

    def converge_states(self, three_states):
        states = np.array([np.arctan2(three_states[1], three_states[0]), three_states[2]])
        return states
