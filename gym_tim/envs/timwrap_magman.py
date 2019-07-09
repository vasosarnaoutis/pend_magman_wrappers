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
        dense_state[0]= (dense_state[0]+0.035)/0.14
        dense_state[1]= dense_state[1]/0.4
        ds = dense_state.reshape((-1, 2))
        batch_distance = np.abs(np.expand_dims(ds, axis=1) - self.xy) / self.member_width
        membership = np.clip(np.min(1 - batch_distance, axis=-1), 0, 1)
        return np.squeeze(membership)

class TimMagWrap(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    def __init__(self):
        ### Run Simulation
        self.dynamics_env= cb.MagmanBenchmark(magnets=2, sampling_time=0.02, max_seconds=2.5, 
                            reward_type=cb.RewardType.QUADRATIC, do_not_normalize=True)
        self.state_mapping = TriangularBFFeature2D(basis_functions_per_dimension=21)
        self.basis_fun_states = False
        self.reset_to_random=True
        self.max_speed=0.4
        self.max_pos = 0.105
        self.min_pos = -0.035
        self.max_torque=0.3
        self.min_torque= -0.3
        self.target_state = np.array([0.035, 0.])
        self.target_action=np.array([0. for _ in range(2)])
        self.state_penalty_weights=np.array([1., 0.]),
        self.action_penalty_weights=np.array([0. for _ in range(2)]),
        self.viewer = None
        high = np.array([self.max_pos,  self.max_speed])
        low  = np.array([self.min_pos, -self.max_speed])
        self.action_space = spaces.Box(low=self.min_torque, high=self.max_torque, shape=self.dynamics_env.action_shape, dtype=np.float32)
        if self.basis_fun_states == True:
            self.observation_space = spaces.Box(low = np.zeros(21*21), high = np.ones(21*21), dtype=np.float32)
        else :
            self.observation_space = spaces.Box(low =low, high=high, dtype=np.float32)
        self.step_counter = 0
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        self.step_counter += 1         
        state = self.state # th := theta , th_dot := velocity
        self.dynamics_env.reset_to_specific_state(state)       
        u += 0.3 
        u = np.clip(u, 0, 2*self.max_torque)
        self.last_u = u  # for rendering
        state, _, terminate, _, = self.dynamics_env.step(u)
        self.state = np.array(state)
        reward = self._get_reward()
        terminate = self._is_terminate(terminate)
        
        return self._get_obs(), reward, terminate, {}

    def _get_reward(self):
        #Quadratic Reward
        #Wx * |x - xr|^2 + Wu * |u - ur|^2
        reward = float(-1* np.sum(self.state_penalty_weights * np.abs((self.state - self.target_state)) + 
                  self.action_penalty_weights* np.abs((self.last_u - self.target_action))))
        return reward


    def _is_terminate(self, terminate):
        if terminate== True or self.step_counter > 125:
            terminate = True
        return terminate     


    def reset(self):
        initial_state = np.array([0., 0.])
        self.state = initial_state
        if self.reset_to_random==True:
            high = np.array([0.075,  self.max_speed/10])
            low  = np.array([0, -self.max_speed/10])
            self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None
        self.step_counter=0
        return self._get_obs()


    def reset_to_specific_state(self, state):
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
        screen_width = 600
        screen_height = 400

        world_width = 0.14
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 30.0
        polelen = 60
        cartwidth = 10.0
        cartheight = 10.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)
            #magnet1
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            magnet1 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            magnet1.set_color(.8,.6,.4)
            self.magnettrans1 = rendering.Transform()
            magnet1.add_attr(self.magnettrans1)
            self.viewer.add_geom(magnet1)
            #magnet2
            magnet2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            magnet2.set_color(.8,.6,.4)
            self.magnettrans2 = rendering.Transform()
            magnet2.add_attr(self.magnettrans2)
            self.viewer.add_geom(magnet2)
            #target
            targ = rendering.make_circle(polewidth/10)
            targ.set_color(.8,.6,.4)
            self.targtrans = rendering.Transform()
            targ.add_attr(self.targtrans)
            
            self.viewer.add_geom(targ)

        if self.state is None: return None
        x = self.state
        cartx = (x[0]+0.035)*scale # MIDDLE OF BALL
        self.carttrans.set_translation(cartx, carty)
        self.magnettrans1.set_translation((0.035+0.025)*scale,50)
        self.magnettrans2.set_translation((0.035+0.05)*scale,50)
        self.targtrans.set_translation((0.035+self.target_state[0])*scale,carty)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def reset_model_to_specific_state(self, state):
        self.state = state
        self.last_u = None
        return self._get_obs()

    def get_all_states(self):
        return self.state

