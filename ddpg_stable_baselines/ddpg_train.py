import gym
import gym_tim
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

env = gym.make('tim_magman-v0')
#env = DummyVecEnv([lambda: env])

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, tensorboard_log='stable_baselines_ddpg/trained_models')


model.learn(total_timesteps=5e6,tb_log_name='magman')
model.save("stable_baselines_ddpg/trained_models/ddpg_magman60")


'''
#######  RUN ON ITERATIONS ################
training_breaks = np.linspace(0,10000000,101)
training_steps = training_breaks[-1]/(len(training_breaks)-1)
for steps_done in training_breaks:
    model.learn(total_timesteps=int(training_steps),tb_log_name='main_run')
    model.save("stable_baselines_ddpg/trained_models/ddpg_Pendulum"+str(int(steps_done)))
'''



'''
######## PLAY LEARNED NETWORK
del model # remove to demonstrate saving and loading

model = DDPG.load("stable_baselines_ddpg/ddpg_hop")

obs = env.reset()
dones=False
while dones is not True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
'''
