import gym
import slimevolleygym
import os

from stable_baselines.ppo1 import PPO1
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback

SEED = 721

env = gym.make("SlimeVolley-v0")
env.seed(SEED)
obs = env.reset()
done = False
total_reward = 0

model = PPO1.load("ppo1/best_model.zip")
obs = env.reset()
while not done:

    action = env.action_space.sample() #model.predict(obs)
    obs, reward, done, info = env.step(action)

    total_reward += reward



    env.render()
    print(info)

