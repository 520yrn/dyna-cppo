# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 21:57:07 2025

@author: Ruining
"""
import safety_gymnasium

def make_safety_env(env_id: str = 'SafetyPointGoal1-v0', render_mode=None):
    """
    Factory function to create a Safety-Gymnasium environment.
    Environment documentations:
        https://safety-gymnasium.readthedocs.io/en/latest/components_of_environments/agents/point.html
        https://safety-gymnasium.readthedocs.io/en/latest/environments/safe_navigation/goal.html

    Args:
        env_id (str): Environment ID, e.g. 'SafetyPointGoal1-v0'
        render_mode (str or None): 'human' for window, 'rgb_array' for frames, None for headless (fastest)

    Returns:
        env (gym.Env): The created environment instance.
    """
    env = safety_gymnasium.make(env_id, render_mode=render_mode)
    return env

'''
env = safety_gymnasium.make('SafetyPointGoal1-v0', render_mode= "human")
obs, info = env.reset()
while True:
    act = env.action_space.sample()
    obs, reward, cost, terminated, truncated, info = env.step(act)
    if terminated or truncated:
        break
    env.render()
env.close()
'''