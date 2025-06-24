from abc import ABC
from stable_baselines3.common.vec_env import DummyVecEnv
from modules import model_manager

import gymnasium as gym

class BaseAgent(ABC):
    def __init__(self, num_ont, TxRate, temp_ciclo, B_guaranteed, seed):
        self.env_id = ""
        self.vec_env = []
        self.seed = seed
        self.num_ont = num_ont
        self.TxRate = TxRate
        self.temp_ciclo = temp_ciclo

        self.n_ciclos = 0

        self.B_available = TxRate * temp_ciclo
        self.B_guaranteed = B_guaranteed
        self.B_max = B_guaranteed * temp_ciclo

        self.model = None


    def _make_env(self):
        env = gym.make(self.env_id, render_mode = None, seed = self.seed, num_ont = self.num_ont, TxRate = self.TxRate, B_guaranteed = self.B_guaranteed, n_ciclos = self.lim_ciclos)
        return env


    def create_model(self, env_id, num_envs, algorithm, lim_ciclos):
        self.env_id = env_id
        self.num_envs = num_envs
        self.lim_ciclos = lim_ciclos
        self.vec_env = DummyVecEnv([self._make_env for _ in range(num_envs)])
        self.model = model_manager.create_model(self.vec_env, algorithm)


    def load_model(self, filename, algorithm):
        self.model = model_manager.load_model(filename, algorithm)


    def save_model(self, filename):
        return model_manager.save_model(self.model, filename)