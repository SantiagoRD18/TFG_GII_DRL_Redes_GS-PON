from abc import ABC, abstractmethod
from stable_baselines3.common.vec_env import DummyVecEnv

import modules.model_manager as model_manager
import gymnasium as gym

class BaseAgent(ABC):
    def __init__(self, num_ont, TxRate, temp_ciclo, B_guaranteed, seed):
        self.env_id = ""
        self.vec_env = []
        self.seed = seed
        self.num_ont = num_ont          #N_ONTS en sim
        self.TxRate = TxRate      #R_tx en sim
        self.temp_ciclo = temp_ciclo                      #T_CICLO en sim

        self.B_available = TxRate*temp_ciclo
        self.B_guaranteed = B_guaranteed  # w_sla en sim
        self.B_max = B_guaranteed*temp_ciclo


    def make_env(self):
        env = gym.make(self.env_id, render_mode = None, seed = self.seed, num_ont = self.num_ont, TxRate = self.TxRate, B_guaranteed = self.B_guaranteed, n_ciclos = self.lim_ciclos)
        return env


    def create_model(self, env_id, num_envs, algorithm, lim_ciclos):
        self.env_id = env_id
        self.num_envs = num_envs
        self.lim_ciclos = lim_ciclos
        self.vec_env = DummyVecEnv([self.make_env for _ in range(num_envs)])
        self.model = model_manager.create_model(self.vec_env, algorithm)


    def load_model(self, env_id, filename, algorithm, lim_ciclos):
        self.env_id = env_id
        self.lim_ciclos = lim_ciclos
        self.vec_env = self.make_env()
        self.model = model_manager.load_model(filename, self.vec_env, algorithm)


    def save_model(self):
        filename = input("Nombre del fichero del modelo: ")
        return model_manager.save_model(self.model, filename)