from abc import ABC, abstractmethod
from custom_env.redes_opticas_env import RedesOpticasEnv
from stable_baselines3.common.vec_env import DummyVecEnv

import modules.model_manager as model_manager
import gymnasium as gym

class BaseAgent(ABC):
    def __init__(self, num_ont, v_max_olt, T, vt_contratada):
        self.env_id = ""
        self.seed = 42
        self.num_ont = num_ont          #N_ONTS en sim
        self.v_max_olt = v_max_olt      #R_tx en sim
        self.T = T                      #T_CICLO en sim

        self.OLT_Capacity = v_max_olt*T
        self.vt_contratada = vt_contratada
        self.Max_bits_ONT = vt_contratada*T


    def make_env(self):
        env = gym.make(self.env_id, render_mode = None, seed = self.seed, num_ont = self.num_ont, v_max_olt = self.v_max_olt, vt_contratada = self.vt_contratada, n_ciclos = self.lim_ciclos)
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