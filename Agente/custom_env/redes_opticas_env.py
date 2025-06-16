import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time

from .traficoparetopython import simular_trafico

class RedesOpticasEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, seed, num_ont, TxRate, B_guaranteed, n_ciclos, render_mode=None):
        self.num_ont = num_ont
        self.temp_ciclo = 0.002  # segundos
        self.B_available = TxRate * self.temp_ciclo  # bits OLT
        self.B_max = B_guaranteed * self.temp_ciclo  # bits por ONT

        self.observation_space = spaces.Box(low=0, high=self.B_max, shape=(self.num_ont,), dtype=np.float64)
        self.action_space = spaces.Box(low=-self.B_max, high=self.B_max, shape=(self.num_ont,), dtype=np.float64)

        self.step_durations = []

        self.B_alloc = np.zeros(self.num_ont)
        self.trafico_entrada = np.zeros(self.num_ont)
        self.B_demand = np.zeros(self.num_ont)

        self.rng = np.random.default_rng(seed)
        
        self.instantes=0
        self.n_ciclos=n_ciclos-1
        self.state = None
        self.onts = None

    def _get_obs(self):
        obs = np.clip(self.trafico_entrada, 0, self.B_available) / self.B_max
        return np.squeeze(obs)
    
    def _get_info(self):
        info = {
            'B_available': self.B_available,
            'trafico_entrada': self.trafico_entrada,
            'B_alloc': self.B_alloc,
            'B_demand': self.B_demand
        }
        return info

    def _calculate_reward(self):
        reward = -sum(self.B_demand)
        return reward

    def step(self, action):
        start_time = time.time()

        self.trafico_entrada_por_ciclo, self.onts = simular_trafico(self.onts)
        self.trafico_entrada = self.trafico_entrada_por_ciclo

        self.B_alloc = np.clip(action, 0, self.B_max)

        # Asegurar que si hay tráfico pendiente, se ajuste adecuadamente el tráfico de salida
        for i in range(self.num_ont):
            self.B_demand[i] +=  self.trafico_entrada[i] - self.B_alloc[i]
            if self.B_demand[i] > 0:
                self.B_alloc[i] = min(self.B_demand[i], self.B_max[i])
                self.B_demand[i] -= self.B_alloc[i]

        if np.sum(self.B_alloc) > self.B_available:
            exceso = np.sum(self.B_alloc) - self.B_available
            self.B_alloc -= (exceso / self.num_ont)

        reward = self._calculate_reward()

        if self.instantes==self.n_ciclos:
            done=True
        else:
            done=False

        self.instantes+=1

        end_time = time.time()
        step_duration = end_time - start_time
        self.step_durations.append(step_duration)

        info = self._get_info()

        return self._get_obs(), reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.instantes = 0

        _, self.onts = simular_trafico()
        
        self.B_demand = np.zeros(self.num_ont)
        self.trafico_entrada = np.zeros(self.num_ont)
        self.B_alloc = np.zeros(self.num_ont)

        self.rng = np.random.default_rng(seed)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info


from gymnasium.envs.registration import register

register(
    id="RedesOpticasEnv-v0",
    entry_point="custom_env.redes_opticas_env:RedesOpticasEnv",
    max_episode_steps=300,
)