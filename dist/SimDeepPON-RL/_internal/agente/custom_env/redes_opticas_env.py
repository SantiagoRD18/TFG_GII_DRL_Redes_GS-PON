import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time

from agente.custom_env.traficoparetopython import simular_trafico

class RedesOpticasEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, seed, num_ont, TxRate, B_guaranteed, n_ciclos, load, render_mode=None):
        self.num_ont = num_ont
        self.temp_ciclo = 0.002
        self.B_available = TxRate * self.temp_ciclo
        self.B_max = B_guaranteed * self.temp_ciclo

        self.max_queue = np.full(self.num_ont, 800e6)
        self.b_excess = 0

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_ont,), dtype=np.float64)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_ont,), dtype=np.float64)

        self.step_durations = []

        self.load = load

        self.B_alloc = np.zeros(self.num_ont)
        self.trafico_entrada = np.zeros(self.num_ont)
        self.B_demand = np.zeros(self.num_ont)
        self.prev_demand = np.zeros(self.num_ont)

        self.rng = np.random.default_rng(seed)
        
        self.instantes=0
        self.n_ciclos=n_ciclos-1
        self.load = load
        self.state = None
        self.onts = None

    def _get_obs(self):
        return self.B_demand / self.max_queue
    
    def _get_info(self):
        return {
            'B_available': self.B_available,
            'input_traffic': self.trafico_entrada.copy(),
            'B_alloc': self.B_alloc.copy(),
            'B_demand': self.B_demand.copy()
        }

    def _calculate_reward(self):
        delta_cola = self.prev_demand - self.B_demand
        delta_cola = np.mean(delta_cola / self.max_queue)

        uso_bw = np.sum(self.B_alloc) / np.sum(self.B_max)

        max_excess = np.sum(self.B_max) - self.B_available
        lim_olt = self.b_excess / max_excess
        self.b_excess = 0

        reward = (
            0.5 * delta_cola +
            1.3 * uso_bw +
            0.5 * lim_olt
        )

        return reward

    def step(self, action):
        start_time = time.time()

        self.prev_demand = np.copy(self.B_demand)

        self.trafico_entrada_por_ciclo, self.onts = simular_trafico(self.onts, self.load)
        self.trafico_entrada = self.trafico_entrada_por_ciclo

        self.B_alloc = np.clip(action, 0, 1) * self.B_max

        total = np.sum(self.B_alloc)
        if total > self.B_available:
            self.b_excess = total - self.B_available
            self.B_alloc *= (self.B_available / total)

        for i in range(self.num_ont):
            self.B_demand[i] += self.trafico_entrada[i]
            
            self.B_alloc[i] = min(self.B_alloc[i], self.B_demand[i])
            self.B_alloc[i] = min(self.B_alloc[i], self.B_max[i])

            self.B_demand[i] -= self.B_alloc[i]

            self.B_demand[i] = np.clip(self.B_demand[i], 0, self.max_queue[i])

        reward = self._calculate_reward()

        done = True if self.instantes >= self.n_ciclos else False

        self.instantes+=1

        end_time = time.time()
        step_duration = end_time - start_time
        self.step_durations.append(step_duration)

        info = self._get_info()

        return self._get_obs(), reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.instantes = 0

        _, self.onts = simular_trafico(load = self.load)
        
        self.B_demand = np.zeros(self.num_ont)
        self.trafico_entrada = np.zeros(self.num_ont)
        self.B_alloc = np.zeros(self.num_ont)

        self.prev_demand = np.zeros(self.num_ont)
        self.b_excess = 0

        self.rng = np.random.default_rng(seed)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info


from gymnasium.envs.registration import register

register(
    id="RedesOpticasEnv-v0",
    entry_point="agente.custom_env.redes_opticas_env:RedesOpticasEnv",
)