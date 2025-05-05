import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
from scipy.stats import pareto

class RedesOpticasEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None, seed=0, num_ont=3, v_max_olt=10e6, vt_contratada=10e6/10, n_ciclos=200):
        self.num_ont = num_ont
        self.v_max_olt = v_max_olt  # bps
        self.temp_ciclo = 0.002  # segundos
        self.OLT_Capacity = v_max_olt * self.temp_ciclo  # bits
        self.velocidadContratada = vt_contratada
        self.Max_bits_ONT=self.velocidadContratada*self.temp_ciclo

        self.observation_space = spaces.Box(low=0, high=self.Max_bits_ONT, shape=(self.num_ont,), dtype=np.float64)
        self.action_space = spaces.Box(low=-self.Max_bits_ONT, high=self.Max_bits_ONT, shape=(self.num_ont,), dtype=np.float64)

        self.step_durations = []
        self.trafico_entrada = []
        self.trafico_pareto_futuro = []
        self.trafico_salida = []
        self.trafico_pareto_actual = []
        self.trafico_pendiente = np.zeros(self.num_ont)

        self.rng = np.random.default_rng(seed)
        
        self.instantes=0
        self.n_ciclos=n_ciclos-1
        self.state = None
        self.reset()

    def _get_obs(self):
        obs = np.clip(self.trafico_entrada, 0, self.OLT_Capacity) / self.Max_bits_ONT
        return np.squeeze(obs)
    
    def _get_info(self):
        info = {
            'OLT_Capacity': self.OLT_Capacity,
            'trafico_entrada': self.trafico_entrada,
            'trafico_salida': self.trafico_salida,
            'trafico_IN_ON_actual': self.trafico_pareto_actual,
            'trafico_pendiente': self.trafico_pendiente
        }
        return info

    def calculate_pareto(self, num_ont=5, traf_pas=[]):
        alpha_on = 1.4
        alpha_off = 1.2
        vel_tx_max = self.v_max_olt*0.01
        trafico_futuro_valores = []
        lista_trafico_act = []
        trafico_actual_lista = [[] for _ in range(self.num_ont)]
        

        for i in range(num_ont):
            if not traf_pas:
                trafico_pareto = list(self.rng.pareto(alpha_on, size=1))
                trafico_pareto += list(self.rng.pareto(alpha_off, size=1))
            else:
                trafico_pareto = traf_pas[i]

            suma = sum(trafico_pareto)
            while suma < 2:
                trafico_pareto += list(self.rng.pareto(alpha_on, size=1)) + list(self.rng.pareto(alpha_off, size=1))
                suma = sum(trafico_pareto)

            traf_act = []
            suma = 0
            while suma < 2:
                traf_act.append(trafico_pareto.pop(0))
                suma = sum(traf_act)

            traf_fut = [0, 0]
            if len(traf_act) % 2 == 0:
                traf_fut[0] = 0
                traf_fut[1] = suma - 2
                traf_act[-1] -= traf_fut[1]
            else:
                traf_fut[0] = suma - 2
                traf_fut[1] = trafico_pareto[-1]
                traf_act[-1] -= traf_fut[0]

            trafico_actual_lista[i].append(traf_act)
            vol_traf_act = sum(traf_act[::2]) * vel_tx_max * 10e-3
            lista_trafico_act.append(vol_traf_act)
            trafico_futuro_valores.append(traf_fut)

        return lista_trafico_act, trafico_actual_lista, trafico_futuro_valores

    def _calculate_reward(self):
        reward = -sum(self.trafico_pendiente)
        return reward

    def step(self, action):
        start_time = time.time()

        self.trafico_entrada, self.trafico_pareto_actual, self.trafico_pareto_futuro = self.calculate_pareto(self.num_ont, self.trafico_pareto_futuro)

        self.trafico_salida = np.clip(action, 0, self.Max_bits_ONT)

        # Asegurar que si hay tráfico pendiente, se ajuste adecuadamente el tráfico de salida
        for i in range(self.num_ont):
            self.trafico_pendiente[i] +=  self.trafico_entrada[i] - self.trafico_salida[i]
            if self.trafico_pendiente[i] > 0:
                self.trafico_salida[i] = min(self.trafico_pendiente[i], self.Max_bits_ONT)
                self.trafico_pendiente[i] -= self.trafico_salida[i]

        if np.sum(self.trafico_salida) > self.OLT_Capacity:
            exceso = np.sum(self.trafico_salida) - self.OLT_Capacity
            self.trafico_salida -= (exceso / self.num_ont)

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
        self.trafico_entrada, self.trafico_pareto_actual, self.trafico_pareto_futuro = self.calculate_pareto(self.num_ont, self.trafico_pareto_futuro)
        self.trafico_salida = self.rng.uniform(low=self.Max_bits_ONT/10, high=self.Max_bits_ONT, size=self.num_ont).astype(np.float32)
        
        self.trafico_pendiente = np.zeros(self.num_ont)

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