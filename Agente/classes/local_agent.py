from base_agent import BaseAgent
from modules import plotter
from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np
import time

class LocalAgent(BaseAgent):
    def __init__(self, num_ont, v_max_olt, T, vt_contratada, seed):
        super().__init__(num_ont, v_max_olt, T, vt_contratada, seed)


    def train_model(self, timesteps):
        start_time = time.time()
        self.model.learn(total_timesteps=timesteps)
        end_time = time.time()
        training_time = end_time - start_time
        print(f"El tiempo de entrenamiento fue de {training_time} segundos.")


    def exec_simulation(self, n_ciclos, num_tests):
        self.n_ciclos = n_ciclos
        self.episode_info = []  
        self.list_ont = []
        self.list_ont_2 = []
        self.list_pendiente=[]
        self.estados_on_off_recolectados = []

        test_env = self.make_env()

        for _ in range(num_tests):

            obs, _ = test_env.reset()
            _states = None
            done = False

            while not done:
                action, _states = self.model.predict(obs, state=_states, deterministic=True)
                obs, _, done, _, info = test_env.step(action)

                self.episode_info.append(info)

                self.list_ont.append(info['trafico_entrada'])
                self.list_ont_2.append(info['trafico_salida'])
                self.list_pendiente.append(info['trafico_pendiente'])
                self.estados_on_off_recolectados.append(info['trafico_IN_ON_actual'])

    

    def plot_results(self):
        trafico_entrada = plotter.process_traffic(self.list_ont, self.T)
        trafico_salida = plotter.process_traffic(self.list_ont_2, self.T)
        trafico_pendiente = plotter.process_traffic(self.list_pendiente, self.T)
        valores_instantes = plotter.calculate_instants(self.estados_on_off_recolectados, self.num_ont)

        for i in range(self.num_ont):
            plotter.plot_input_output(trafico_entrada[i], trafico_salida[i], i)
            plotter.plot_pareto(valores_instantes[i], i, self.n_ciclos)
            plotter.plot_pending(trafico_pendiente[i], i)