from base_agent import BaseAgent
from modules import plotter
from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np
import time

class LocalAgent(BaseAgent):
    def __init__(self, num_ont, TxRate, temp_ciclo, B_guaranteed, seed):
        super().__init__(num_ont, TxRate, temp_ciclo, B_guaranteed, seed)


    def train_model(self, timesteps):
        start_time = time.time()
        self.model.learn(total_timesteps=timesteps)
        end_time = time.time()
        training_time = end_time - start_time
        print(f"El tiempo de entrenamiento fue de {training_time} segundos.")


    def exec_simulation(self, n_ciclos, num_tests):
        self.n_ciclos = n_ciclos
        self.episode_info = []
        self.list_input_traffic = []
        self.list_b_alloc = []
        self.list_b_demand=[]
        self.list_on_off_states = []

        test_env = self.make_env()

        for _ in range(num_tests):

            obs, _ = test_env.reset()
            _states = None
            done = False

            while not done:
                action, _states = self.model.predict(obs, state=_states, deterministic=True)
                obs, _, done, _, info = test_env.step(action)

                self.episode_info.append(info)

                self.list_input_traffic.append(info['trafico_entrada'])
                self.list_b_alloc.append(info['trafico_salida'])
                self.list_b_demand.append(info['trafico_pendiente'])
                self.list_on_off_states.append(info['trafico_IN_ON_actual'])

    

    def plot_results(self):
        input_taffic = plotter.process_traffic(self.list_input_traffic, self.temp_ciclo)
        B_alloc = plotter.process_traffic(self.list_b_alloc, self.temp_ciclo)
        B_demand = plotter.process_traffic(self.list_b_demand, self.temp_ciclo)
        instant_values = plotter.calculate_instants(self.list_on_off_states, self.num_ont)

        for i in range(self.num_ont):
            plotter.plot_input_output(input_taffic[i], B_alloc[i], i)
            plotter.plot_pareto(instant_values[i], i, self.n_ciclos)
            plotter.plot_pending(B_demand[i], i)