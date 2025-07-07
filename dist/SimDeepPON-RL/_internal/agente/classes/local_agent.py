from agente.classes.base_agent import BaseAgent
from agente.modules import plotter
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path
import pandas as pd
import os

import time

class LocalAgent(BaseAgent):
    def __init__(self, num_ont, TxRate, temp_ciclo, B_guaranteed, seed):
        super().__init__(num_ont, TxRate, temp_ciclo, B_guaranteed, seed)


    def train_model(self, timesteps):
        self.timesteps = timesteps
        start_time = time.time()
        self.model.learn(total_timesteps = self.timesteps)
        end_time = time.time()
        training_time = end_time - start_time
        print(f"El tiempo de entrenamiento fue de {training_time} segundos.")


    def exec_simulation(self, n_ciclos, num_tests, env_id, load):
        self.env_id = env_id
        self.n_ciclos = n_ciclos
        self.episode_info = []
        self.list_input_traffic = []
        self.list_b_alloc = []
        self.list_b_demand=[]

        self.load = load

        test_env = DummyVecEnv([self._make_env])

        for _ in range(num_tests):

            obs = test_env.reset()
            _states = None
            done = False

            while not done:
                action, _states = self.model.predict(obs, state=_states, deterministic=True)
                obs, _, done, info = test_env.step(action)

                info = info[0]
                self.episode_info.append(info)

                self.list_input_traffic.append(info['input_traffic'].copy())
                self.list_b_alloc.append(info['B_alloc'].copy())
                self.list_b_demand.append(info['B_demand'].copy())
    

    def save_results(self, plot_results=False, filename=None):
        input_taffic = plotter.process_traffic(self.list_input_traffic, self.temp_ciclo)
        B_alloc = plotter.process_traffic(self.list_b_alloc, self.temp_ciclo)
        B_demand = plotter.process_traffic(self.list_b_demand, self.temp_ciclo)
        df = pd.DataFrame({
            'onts': range(self.num_ont),
            'trafico_entrada': input_taffic,
            'trafico_salida': B_alloc,
            'colas': B_demand
        })

        current_dir = Path(__file__).resolve().parent
        if filename is not None:
            df.to_csv(os.path.join(current_dir.parent, "logs", filename), index=False)
        else:
            df.to_csv(os.path.join(current_dir.parent, f"eval_log_{self.load}.csv"), index=False)

        if plot_results:
            for i in range(self.num_ont):
                plotter.plot_input_output(input_taffic[i], B_alloc[i], i)
                plotter.plot_pending(B_demand[i], i)