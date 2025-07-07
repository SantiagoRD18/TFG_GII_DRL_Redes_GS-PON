import numpy as np


import sys
import os

from agente.classes.sim_agent import SimulatorAgent


class AgentInterface:
    def __init__(self, num_ont, TxRate, temp_ciclo, B_guaranteed, B_available, B_max, max_queue, model_name):
        self.num_ont = num_ont
        self.TxRate = TxRate
        self.temp_ciclo = temp_ciclo
        self.B_guaranteed = B_guaranteed
        self.B_available = B_available
        self.B_max = B_max
        self.max_queue = max_queue
        self.model_name = model_name

        self.agent = SimulatorAgent(num_ont, TxRate, temp_ciclo, B_guaranteed, model_name)

    def predict_values(self, B_demand, ont_id):
        norm_b_demand = self.convert_data_from_sim(B_demand)
        predicted_b_alloc = self.convert_data_from_agent(self.agent.get_prediction(norm_b_demand, ont_id), ont_id)
        return predicted_b_alloc

    def _convert_data_from_sim(self, b_demand):
        return np.array(b_demand) / np.array(self.max_queue)
    
    def _convert_data_from_agent(self, norm_b_alloc, ont_id):
        return norm_b_alloc * self.B_max[ont_id]