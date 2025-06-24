from .base_agent import BaseAgent

import numpy as np

class SimulatorAgent(BaseAgent):
    def __init__(self, num_ont, TxRate, temp_ciclo, B_guaranteed):
        super().__init__(num_ont, TxRate, temp_ciclo, np.array(B_guaranteed), 42)
        self.load_model("model.zip", "ppo")

    def get_prediction(self, obs, ont_id):
        prediction, _ = self.model.predict(obs, deterministic=True)
        return prediction[ont_id]