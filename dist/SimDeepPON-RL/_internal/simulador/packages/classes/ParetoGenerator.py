## Generador de números de Pareto   
class ParetoGenerator:
    # Esta clase de por sí no hace nada, usamos el método pareto_generator para generar valores que sigan una distribución de Pareto
    def __init__(self):

        pass

    def pareto_generator(self,rng, a, m):
        # Generador de números de Pareto
        while True:
            yield (rng.pareto(a) + 1) * m
