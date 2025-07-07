from scipy.stats import norm

class EstadisticasWelford:
    # Va calculando la media y la varianza de una secuencia de números
    # usando el algoritmo de Welford.
    def __init__(self):
        self.n = 0 # Número de muestras
        self.media = 0.0 # Va acumulando la media
        self.M2 = 0.0 # Agrega el cuadrado de las diferencias

    def actualizar(self, x):
        # Actualiza la media y la varianza con una nueva muestra
        self.n += 1
        delta = x - self.media
        self.media += delta / self.n
        delta2 = x - self.media
        self.M2 += delta * delta2

    def varianza(self):
        # Devuelve la varianza de las muestras
        if self.n < 2:
            return float('nan')
        return self.M2 / self.n

    def desviacion_tipica(self):
        # Devuelve la desviación típica de las muestras
        return self.varianza()**0.5
    
    def intervalo_confianza(self, nivel_confianza=.95):
        # Devuelve el intervalo de confianza de las muestras
        # Calculamos el nivel de significación
        alpha = 1.0 - nivel_confianza
        # Calculamos el Valor crítico
        valor_critico = norm.ppf(1.0 - alpha / 2.0)
        confidence_interval = (self.media - valor_critico*self.desviacion_tipica()/self.n, self.media + valor_critico*self.desviacion_tipica()/self.n)
        return confidence_interval
        