from packages.configuration.parameters import *
import simpy

## Enlace
class Enlace:
    # Simula un enlace entre un dispositivo fuente y uno o más dispositivos de destino
    def __init__(self, env, n_destinatarios = 1, capacidad = simpy.core.Infinity, delay = T_propagacion):
        # Si n_destinatarios=1, se trata de un enlace punto a punto.
        # Si n_destinatarios>1, se trata de un enlace broadcast
        # Si capacidad=sipmy.core.Infinity, eso quiere decir que un número ilimitado de mensajes
        # pueden estar en el enlace al mismo tiempo.
        # delay representa el retardo de propagación del enlace
        self.env = env
        self.delay = delay 
        self.capacidad = capacidad 
        self.store = simpy.Store(env, self.capacidad)
        self.n_destinatarios = n_destinatarios

    def enviar_con_retardo(self, value):
        # Añade un retardo al envío del mensaje
        yield self.env.timeout(self.delay)
        for i in range(self.n_destinatarios):
            self.store.put(value) # Aquí se envía el mensaje

    def enviar(self, value):
        # Para enviar un mensaje.
        # Llama a la función enviar_con_retardo para simular el retardo de propagación.
        self.env.process(self.enviar_con_retardo(value))

    def get(self):
        # Se llama a esta función cuando se reciba un mensaje. El mensaje se guarda en el self.store
        return self.store.get()
