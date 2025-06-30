from packages.configuration.parameters import *
from packages.classes.ParetoGenerator import ParetoGenerator
from packages.classes.TramaEthernet import TramaEthernet
from packages.classes.EstadisticasWelford import EstadisticasWelford
import numpy as np
import csv

## Generador de tráfico
class GeneraTrafico:
    # Simula por una parte la capa de aplicación que va generando daos a una velocidad R_datos, 
    # y por otra simula la capa de transporte que va segmentando los datos en datagramas de 1500 Bytes
    def __init__(self, env, id, carga, seed_1=None, seed_2=None):
        # self.colas es un array que va guardando los paquetes en un bitarrray uno detrás de otro.
        # self.bytes descartados va contando los bytes descartados en total en la ONT.
        # self.bytes generados va contando los bytes generados en total en la ONT.
        self.env = env
        self.id = id
        self.colas = []
        self.colas_longitudes = []
        self.Bytes_generados=0
        self.Bytes_descartados=0
        self.paquetes_generados=0
        self.paquetes_descartados=0
        self.id_paq = 0
        self.carga_onu = 0 # carga de la onu en bps
        self.i_on_total = []
        self.i_off_total = []
        self.retardo_estadisticas = []

        # Generador de números de pareto
        if (multiples_colas):
            carga_real = carga-.0448
        else : 
            carga_real=carga
        self.rng_on = np.random.RandomState(seed_1)
        self.rng_off = np.random.RandomState(seed_2)
        pareto_generator_on_class = ParetoGenerator()
        pareto_generator_off_class = ParetoGenerator()
        self.generador_pareto_on = pareto_generator_on_class.pareto_generator(self.rng_on, a_on, m_on)
        self.generador_pareto_off = pareto_generator_off_class.pareto_generator(self.rng_off, a_off, m_on*(1-carga_real)/carga_real)


        self.action = self.env.process(self.generador_uniforme_paquetes())
        for i in range(len(N_SOURCES)):
            for j in range(N_SOURCES[i]):
                self.action = env.process(self.generador_pareto_paquetes(tamano_payload[i]))


        for i in range(N_COLAS):
            # Cada ONT tiene N_COLAS colas. Aquí inicializamos las colas
            self.colas.append([])
            self.colas_longitudes.append(0)
            self.retardo_estadisticas.append(EstadisticasWelford())


    def encolador_colas_separadas(self, paquete, prioridad):
        ## Encolador que usa el método de las colas separadas
        # Imput: La variable prioridad es un entero que designa el índice de la cola en el array self.colas.

        # Calculamos longitud de la cola actualmente
        lon_cola_total = sum(self.colas_longitudes)
        if lon_cola_total + paquete.len <= L_BUFFER_ONTS//N_COLAS:
            # Si el tamaño de paquete a encolar más el tamaño de datos en cola no supera el máximo de la cola,
            # Añadimos el paquete a la cola
            self.colas[prioridad].append(paquete)
            self.colas_longitudes[prioridad] += paquete.len
        else:
            # Si el paquete no cabe en la cola, se descarta. 
            self.Bytes_descartados += paquete.len/8
            self.paquetes_descartados += 1

     
    def encolador_prioridad_colas(self, paquete, prioridad):
        ## Encolador que usa el método de prioridad de colas
        # Input: La variable prioridad es un entero que designa el índice de la cola en el array self.colas. 
        # Nota: asumimos que todos los paquetes son de 1500 Bytes, y todas las cabeceras son de 26 Bytes

        # Calculamos longitud total de las colas
        lon_cola_total = sum(self.colas_longitudes)

        if lon_cola_total + paquete.len <= L_BUFFER_ONTS:
            # Si la suma de todos los arrays en self.colas, más la longitud del paquete no supera el máximo, el paquete se añade en la cola correspondiente a su prioridad. 
            self.colas[prioridad].append(paquete)
            self.colas_longitudes[prioridad] += paquete.len
        elif(prioridad==N_COLAS-1):
            # Si el paquete es de la menor prioridad de todas y no queda sitio, lo descartamos. 
            # Si tenemos cola única y el paquete no cabe, se descarta.
            self.Bytes_descartados += paquete.len/8
            self.paquetes_descartados += 1

        else:
            # Si el paquete tiene una prioridad que no es la mínima, vamos recorriendo las prioridades
            # para ver donde lo podemos encajar
            bits_eliminados = 0
            for i in range(N_COLAS-1, prioridad, -1):
                #watch
                # if(self.env.now >.4 and lon_cola_total%100==0):
                #     print(YELLOW + f"(t={(self.env.now):,.12f}ns) (ONT id = {self.id}), iteramos prioridad ={i}/{prioridad} ------------- " + RESET)
                #     print(YELLOW + f"(t={(self.env.now):,.12f}ns) (ONT id = {self.id}), queremos meter paquete ({paquete.len} bits, prioridad= {prioridad}) en lon_cola_total = {lon_cola_total}" + RESET)
                #     print(YELLOW + f"(t={(self.env.now):,.12f}ns) (ONT id = {self.id}) colas: cola 0 = {len(self.colas[0])}, cola 1 = {len(self.colas[1])}, cola 2 = {len(self.colas[2])} " + RESET)

                # Por cada cola ...
                for j in range(len(self.colas[i])):
                    # ... va extrayendo paquetes hasta hacer hueco para el nuevo paquete
                    bits_eliminados += self.colas[i][0].len
                    self.colas_longitudes[i] -= self.colas[i][0].len
                    self.colas[i].pop(0)
                    #watch
                    # if(self.env.now >.4):
                    #     print(YELLOW + f"(t={(self.env.now):,.12f}ns) (ONT id = {self.id}) colas: cola 0 = {len(self.colas[0])}, cola 1 = {len(self.colas[1])}, cola 2 = {len(self.colas[2])}, paquete retirado de la cola {i}, hemos retirado {j+1} paqs ({bits_eliminados} bits)" + RESET)
                
                    if(bits_eliminados>=paquete.len):
                        self.colas[prioridad].append(paquete)
                        self.colas_longitudes[prioridad] += paquete.len
                        #watch
                        # if(self.env.now >.4):
                        #     print(YELLOW + f"(t={(self.env.now):,.12f}ns) (ONT id = {self.id}) colas: cola 0 = {len(self.colas[0])}, cola 1 = {len(self.colas[1])}, cola 2 = {len(self.colas[2])}, paquete ({paquete.len} bits) añadido" + RESET)
                        break
                if(bits_eliminados>=paquete.len):
                    break

    def generador_pareto_paquetes(self, tam_paq):
        # Genera paquetes distribuidos en el tiempo de acuerdo con la distr. de Pareto II
        # Los va asignando distintas prioridades.
        # Cada ráfaga tiene la misma prioridad
        if (multiples_colas):
            prioridad=1
        else: 
            prioridad=0
        while True:
            # Calculamos tiempo de ráfaga
            i_on_rough = next(self.generador_pareto_on)
            self.i_on_total.append(i_on_rough)

            # Calculamos tamaño de la ráfaga en Bytes
            tamano_rafaga = int(i_on_rough/((tamano_cabecera+tam_paq)/R_datos))
            # # Calculamos tiempo real de la ráfaga
            # i_on = tamano_rafaga*(tamano_cabecera+tamano_payload)/R_datos

            # Comienza la ráfaga
            for i in range(tamano_rafaga):
                # Introducimos tiempo de espera del paquete enviado
                yield self.env.timeout(sum(N_SOURCES)*N_ONTS*(tam_paq + tamano_cabecera)/(R_datos))
                # La ráfaga va a consistir de paquetes encadenados que se transmiten
                paquete = TramaEthernet(self.id_paq, 'L', self.id, tam_paq + tamano_cabecera, self.env.now, prioridad)
                self.id_paq += 1

                # Contabilizamos paquetes                
                self.Bytes_generados +=(tam_paq + tamano_cabecera)/8
                self.paquetes_generados += 1
                
                # Metemos el paquete en la cola correspondiente, en función de la configuración
                if insertionmethod_separatequeue0_priorityqueue1==False:
                    # Método de inserción de colas separadas
                    self.encolador_colas_separadas(paquete, prioridad)
                else:
                    # Método de inserción de prioridad de colas
                    self.encolador_prioridad_colas(paquete, prioridad)

            if(prioridad==1):
                prioridad=2
            elif(prioridad==2):
                prioridad=1

            # Generamos tiempo de silencio
            i_off = next(self.generador_pareto_off)
            self.i_off_total.append(i_off)


            # Esperamos tiempo de silencio
            yield self.env.timeout(sum(N_SOURCES)*N_ONTS*i_off)



    def generador_uniforme_paquetes(self):
        # Genera paquetes de manera uniforme en el tiempo (frecuencia y tamaño fijos)
        # Los va asignando a las distintas prioridades de forma uniforme
        while True:
            # Esperamos el tiempo que tarda en generarse el paquete
            yield self.env.timeout(125e-6)
            len_payload = 70*8-tamano_cabecera # tamaño de payload
            prioridad = 0  # anotamos prioridad 0 para los paquetes generados uniformemente
            paquete = TramaEthernet(self.id_paq, 'L', self.id, len_payload, self.env.now, prioridad) # encapsulamos datos con cabecera ethernet
            self.id_paq += 1

            # Metemos el paquete en la cola correspondiente, en función de la configuración
            if insertionmethod_separatequeue0_priorityqueue1==False:
                # Método de inserción de colas separadas
                self.encolador_colas_separadas(paquete, prioridad)
            else:
                # Método de inserción de prioridad de colas
                self.encolador_prioridad_colas(paquete, prioridad)
            
            # Contabilizamos paquetes  
            self.Bytes_generados += 70
            self.paquetes_generados += 1
