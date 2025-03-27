from packages.configuration.parameters import *
from packages.classes.MensajeGate import MensajeGate
from packages.classes.MensajeReport import MensajeReport
from packages.classes.TramaEthernet import TramaEthernet
from packages.classes.EstadisticasWelford import EstadisticasWelford

import csv
import time
import os
import random 
import time 
import torch
import torch.nn as nn

## Definir el modelo de la red neuronal
class DNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

## OLT
class OLT:
    # Simula la OLT
    def __init__(self, env, splitter_in, splitter_out):
        self.env = env
        self.splitter_in = splitter_in # enlace que representa el Splitter en sentido Upstream
        self.splitter_out = splitter_out # enlace que representa el Splitter en sentido Downstream
        self.B_demand = [] # vector que representa la ventana asignada a cada ONT en bits
        self.B_alloc = [] # vector que representa la ventana asignada a cada ONT en bits
        self.B_alloc_acum = [] # vector que representa la ventana acumulada asignada a cada ONT en bits
        self.B_alloc_acum_aux = [] # vector que representa la ventana acumulada asignada a cada ONT en bits
        self.n_alloc = [] # vector que representa el número de veces que hemos asignado una ventana a cada ONT
        self.T_alloc = [] # vector que recoge los tiempos de transmisión asignados a cada ONT
        self.t_inicio_tx = [] # vector que representa el tiempo de inicio de transmisión de cada ONT
        self.colas_tamanos = [] # Registro del tamaño de cada cola en bits
        self.w_sla = []         # Ponderaciones de cada SLA
        self.B_max = []    # BW máximo asignado a cada onu, en número de tramas enviables
        self.B_max_1 = []    # BW máximo asignado a cada onu, en número de tramas enviables
        self.contador_paquetes_recibidos_olt = 0 # cuenta el total de paquetes llegados a la olt
        self.contador_Bytes_recibidos_olt = 0 # cuenta el total de Bytes llegados a la olt (payload+cabeceras, sin contar reports)
        self.contador_gates = 0 # cuenta el total de gates enviados por la olt
        self.retardos_estadisticas = [] # vector que recoge las estadísticas de retardo de cada ONU
        self.action = env.process(self.escucha_splitter(self.env))
        self.action = env.process(self.enviar_gate_inicial(self.env))
        
        self.T_alloc_acum = []  
        self.T_alloc_acum_total = []    
        self.B_guaranteed = [] # Aca se guardará el ancho de banda garantizado (70 (138K bits)-75 mbs promedio)
        self.B_alloc_acum_MBS = []
        
        for i in range(N_ONTS):
            # Iniicalizamos variables
            self.colas_tamanos.append([])
            self.retardos_estadisticas.append([])
            self.w_sla.append(1)        # Suponemos para toda ONUs tenemos un SLA_0 donde w=1
            self.B_demand.append(0)
            self.B_alloc.append(B_inicial)
            self.B_alloc_acum.append(B_inicial)
            self.B_alloc_acum_aux.append(B_inicial)
            self.n_alloc.append(1)
            self.T_alloc.append(0)
            self.T_alloc_acum.append(0)
            self.T_alloc_acum_total.append(0)
            self.t_inicio_tx.append(0)

        B_AVAILABLE = T_AVAILABLE * R_tx

        for i in range(N_ONTS):
            # Calculamos el BW máximo para cada ONU y para cualquier ciclo
            self.B_max.append(B_AVAILABLE * self.w_sla[i] / sum(self.w_sla))
            self.B_guaranteed.append(138000) # Cambiar a 600 mb con 10 gb
            
            if watch_on == True:
                print(PURPLE + f"B_max (ont {i}, sla = {self.w_sla[i]}) = {self.B_max[i] / 8:,.0f} Bytes ({self.B_max[i] / R_tx} s)" + RESET)

        if watch_on == True:
            print(PURPLE + f"B_max (total ONTs) = {sum(self.B_max) / 8:,.0f} Bytes ({sum(self.B_max) / R_tx} s)" + RESET) # B_max es el valor a "balancear"

    def procesa_report(self, env, mensaje_report):
        ## Método que actualiza el ancho de banda y los tiempos de inicio asignados a las ONTs
        tiempo_inicio = time.time()
        if not hasattr(self, 'B_alloc_acum_MBS'):
            self.B_alloc_acum_MBS = [0] * int(N_ONTS)  # N es el número total de ont_id, convertido a int
        print("Longitud de self.B_demand:", len(self.B_demand))
        print("Valor de ont_id:", ont_id)
        
            
        # Verificar si ont_id está dentro del rango válido
        if ont_id < 0 or ont_id >= len(self.B_demand):
            print("Error: ont_id fuera de rango")
            return
        
        # Actualización de tiempos de transmisión para cada ONU
        # Actualizamos el registro en la OLT en el que se registra la cola de cada ONU
        colas_tamanos = mensaje_report.colas_tamanos
        ont_id = mensaje_report.mac_src
        for i in range(N_COLAS):
            self.colas_tamanos[ont_id][i] = colas_tamanos[i]

        # Actualizamos la demanda de cada ONU
        self.B_demand[ont_id] = sum(self.colas_tamanos[ont_id]) # B_demand es el valor por pedir. Esto es lo que se puede predecir.

        # Predicción de B_max dado B_demand
        input_data = str(self.B_demand[ont_id]) + ',' + ont_id + ',' + '0.7' # Codificar los valores como cadena para la entrada del modelo
        input_tensor = torch.tensor([input_data], dtype=torch.float32) # Convertir los datos de entrada en un tensor
        predicted_B_max = model(input_tensor).item() # Obtener la predicción del modelo
        self.B_max_1[ont_id] = predicted_B_max # Actualizar B_max con la predicción

        # Actualizamos el ancho de banda que a cada ONT se le permite transmitir, según IPACT
        self.B_alloc[ont_id] = min(self.B_demand[ont_id], self.B_max_1[ont_id]) + tamano_report # B_max se asigna aquí. Aquí es donde se debe balancear
        self.B_alloc_acum[ont_id] += self.B_alloc[ont_id] - tamano_report 
        self.B_alloc_acum_aux[ont_id] += self.B_alloc_acum[ont_id] * 8 * 1e-6    
        self.n_alloc[ont_id] += 1

        # Actualizamos el tiempo de transmisión que a cada ONT se le permite transmitir, según IPACT
        self.T_alloc[ont_id] = self.B_alloc[ont_id] / R_tx 
        self.T_alloc_acum[ont_id] += self.T_alloc[ont_id] 
        self.T_alloc_acum_total[ont_id] += self.T_alloc[ont_id] + self.t_inicio_tx[ont_id] + T_GUARDA + T_propagacion

        # Actualización de tiempos de inicio para cada ONU
        ont_id_prev = (ont_id - 1) % N_ONTS
        if (self.env.now + tamano_gate / R_tx + T_propagacion > self.t_inicio_tx[ont_id_prev] + self.T_alloc[ont_id_prev] + T_GUARDA):
            self.t_inicio_tx[ont_id] = self.env.now + tamano_gate / R_tx + T_propagacion 
        else:
            self.t_inicio_tx[ont_id] = self.t_inicio_tx[ont_id_prev] + self.T_alloc[ont_id_prev] + T_GUARDA

        # Guardar los valores en un archivo CSV
        carpeta_resultados = 'RND_resultados_csv_L05_Bmax_V2'
        if not os.path.exists(carpeta_resultados):
            os.makedirs(carpeta_resultados)

        nombre_archivo = f'valores_{time.strftime("%Y%m%d_%H%M%S")}.csv'
        ruta_completa = os.path.join(carpeta_resultados, nombre_archivo)

        with open(ruta_completa, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(['Onu_id', 'B_demand_bits', 'B_demand_MBS', 'B_max_bits', 'B_max_MBS','B_alloc_bits', 'B_alloc_MBS', 'B_guaranteed', 'error_max_aloc', 'error_demand_alloc','error_gted_alloc_acum', 'error_max_demand'])
            writer.writerow([ont_id, self.B_demand[ont_id], ((self.B_demand[ont_id] * (10 ** -6)) / (T_AVAILABLE)) / 8, self.B_max[ont_id], ((self.B_max[ont_id] * (10 ** -6)) / (T_AVAILABLE)) / 8, self.B_alloc[ont_id], self.B_alloc[ont_id],((self.B_alloc[ont_id] * (10 ** -6)) / (T_AVAILABLE)) / 8, self.B_guaranteed[ont_id], (self.B_max[ont_id] - self.B_alloc[ont_id]), (self.B_demand[ont_id] - self.B_alloc[ont_id]), (self.B_max[ont_id] - self.B_demand[ont_id])])                         
        return ont_id

    def enviar_gate(self, env, ont_id):
        if watch_on == True:
            print(MAGENTA + f"(t={(self.env.now):,.12f}ns) OLT -> ONT {ont_id}: gate | t_init = {self.t_inicio_tx[ont_id]:,.12f} ns | B_alloc = {self.B_alloc[ont_id] / 8:,.0f}  Bytes | T_alloc = {self.B_alloc[ont_id] / R_tx:,.12f} s" + RESET)

        trama_enviada = MensajeGate(self.contador_gates, ont_id, 'L', self.env.now, self.t_inicio_tx[ont_id], self.B_alloc[ont_id])
        self.contador_gates += 1
        yield env.timeout(trama_enviada.len / R_tx)
        self.splitter_out.enviar(trama_enviada)
    
    def extraer_retardo(self, env, trama):
        timestamp_creacion = trama.timestamp
        id_ont = trama.mac_src
        timestamp_llegada = self.env.now
        retardo = timestamp_llegada - timestamp_creacion

        if retardo == 0:
            print(f"retardo 0!")

        self.retardos_estadisticas[id_ont][trama.prioridad].actualizar(retardo)
        
    def enviar_gate_inicial(self, env):
        for ont_id in range(N_ONTS):
            if ont_id == 0:
                self.t_inicio_tx[0] = self.env.now + tamano_gate / R_tx + T_propagacion
            else:
                self.t_inicio_tx[ont_id] = self.t_inicio_tx[ont_id - 1] + B_inicial / R_tx + T_GUARDA

            trama_enviada = MensajeGate(self.contador_gates, ont_id, 'L', self.env.now, self.t_inicio_tx[ont_id], self.B_alloc[ont_id])
            self.contador_gates += 1

            if watch_on == True:
                print(MAGENTA + f"(t={(self.env.now):,.12f}ns) OLT -> ONT {ont_id}: gate | t_init = {self.t_inicio_tx[ont_id]:,.12f} ns | B_alloc = {self.B_alloc[ont_id] / 8:,.0f}  Bytes | T_alloc = {self.B_alloc[ont_id] / R_tx:,.12f} s" + RESET)

            yield env.timeout(trama_enviada.len / R_tx)
            self.splitter_out.enviar(trama_enviada)

    def escucha_splitter(self, env):
        while True:
            trama_recibida = yield self.splitter_in.get()

            if mostrar_progreso == True:
                progreso = 100 * self.env.now / T_SIM
                print(f"Progreso : {(progreso):.2f}% | t = {self.env.now * 1e9:,.3f} ", end='\r', flush=True)

            if isinstance(trama_recibida, MensajeReport):
                ont_id = self.procesa_report(env, trama_recibida)
                self.env.process(self.enviar_gate(env, ont_id))
            elif isinstance(trama_recibida, TramaEthernet):
                self.contador_paquetes_recibidos_olt += 1
                self.contador_Bytes_recibidos_olt += trama_recibida.len

                self.extraer_retardo(env, trama_recibida)


