from packages.configuration.parameters import *
from packages.classes.MensajeGate import MensajeGate
from packages.classes.MensajeReport import MensajeReport
from packages.classes.TramaEthernet import TramaEthernet
from packages.classes.EstadisticasWelford import EstadisticasWelford

import csv
import numpy as np
import os
import time 


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

        self.B_alloc_Mbps=[0.0]*N_ONTS # Vector que representa el sumatorio de todos los B_alloc asignados a cada ONT en bits
        self.B_guaranteed=[0.0]*N_ONTS # Vector que guarda el SLA que tendrá cada ONU en Mbps
        self.B_max_sum=[]
        self.B_max_sum2=0.0
        self.B_Disp=0
    
        self.n_alloc = [] # vector que representa el número de veces que hemos asignado una ventana a cada ONT
        self.T_alloc = [] # vector que recoge los tiempos de transmisión asignados a cada ONT
        self.t_inicio_tx = [] # vector que representa el tiempo de inicio de transmisión de cada ONT
        self.colas_tamanos = [] # Registro del tamaño de cada cola en bits
        self.w_sla = []      # Ponderaciones de cada SLA
        self.B_max = []    # BW máximo asignado a cada onu, en número de tramas enviables
        
        self.contador_paquetes_recibidos_olt = 0 # cuenta el total de paquetes llegados a la olt
        self.contador_Bytes_recibidos_olt = 0 # cuenta el total de Bytes llegados a la olt (payload+cabeceras, sin contar reports)
        self.contador_gates = 0 # cuenta el total de gates enviados por la olt
        self.retardos_estadisticas = [] # vector que recoge las estadísticas de retardo de cada ONU
        self.action = env.process(self.escucha_splitter(self.env))
        self.action = env.process(self.enviar_gate_inicial(self.env))
        self.i_ball=0

        for i in range(N_ONTS):
            # Iniicalizamos variables
            self.colas_tamanos.append([])
            self.retardos_estadisticas.append([])
            #self.w_sla.append(1)        # Suponemos para toda ONUs tenemos un SLA_0 donde w=1
            #self.w_sla=[300e6,300e6,300e6] # Vector con todos los diferentes SLAs en Mbps
            self.B_max=([])
            self.B_max_sum=([])
            #self.B_max_sum2=(0)
            self.B_demand.append(0)
            self.B_alloc.append(B_inicial)
            self.B_alloc_acum.append(B_inicial)
            
            self.n_alloc.append(1)
            self.T_alloc.append(0)
            self.t_inicio_tx.append(0)
           

    

        for i in range(N_ONTS):
            # Inicializamos las colas
            for j in range(N_COLAS):
                self.colas_tamanos[i].append(0)
                self.retardos_estadisticas[i].append(EstadisticasWelford())
            # Asignar ancho de banda garantizado según el grupo de ONTs
            if i<=1:
                self.B_guaranteed[i]=700000000 # 700 Mbps
            if 2<=i<=5:
                self.B_guaranteed[i]=500000000 # 500 Mbps
            if 6<=i<=15:
                self.B_guaranteed[i]=300000000 # 300 Mbps
        
        B_AVAILABLE = T_AVAILABLE*R_tx
        self.B_Disp=B_AVAILABLE
        # watch
        if watch_on==True:
            print(PURPLE + f" B_AVAILABLE = {B_AVAILABLE/8:,.0f} Bytes ({B_AVAILABLE/R_tx} s)" + RESET)
        for i in range(N_ONTS):
            # Calculamos el BW máximo para cada ONU y para cualquier ciclo
            #self.B_max.append(B_AVAILABLE*self.w_sla[i]/sum(self.w_sla))

            print (f"{self.B_guaranteed[i]/1e6} Mbps\n") # Vemos que el B garantizado es el que queremos para cada ONU dependiendo SLA
            #self.B_max.append((B_AVAILABLE*self.B_guaranteed[self.w_sla[i]])/sum(self.B_guaranteed)) # Cálculo del B máximo en caso de tener diferentes SLAs
            
            #self.B_max.append((B_AVAILABLE*self.w_sla[i])/sum(self.w_sla)) # Cálculo del B máximo en caso de tener diferentes SLAs
            self.B_max.append(T_AVAILABLE*self.B_guaranteed[i])
            

        
            # watch
            if watch_on==True:
                print(PURPLE + f"B_max (ont {i}, sla = {self.w_sla[i]}) = {self.B_max[i]/8:,.0f} Bytes ({self.B_max[i]/R_tx} s)" + RESET)
        print(f"B_Max después de asignarlo {[f'{x:,.3f}' for x in self.B_max]}\n") 
        print(f"B_Available después de asignarlo {B_AVAILABLE}\n") 
        
        # watch
        if watch_on==True:
            print(PURPLE + f"B_max (total ONTs) = {sum(self.B_max)/8:,.0f} Bytes ({sum(self.B_max)/R_tx} s)" + RESET)

        self.B_max_sum=sum(self.B_max)
        
    def procesa_report(self, env, mensaje_report):
        ## Método que actualiza el ancho de banda y los tiempos de inicio asignados a las ONTs

        # Actualización de tiempos de transmisión para cada ONU
        # Actualizamos el registro en la OLT en el que se registra la cola de cada ONU
        colas_tamanos = mensaje_report.colas_tamanos
        ont_id = mensaje_report.mac_src
        for i in range(N_COLAS):
            self.colas_tamanos[ont_id][i] = colas_tamanos[i]

        # Actualizamos la demanda de cada ONU
        self.B_demand[ont_id] = sum(self.colas_tamanos[ont_id])

        # watch
        if watch_on==True:
            print(PURPLE + f"(t={(self.env.now):,.12f}ns) OLT <- ONT {ont_id} demanda B_demand = {self.B_demand[ont_id]/8:,.0f} Bytes", end = "")
        

        # Si la suma total del tamaño de las colas es 0, damos un valor predeterminado de 154
        if self.B_demand[ont_id] == 0:
            self.B_demand[ont_id] = B_inicial
        
        # Actualizamos el ancho de banda que a cada onu se le permite transmitir, segun IPACT
        #print(f"\nB_demand[{ont_id}]= {self.B_demand[ont_id]:,.3f} bits\n")
        #print(f"\nB_max[{ont_id}]= {self.B_max[ont_id]:,.3f} bits\n")
        #self.B_alloc[ont_id] = min(self.B_demand[ont_id], self.B_max[ont_id]) + tamano_report
        self.B_alloc[ont_id] = min(self.B_demand[ont_id], self.B_max[ont_id]) + tamano_report
        
        #print(f"\nB_alloc[{ont_id}]= {self.B_alloc[ont_id]-tamano_report:,.3f} bits\n")

        self.B_alloc_acum[ont_id] += self.B_alloc[ont_id] - tamano_report
        self.n_alloc[ont_id] += 1
        
        self.B_alloc_Mbps[ont_id] = self.B_alloc_Mbps[ont_id]+(self.B_alloc[ont_id] - tamano_report) # Sumo cada vez para ir viendo cuantos bits vamos acumulando de BW
        
        #print(f"B_alloc_acum[{ont_id}]={self.B_alloc_acum[ont_id]:,.3f} bits\n")
        self.B_max_sum2+=self.B_max[ont_id]
        #print(f"\nB_alloc_Mbps[{ont_id}]= {self.B_alloc_Mbps[ont_id]:,.3f} bits\n")
        
        #print(f"ONU {ont_id}: B_demand={self.B_demand[ont_id]}, B_alloc={self.B_alloc[ont_id]}, B_max={self.B_max[ont_id]}, SLA={self.w_sla[ont_id]}")
       

        # Actualizamos el tiempo de transmisión que a cada onu se le permite transmitir, según IPACT
        self.T_alloc[ont_id] = self.B_alloc[ont_id]/R_tx

        # Actualización de tiempos de inicio para cada ONU
        ont_id_prev = (ont_id - 1) % N_ONTS
        # ¿Lo siguiente se puede sustituir por un max{A,B}?
        # Caso A: La ONT no tiene que esperar a que terminen de transmitir ONTs previas
        if(self.env.now + tamano_gate/R_tx + T_propagacion > self.t_inicio_tx[ont_id_prev] + self.T_alloc[ont_id_prev] + T_GUARDA):
            self.t_inicio_tx[ont_id] = self.env.now + tamano_gate/R_tx + T_propagacion 
            caso='A'
        # Caso B: La ONT tiene que esperar a que terminen de transmitir ONTs previas
        if(self.env.now + tamano_gate/R_tx + T_propagacion <= self.t_inicio_tx[ont_id_prev] + self.T_alloc[ont_id_prev] + T_GUARDA):
            self.t_inicio_tx[ont_id] = self.t_inicio_tx[ont_id_prev] + self.T_alloc[ont_id_prev] + T_GUARDA
            caso='B'
        
        # watch
        if watch_on==True:
            print(f" | B_max = {self.B_max[ont_id]/8:,.0f} Bytes + 64 Bytes (report) | B_alloc = {self.B_alloc[ont_id]/8:,.0f} Bytes | T_alloc = {self.B_alloc[ont_id]/R_tx:,.12f} s | t_init = {self.t_inicio_tx[ont_id]:,.12f} s | caso = {caso}" + RESET)

        # Devolvemos el id de la ont
        return ont_id

    def enviar_gate(self, env, ont_id):
        # Función que envía un mensaje gate a la ONT ont_id

        # watch
        if watch_on==True:
            print(MAGENTA+f"(t={(self.env.now):,.12f}ns) OLT -> ONT {ont_id}: gate | t_init = {self.t_inicio_tx[ont_id]:,.12f} ns | B_alloc = {self.B_alloc[ont_id]/8:,.0f}  Bytes | T_alloc = {self.B_alloc[ont_id]/R_tx:,.12f} s"+RESET)

        # Encapsulamos trama de gate
        trama_enviada = MensajeGate(self.contador_gates, ont_id, 'L', self.env.now, self.t_inicio_tx[ont_id], self.B_alloc[ont_id])
        self.contador_gates += 1

        # Retardo de transmisión
        yield env.timeout(trama_enviada.len/R_tx)

        # Enviamos la trama
        self.splitter_out.enviar(trama_enviada)
    
    def extraer_retardo(self, env, trama):
        # Método que actualiza registro del retardo

        # Extraemos timestamp de creacion
        timestamp_creacion = trama.timestamp

        # Averiguamos de qué ONT proviene la trama
        id_ont = trama.mac_src
        timestamp_llegada = self.env.now

        # Actualizamos la tabla de retardos
        retardo = timestamp_llegada - timestamp_creacion
        # watch
        if(retardo==0):
            print(f"retardo 0!")

        # Actualizamos las estadísticas de retardo
        self.retardos_estadisticas[id_ont][trama.prioridad].actualizar(retardo)

    def enviar_gate_inicial(self, env):
        # Al inicio de la simulación enviamos mensajes gate para que las ONTs comiencen a transmitir
       
        for ont_id in range(N_ONTS):
            #Para cada ONT:
            # Ajustamos tiempo de inicio de transmisión
            if(ont_id==0):
                 self.t_inicio_tx[0] = self.env.now + tamano_gate/R_tx + T_propagacion
            else:
                self.t_inicio_tx[ont_id] = self.t_inicio_tx[ont_id-1] + B_inicial/R_tx + T_GUARDA

            # Encapsulamos trama de gate
            trama_enviada = MensajeGate(self.contador_gates, ont_id, 'L', self.env.now, self.t_inicio_tx[ont_id], self.B_alloc[ont_id])
            self.contador_gates += 1

            # watch
            if watch_on==True:
                print(MAGENTA+f"(t={(self.env.now):,.12f}ns) OLT -> ONT {ont_id}: gate | t_init = {self.t_inicio_tx[ont_id]:,.12f} ns | B_alloc = {self.B_alloc[ont_id]/8:,.0f}  Bytes | T_alloc = {self.B_alloc[ont_id]/R_tx:,.12f} s"+RESET)


            # Retardo de transmisión
            yield env.timeout(trama_enviada.len/R_tx)

            # Enviamos la trama
            self.splitter_out.enviar(trama_enviada)

    def escucha_splitter(self, env):
        # Método que escucha de forma continua el splitter en sentido Upstream (splitter_in)
        while True:
            trama_recibida = yield self.splitter_in.get() # Atrapamos el mensaje entrante con get

            # Mostramos por pantalla un indicador del progreso
            if mostrar_progreso==True:
                progreso = 100*self.env.now/T_SIM
                print(f"Progreso : {(progreso):.2f}% | t = {self.env.now*1e9:,.3f} ", end = '\r', flush=True)



            if(isinstance(trama_recibida, MensajeReport)):
                # Si el mensaje es un report, lo procesamos
                # Primero actualizamos el registro en la OLT en el que se guarda la cola de cada ONU
                ont_id = self.procesa_report(env, trama_recibida)
                # Enviamos el mensaje gate
                self.env.process(self.enviar_gate(env, ont_id))
            elif(isinstance(trama_recibida, TramaEthernet)):
                # Si la trama recibida no está vacía, se trata de una trama de datos
                self.contador_paquetes_recibidos_olt += 1
                self.contador_Bytes_recibidos_olt += trama_recibida.len/8
                self.extraer_retardo(env, trama_recibida)
