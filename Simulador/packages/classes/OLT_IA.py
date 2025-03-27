from packages.configuration.parameters import *
from packages.classes.MensajeGate import MensajeGate
from packages.classes.MensajeReport import MensajeReport
from packages.classes.TramaEthernet import TramaEthernet
from packages.classes.EstadisticasWelford import EstadisticasWelford

 
import csv
import time
import os
import random 
import torch
import joblib
import numpy as np
import warnings
from sklearn.exceptions import DataConversionWarning



# Cargar el modelo entrenado
model = joblib.load("mejor_modelo_dnn_v2.pkl")

# Definir el MinMaxScaler y cargarlo desde el archivo
scaler = joblib.load("scaler_v2.pkl")
'''
def estimar_valor(demand_bits, onu_id, carga):
    # Normalizar los datos de entrada
    entrada_normalizada = scaler.transform([[demand_bits, onu_id, carga]])
    # Realizar la estimación utilizando el modelo
    estimacion = model.predict(entrada_normalizada)
    return estimacion[0]
'''
def estimar_valor(demand_bits, onu_id, carga):
    import warnings

    from sklearn.exceptions import DataConversionWarning

    # Desactivar todas las advertencias relacionadas con la conversión de datos
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    try:
        # Normalizar los datos de entrada
        entrada_normalizada = scaler.transform([[demand_bits, onu_id, carga]])
    except DataConversionWarning:
        pass  # Ignorar las advertencias relacionadas con la conversión de datos

    # Realizar la estimación utilizando el modelo
    estimacion = model.predict(entrada_normalizada)

    # Restaurar el comportamiento normal de las advertencias
    warnings.filterwarnings(action='default', category=DataConversionWarning)

    return estimacion[0]



## OLT
class OLT:

    
    # Simula la OLT
    def __init__(self, env, splitter_in, splitter_out):
    
        self.env = env
        #self.start_time1 = start_time1
        
        self.splitter_in = splitter_in # enlace que representa el Splitter en sentido Upstream
        self.splitter_out = splitter_out # enlace que representa el Splitter en sentido Downstream
        self.B_demand = [] # vector que representa la ventana asignada a cada ONT en bits
        self.B_alloc = [] # vector que representa la ventana asignada a cada ONT en bits
        self.B_alloc_acum = [] # vector que representa la ventana acumulada asignada a cada ONT en bits
        self.B_alloc_acum_aux = [] # vector que representa la ventana acumulada asignada a cada ONT en bits
        self.B_alloc_Mbps=[] # Vector que representa el sumatorio de todos los B_alloc asignados a cada ONT en bits
        self.n_alloc = [] # vector que representa el número de veces que hemos asignado una ventana a cada ONT
        self.T_alloc = [] # vector que recoge los tiempos de transmisión asignados a cada ONT
        self.t_inicio_tx = [] # vector que representa el tiempo de inicio de transmisión de cada ONT
        self.colas_tamanos = [] # Registro del tamaño de cada cola en bits
        self.w_sla = []         # Ponderaciones de cada SLA, ESTO LO VAMOS A VARIAR, AL INICIO SOLO 1 SLA
        self.B_max = []    # BW máximo asignado a cada onu, en número de tramas enviables
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
        self.Carga = []
        
        #print(tiempo_inicio)

        for i in range(N_ONTS):
            # Iniicalizamos variables
            self.colas_tamanos.append([])
            self.retardos_estadisticas.append([])
            self.w_sla.append(1)        # Suponemos para toda ONUs tenemos un SLA_0 donde w=1
            self.B_demand.append(0)
            self.B_alloc.append(B_inicial)
            self.B_alloc_acum.append(B_inicial)
            self.B_alloc_acum_aux.append(B_inicial)
            self.B_alloc_Mbps.append(0)
            self.n_alloc.append(1)
            self.T_alloc.append(0)
            self.T_alloc_acum.append(0)
            self.T_alloc_acum_total.append(0)
            self.t_inicio_tx.append(0)
            

        for i in range(N_ONTS):
            # Inicializamos las colas
            for j in range(N_COLAS):
                self.colas_tamanos[i].append(0)
                self.retardos_estadisticas[i].append(EstadisticasWelford())

        B_AVAILABLE = T_AVAILABLE*R_tx
        
        
        
        
        ##----------------------------------------------------------------------------------------
        #
        #Aqui es donde se cambia el bmax al randomizer pue
        #
        ##----------------------------------------------------------------------------------------

        # watch
        if watch_on==True:
            print(PURPLE + f" B_AVAILABLE = {B_AVAILABLE/8:,.0f} Bytes ({B_AVAILABLE/R_tx} s)" + RESET)
        for i in range(N_ONTS):
            # Calculamos el BW máximo para cada ONU y para cualquier ciclo
            self.B_max.append(B_AVAILABLE*self.w_sla[i]/sum(self.w_sla)) ## Forma original de definir B_max por ont
            
            #self.B_max.append((B_AVAILABLE*self.w_sla[i])/sum(self.w_sla) #+(random.randrange(-8, 8) * 10000))
            # watch
            self.B_guaranteed.append(138000) # Cambiar a 600 mb con 10 gb
            
            
            
            
            ## ESTO....VAYAMOS ACUMULANDO EL TIEMPO TOTAL ASIGNADO(?)
            self.B_alloc_acum_MBS.append(self.B_alloc_acum)
            
            
            
            #print('B_guaranteed es:')
            #print(self.B_guaranteed)
            
            if watch_on==True:
                print(PURPLE + f"B_max (ont {i}, sla = {self.w_sla[i]}) = {self.B_max[i]/8:,.0f} Bytes ({self.B_max[i]/R_tx} s)" + RESET)

        # watch
        if watch_on==True:
            print(PURPLE + f"B_max (total ONTs) = {sum(self.B_max)/8:,.0f} Bytes ({sum(self.B_max)/R_tx} s)" + RESET) #B_max es el valor a "balancear"
            #Con esto revisar ecuacion 5 de la tesis. El peso se supone 1 al inicio.

    def procesa_report(self, env, mensaje_report):
        ## Método que actualiza el ancho de banda y los tiempos de inicio asignados a las ONTs
        tiempo_inicio = time.time()
        if not hasattr(self, 'B_alloc_acum_MBS'):
            self.B_alloc_acum_MBS = [0]*int(N_ONTS)  # N es el número total de ont_id, convertido a int

        # Actualización de tiempos de transmisión para cada ONU
        # Actualizamos el registro en la OLT en el que se registra la cola de cada ONU
        colas_tamanos = mensaje_report.colas_tamanos
        ont_id = mensaje_report.mac_src
        for i in range(N_COLAS):
            self.colas_tamanos[ont_id][i] = colas_tamanos[i]

        # Actualizamos la demanda de cada ONU
        self.B_demand[ont_id] = sum(self.colas_tamanos[ont_id]) #B_demand es el valor por pedir. Esto es lo que se puede predecir.
        self.Carga.append(0.8)
        
        # watch
        if watch_on==True:
            print(PURPLE + f"(t={(self.env.now):,.12f}ns) OLT <- ONT {ont_id} demanda B_demand = {self.B_demand[ont_id]/8:,.0f} Bytes", end = "")
        

        # Si la suma total del tamaño de las colas es 0, damos un valor predeterminado de 154
        if self.B_demand[ont_id] == 0: #Aqui está el vector estado que se debe verificar.
            self.B_demand[ont_id] = B_inicial #B_demand es requisito para B_Alloc. B_Alloc es lo que voy balancear 
         
         
        # Si la suma total del tamaño de las colas es 0, damos un valor predeterminado de 154
        if self.B_demand[ont_id] == 0:
            self.B_demand[ont_id] = B_inicial

#########################################################################################################################            
#
#
#
#########################################################################################################################

        # Extraer el valor de B_demand[ont_id] como un número solito
        B_demand_bits = self.B_demand[ont_id]

        # Definir Onu_id como el propio ont_id
        Onu_id = ont_id

        # Definir Carga como un valor fijo, como 0.7
        Carga = 0.8

        # Empaquetar las variables en un arreglo de NumPy
        input_data = np.array([[B_demand_bits, Onu_id,Carga]], dtype=np.float32)
        self.predicted_B_max = estimar_valor(B_demand_bits, Onu_id,Carga)
        
        #print(self.predicted_B_max)
        # Realizar la consulta a la función consultar_B_max
        #self.predicted_B_max = consultar_B_max('modelo_entrenado_DNN.onnx', input_data)

#########################################################################################################################            
#
#
#
#########################################################################################################################

        
                # Actualizamos el ancho de banda que a cada onu se le permite transmitir, segun IPACT
        #self.B_alloc[ont_id] = min(self.B_demand[ont_id], self.B_max[ont_id]) + tamano_report #B_max se asigna aqui. Aqui es donde se debe balancear
        
        self.B_alloc[ont_id] = min(self.B_demand[ont_id], self.predicted_B_max) + tamano_report #B_max se asigna aqui. Aqui es donde se debe balancear
        
        self.B_alloc_acum[ont_id] += self.B_alloc[ont_id] - tamano_report 
        
        self.B_alloc_acum_aux[ont_id] += self.B_alloc_acum[ont_id]*8*1e-6    
        
        self.n_alloc[ont_id] += 1
        
        
        
        # Actualizamos el tiempo de transmisión que a cada onu se le permite transmitir, según IPACT
        self.T_alloc[ont_id] = self.B_alloc[ont_id]/R_tx 
        self.T_alloc_acum[ont_id] += self.T_alloc[ont_id] 
        self.B_alloc_Mbps[ont_id]=self.B_alloc[ont_id]+self.B_alloc_Mbps[ont_id] # Actualizamos el vector de B_Alloc_Mbps para luego usarle cuando acabe la simulación
        self.T_alloc_acum_total[ont_id] += self.T_alloc[ont_id] + self.t_inicio_tx[ont_id] + T_GUARDA + T_propagacion
        
        #print(f"Este es el valor T total: {self.T_alloc_acum_total[ont_id]}")          
            
           
            
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
        #print(self.t_inicio_tx[ont_id])
        #print(self.T_alloc[ont_id])
        #print(self.t_inicio_tx[ont_id]-self.T_alloc[ont_id])
      
                


 ## Aqui es donde se hace extracción de datos
 
        
        # Creamos la carpeta donde se guardarán los archivos CSV si no existe
        carpeta_resultados = 'RND_resultados_csv_L05_Bmax_V2_0.8'
        if not os.path.exists(carpeta_resultados):
            os.makedirs(carpeta_resultados)

        # Generamos el nombre del archivo CSV con un marcador de tiempo
        nombre_archivo = f'valores_{time.strftime("%Y%m%d_%H%M%S")}.csv'

        # Combinamos la ruta de la carpeta con el nombre del archivo
        ruta_completa = os.path.join(carpeta_resultados, nombre_archivo)
        
        #self.B_alloc_MBS=((self.B_alloc[ont_id]*(10**-5))/(T_AVAILABLE))/8


        #print('T_AVAILABLE en S es:')
        #print(T_AVAILABLE)
         
        #print('B_alloc en bits es:')
       # print(self.B_alloc[ont_id])
        
        #print('B_alloc en mbs es:')
        #print(B_alloc_MBS)


        # Guardamos los valores de B_alloc y n_alloc en el archivo CSV
        with open(ruta_completa, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(['Carga','Onu_id', 'B_demand_bits', 'B_demand_MBS', 'B_max_bits', 'B_max_MBS','B_alloc_bits', 'B_alloc_MBS', 'B_guaranteed', 'error_max_aloc', 'error_demand_alloc','error_gted_alloc_acum', 'error_max_demand'])
                #writer.writerow(['Onu_id', 'B_demand_bits', 'B_demand_MBS', 'B_max_bits', 'B_max_MBS','B_alloc_bits', 'B_alloc_MBS', 'error_max_aloc', 'error_demand_alloc','B_alloc_acum','B_guaranteed','error_gted_alloc_acum','B_alloc_acum_MBS'])
                #writer.writerow(['Onu_id', 'B_alloc_acum','T_alloc_acum_ns','B_alloc_acum_MBS','B_max','B_max_MBS','B_demand','B_demand_MBS', 'error_alloc_guaranted','B_alloc_acum_AUX','BALLOCACUMMBS'])
            #writer.writerow([ont_id, self.B_demand[ont_id],((self.B_demand[ont_id]*(10**-6))/(T_AVAILABLE))/8 ,self.B_max[ont_id], ((self.B_max[ont_id]*(10**-6))/(T_AVAILABLE))/8,self.B_alloc[ont_id], ((self.B_alloc[ont_id]*(10**-4))/(T_AVAILABLE))/8, (self.B_max[ont_id]-self.B_alloc[ont_id])+512, (self.B_demand[ont_id]-self.B_alloc[ont_id])+512,self.B_alloc_acum[ont_id],self.B_guaranteed[ont_id],(self.B_alloc[ont_id]-self.B_guaranteed[ont_id]),self.B_alloc_acum_MBS[ont_id]]) #Poner [ont_id] al lado de las variables para dato individual
            #writer.writerow([ont_id, self.B_alloc_acum[ont_id], self.T_alloc_acum[ont_id],self.B_alloc_acum[ont_id]//(self.T_alloc_acum_total[ont_id]*10000),self.B_max[ont_id],self.B_max[ont_id]//(self.T_alloc[ont_id]*10200000),self.B_demand[ont_id],self.B_demand[ont_id]//(self.T_alloc[ont_id]*10200000),(self.B_guaranteed[ont_id]-self.B_alloc[ont_id]),self.B_alloc_acum_aux[ont_id],self.B_alloc_acum_aux[ont_id]//(self.T_alloc_acum_total[ont_id])]) #Poner [ont_id] al lado de las variables para dato individual
            writer.writerow([CONFIG_CARGA,ont_id, self.B_demand[ont_id], ((self.B_demand[ont_id]*(10**-6))/(T_AVAILABLE))/8, self.B_max[ont_id], ((self.B_max[ont_id]*(10**-6))/(T_AVAILABLE))/8, self.B_alloc[ont_id], self.B_alloc[ont_id],((self.B_alloc[ont_id]*(10**-6))/(T_AVAILABLE))/8, self.B_guaranteed[ont_id], (self.B_max[ont_id]-self.B_alloc[ont_id]),(self.B_demand[ont_id]-self.B_alloc[ont_id]),(self.B_max[ont_id]-self.B_demand[ont_id])])                         
            # self.B_alloc_acum[ont_id], self.T_alloc_acum[ont_id],self.B_alloc_acum[ont_id]//(self.T_alloc_acum_total[ont_id]*10000),self.B_max[ont_id],self.B_max[ont_id]//(self.T_alloc[ont_id]*10200000),self.B_demand[ont_id],self.B_demand[ont_id]//(self.T_alloc[ont_id]*10200000),(self.B_guaranteed[ont_id]-self.B_alloc[ont_id]),self.B_alloc_acum_aux[ont_id],self.B_alloc_acum_aux[ont_id]//(self.T_alloc_acum_total[ont_id])]) #Poner [ont_id] al lado de las variables para dato individual
        

## Aqui es donde termina 
        
        
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
