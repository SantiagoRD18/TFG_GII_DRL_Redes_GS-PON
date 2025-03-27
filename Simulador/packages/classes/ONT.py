from packages.configuration.parameters import *
from packages.classes.MensajeReport import MensajeReport
from packages.classes.TramaEthernet import TramaEthernet
from bitarray import bitarray
from bitarray import util

## ONT
class ONT:
    # Simula una ONT
    def __init__(self, env, id, generador_trafico, splitter_in, splitter_out):
        # generador_trafico es una instancia de la clase GeneraTrafico, que modela la capa de de aplicación, la capa de transporte y las colas presentes en una ONT. Va generando datos, que se van segmentando para crear paquetes que se van almacenando en colas. Dichas colas se guardan en generador_trafico
        self.env = env
        self.id = id
        self.generador_trafico = generador_trafico
        self.splitter_in = splitter_in # enlace que representa el Splitter en sentido Downstream
        self.splitter_out = splitter_out # enlace que representa el Splitter en sentido Upstream
        self.seq = [0,0,0]  # Número de secuencia de payload enviado
        self.reports_enviados = 0 # Número de reports enviados
        self.action = env.process(self.escucha_splitter(self.env))

    def transmite_respuesta(self, env, respuesta):

        if isinstance(respuesta, MensajeReport):
            # Retardo de transmisión
            yield env.timeout(respuesta.len/R_tx)
        elif isinstance(respuesta, TramaEthernet):
            yield env.timeout(respuesta.len/R_tx)

        # Enviamos la trama
        self.splitter_out.enviar(respuesta)
    
    def extrae_paquete_prioridad_estricta(self):
        # Devuelve un paquete extraído de las colas siguiendo el protocolo de Prioridad Estricta
        paquete = TramaEthernet(0, 'L', self.id, 0, self.env.now)

        for i in range(N_COLAS):
            # Va recorriendo las colas de la más prioritaria (i=0) a la menos prioritaria (i=N_COLAS-1)
            # print(f"# Examinamos la cola {self.id}.{i}")
            if len(self.generador_trafico.colas[i]) != 0:
                # print(f"# !! La cola {i} no está vacía")
                # Si la cola no esta vacía, el método devuelve el paquete a enviar,
                # actualiza el número de secuencia y elimina del paquete de la cola

                # Guardamos en 'paquete' el primer paquete de la cola (FIFO)
                paquete = self.generador_trafico.colas[i][0]

                # Reducimos el tamaño de la cola
                self.generador_trafico.colas_longitudes[i] -= paquete.len

                # Liberamos el primer paquete de la cola
                self.generador_trafico.colas[i].pop(0)

                # Actualizamos el número de secuencia
                self.seq[i] += int(paquete.len/8)

                break

        return paquete

    def envia_report(self, env):
        # Procesa y envía el mensaje de report

        # Creamos report bitmap
        report_bitmap=bitarray('10000000')

        # Anotamos las longitudes de las tres colas, el resto es cero
        colas_tamanos=[self.generador_trafico.colas_longitudes[0],0,0]

        if(multiples_colas):
            report_bitmap=bitarray('11100000')
            # Si tenemos más de una cola, guardamos los valores
            colas_tamanos[1]=self.generador_trafico.colas_longitudes[1]
            colas_tamanos[2]=self.generador_trafico.colas_longitudes[2]

        # Generamos el mensaje report y lo enviamos
        mensaje_report = MensajeReport(self.reports_enviados, 'L', self.id, self.env.now, report_bitmap, colas_tamanos)
        self.reports_enviados += 1

        # watch
        if watch_on==True:
            print(BLUE + f"(t={(self.env.now):,.12f}ns) ONT {self.id} -> OLT: envía report, cola = {colas_tamanos[0]/8:,.0f}  Bytes (inicio)" + RESET)

        # Enviamos el paquete
        yield env.process(self.transmite_respuesta(env, mensaje_report))

        # watch
        if watch_on==True:
            print(BLUE + f"(t={(self.env.now):,.12f}ns) ONT {self.id} -> OLT: ha enviado report, cola = {colas_tamanos[0]/8:,.0f} Bytes (fin)" + RESET)

    def procesa_respuesta(self, env, msg):
        # Procesa el mensaje GATE recibido
        if msg.mac_dst==self.id:
            # Si el mensaje GATE se dirige a la ONT, lo procesa

            # Extraemos los grants obtenidos
            B_alloc = msg.grants_lengths

            # Extraemos los tiempos de espera
            t_initio_tx = msg.grants_start_times

            # Calculamos el tiempo de espera
            t_espera = t_initio_tx - self.env.now

            # watch
            if watch_on==True:
                try:
                    print(GREEN + f"(t={(self.env.now):,.12f}ns) ONT {self.id} recibe un GATE: | B_alloc = {B_alloc/8:,.0f} Bytes | T_alloc = {B_alloc/R_tx:,.12f} s | t_espera = {t_espera:,.12f} s" + RESET)
                except UnboundLocalError as e:
                    print(f"Exception occured: {e}")
                    
            # Si el tiempo de espera es positivo, esperamos
            if t_espera > 0:
                yield env.timeout(t_espera)
                
            # Enviamos datos hasta que se agote la ventana
            B_enviado = 0
            # watch
            if watch_on==True:
                print(YELLOW + f"(t={(self.env.now):,.12f}ns) ONT {self.id} -> OLT: comienza a enviar hasta {B_alloc/8:,.0f} Bytes (inicio)" + RESET)
            while True:
                # Durante toda la ventana concedida, vamos extrayendo paquetes de la cola
                paquete_respuesta = self.extrae_paquete_prioridad_estricta()

                # Enviamos el paquete extraído
                yield env.process(self.transmite_respuesta(env, paquete_respuesta))

                # Actualizamos el indicador de datos enviados
                B_enviado += paquete_respuesta.len
                # Aquí se puede añadir una función que mire cuánto tiene el próximo paquete?
                if B_enviado + (max(tamano_payload) + tamano_cabecera + tamano_report) >= B_alloc: 
                    # Si ya hemos enviado toda la ventana (a falta del report más un paquete, para no excedernos), salimos del bucle
                    break
            # watch
            if watch_on==True:
                print(YELLOW + f"(t={(self.env.now):,.12f}ns) ONT {self.id} -> OLT: ha enviado {(B_enviado)/8:,.0f} Bytes (fin). Le falta enviar el report (64 Bytes)" + RESET)

            # Enviamos mensaje report
            env.process(self.envia_report(env))
            # Esperamos tiempo de guarda
            yield env.timeout(T_GUARDA)

    def escucha_splitter(self, env):
        # Cuando se recibe una trama, se procesa con la función procesa_respuesta
        while True:
            msg = yield self.splitter_in.get()
            env.process(self.procesa_respuesta(env, msg))