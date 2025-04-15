from packages.configuration.configuration import *

c = 299792458*2/3
L_RED = 20e3
T_propagacion = L_RED/c
RTT = T_propagacion*2

R_datos = 1.6e9  # Bitrate de datos en la capa de aplicación (bps)
R_tx = 1e9       # Bitrate de transmisión (bps)
N_ONTS = 16
tamano_cabecera = (22+4)*8  # Tamaño de la cabecera total (208 bits)
tamano_cabecera_1 = 22*8    # Tamaño de la cabecera al principio del paquete
tamano_cabecera_2 = 4*8     # Tamaño de la cabecera al final del paquete
tamano_payload  = [64*8, 594*8, 1500*8]   # Tamaño paquete
tamano_report = 64*8
tamano_gate = 64*8
T_REPORT = tamano_gate/R_tx
T_CICLO = 2e-3
T_GUARDA = 5e-6
T_AVAILABLE  = T_CICLO - N_ONTS*(T_GUARDA + T_REPORT)   # Tiempo disponible para enviar datos en cada ciclo
T_TRAMA = (160+tamano_payload[2]+tamano_cabecera)/R_tx + 2*T_propagacion
B_inicial = tamano_cabecera+tamano_payload[0] + tamano_report            # Bytes iniciales en la ONT
L_BUFFER_ONTS = 10e6*8

T_SIM = CONFIG_T_SIM

# Configuramos las fuentes de tráfico de pareto
if(multiples_colas):
    N_COLAS = 3   # Número de colas en la ONT
else:
    N_COLAS = 1
N_SOURCES = [3, 5, 24]   # Número de Sources de las fuentes de Pareto de cada tipo (hacer ésto con un diccionario)        
a = 1.2
media = .8e-4
m = media * (a-1)/a
m_on = m
a_on = 1.4
a_off = 1.2

# Caracteres ANSI para colorear texto
YELLOW = "\033[0;33m"
MAGENTA = "\u001b[31m"
CYAN = "\u001b[34m"
WHITE = "\033[0;37m"
RESET = "\033[0m"   # Reset al color por defecto del texto
GREEN = "\033[0;32m"
PURPLE = "\033[0;35m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"
BROWN = "\033[0;33m"