from simulador.packages.configuration.configuration import *

### Variables globales
c = 299792458*2/3
L_RED = 20e3
T_propagacion = L_RED/c
# T_propagacion = 1/10e9*3/2
RTT = T_propagacion*2

R_datos = 1.6e10      # Bitrate de datos en la capa de aplicación (bps)
R_tx = 1e10        # Bitrate de transmisión (bps)
N_ONTS = 16          # Número de ONTs
# N_ONTS = 3                 # Número de ONTs
tamano_cabecera = (22+4)*8  # 208 bits, tamaño de la cabecera total
tamano_cabecera_1 = 22*8    # Tamaño de la cabecera al principio del paquete
tamano_cabecera_2 = 4*8     # Tamaño de la cabecera al final del paquete
tamano_payload  = [64*8, 594*8, 1500*8]    # tamanos paquete
tamano_report = 64*8        # Tamaño de la trama de report
tamano_gate = 64*8          # Tamaño de la trama de gate
T_REPORT = tamano_gate/R_tx # Tiempo de transmisión de un report
T_CICLO = 2e-3              # Periodo de ciclo
T_GUARDA = 5e-6             # Tiempo de guarda
T_AVAILABLE  = T_CICLO - N_ONTS*(T_GUARDA + T_REPORT)   # Tiempo disponible para enviar datos en cada ciclo
T_TRAMA = (160+tamano_payload[2]+tamano_cabecera)/R_tx + 2*T_propagacion
B_inicial = tamano_cabecera+tamano_payload[0] + tamano_report            # Bytes iniciales en la ONT
L_BUFFER_ONTS = 100e6*8

## Configuramos las fuentes de tráfico
if(multiples_colas):
    N_COLAS = 3                 # Número de colas en la ONT
else:
    N_COLAS = 1
N_SOURCES = [3, 5, 24]      # Número de Sources de las fuentes de Pareto de cada tipo (hacer ésto con un diccionario)        
a = 1.2
#carga = CONFIG_CARGA
# media = .5e-4
media = .8e-4
# Calculamos m
m = media * (a-1)/a
# Para los periodos de ON.
m_on = m
a_on = a
# Para los periodos de OFF
# m_off = m_on*(1-carga)/carga
a_off = a

# T_SIM = 5e3*T_TRAMA     # Tiempo de simulación (s)
# T_SIM = 100*T_TRAMA
T_SIM = CONFIG_T_SIM
# T_SIM = 4.294967296
# T_SIM = 4.294967296*24


### Caracteres ANSI para colorear texto
YELLOW = "\033[0;33m"
MAGENTA = "\u001b[31m"
CYAN = "\u001b[34m"
WHITE = "\033[0;37m"
RESET = "\033[0m"  # Reset al color por defecto del texto
GREEN = "\033[0;32m"  # Green
PURPLE = "\033[0;35m"  # Purple
RED = "\033[0;31m"  # Red
BLUE = "\033[0;34m"  # Blue
BROWN = "\033[0;33m"  # Brown (sometimes appears as yellow)