from packages.configuration.parameters import *

## Mensaje Gate        
class MensajeGate:
    # Clase que representa un mensaje gate
    # +---------------------------------+--------------------+------------+
    # | Campo                           | Número de octetos  | Selector   |
    # +---------------------------------+--------------------+------------+
    # | Dirección destino               | 6                  |[1:48]      |
    # | Dirección origen                | 6                  |[48:96]     |
    # | Longitud/Tipo = 88-08           | 2                  |[96:112]    |
    # | Código de operación = 00-02     | 2                  |[112:128]   |
    # | Tiempo de creación (timestamp)  | 4                  |[128:160]   |
    # | Number of grants/flags          | 1                  |[160:168]   |
    # | Grant #1 Start time             | 0/5                |            |
    # | Grant #1 Length                 | 0/4                |            |
    # | Grant #2 Start time             | 0/5                |            |
    # | Grant #2 Length                 | 0/4                |            |
    # | Grant #3 Start time             | 0/5                |            |
    # | Grant #3 Length                 | 0/4                |            |
    # | Pad / reserved                  | 12                 |            |
    # | FCS                             | 4                  |[-4:]       |
    # +---------------------------------+--------------------+------------+

    # Longitud de la trama:
    # Bytes:    64 Bytes
    # bits:     512 bits
    def __init__(self, id, mac_dst, mac_src, timestamp, grants_start_times, grants_lengths):
        self.id = id    # id del mensaje
        self.len = tamano_gate  # tamaño en bits del mensaje
        self.mac_dst = mac_dst  # dirección MAC de destino
        self.mac_src = mac_src  # dirección MAC de la fuente
        self.timestamp = timestamp  # timestamp de creación del mensaje
        self.grants_start_times = grants_start_times    # tiempos de inicio del mensaje
        self.grants_lengths = grants_lengths # longitud de los grants
