from packages.configuration.parameters import *

## Mensaje Report
class MensajeReport:
    # Clase que representa un mensaje report
    # +---------------------------------+--------------------+------------+
    # | Campo                           | Número de octetos  | Selector   |
    # +---------------------------------+--------------------+------------+
    # | Dirección destino               | 6                  |[1:48]      |
    # | Dirección origen                | 6                  |[48:96]     |
    # | Longitud/Tipo = 88-08           | 2                  |[96:112]    |
    # | Código de operación = 00-03     | 2                  |[112:128]   |
    # | Tiempo de creación (timestamp)  | 4                  |[128:160]   |
    # | Number of queue sets            | 1                  |[160:168]   |
    # | Report bitmap                   | 1                  |[168:176]   |
    # | Cola #0 Report                  | 0/4                |            |
    # | Cola #1 Report                  | 0/4                |            |
    # | Cola #2 Report                  | 0/4                |            |
    # | Pad/Reserved                    | --                 |            |
    # | FCS                             | 4                  |[-4:]       |
    # +---------------------------------+--------------------+------------+

    # Longitud de la trama:
    # Bytes:    64 Bytes
    # bits:     512 bits

    def __init__(self, id, mac_dst, mac_src, timestamp, report_bitmap, colas_tamanos):
        self.id = id    # id del mensaje
        self.len = tamano_report # tamano en bits del mensaje
        self.mac_dst = mac_dst  # dirección MAC de destino
        self.mac_src = mac_src  # dirección MAC de la fuente
        self.timestamp = timestamp  # timestamp de creación del mensaje
        self.report_bitmap = report_bitmap    # bitmap de report
        self.colas_tamanos = colas_tamanos    # tamaños de las colas
