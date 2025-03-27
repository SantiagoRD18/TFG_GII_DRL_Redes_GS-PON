from packages.configuration.parameters import *

## Trama Ethernet
class TramaEthernet:
    # Estructura de trama Ethernet 802.3
    # +--------------------------------+-----------------------+------------+
    # | Campo                          | Número de octetos     | bits       |
    # +--------------------------------+-----------------------+------------+
    # | Preambulo                      | 3 octetos             | :24        -
    # | Id paquete                     | 4 octetos             | 24:56      -
    # | Delimitador de inicio de trama | 1 octeto              | 56:64      -
    # | Dirección de destino (MAC)     | 6 octetos             | 64:112     -
    # | Dirección de origen (MAC)      | 6 octetos             | 112:160    -
    # | Longitud/Tipo                  | 2 octetos             | 160:176    -
    # | Datos                          | Variable(46-1500 oct) | 176:-32    -
    # | Timestamp                      | 4 octetos             | -32:       -
    # +--------------------------------+-----------------------+------------+
    def __init__(self, id_paq, mac_dst, mac_src, len, timestamp, prioridad=0, payload=None):
        self.id_paq = id_paq # id del paquete
        self.mac_dst = mac_dst # dirección MAC de destino
        self.mac_src = mac_src # dirección MAC de la fuente
        self.len = len # longitud del payload en bits
        self.timestamp = timestamp # timestamp de creación del mensaje
        self.prioridad = prioridad # prioridad del mensaje
        self.payload = payload # payload del mensaje
