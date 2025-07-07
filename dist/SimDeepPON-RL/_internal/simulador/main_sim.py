### Importamos módulos propios
from simulador.ejecutar_simulacion import ejecutar_simulacion
from simulador.calcular_estadisticas import calcular_estadisticas
from simulador.packages.configuration.configuration import *
import time
import os
import csv
import sys


## Simulación
def main(modo_ejecucion=0, nombre_modelo = None):
    for i in range(len(CONFIG_CARGA)):
        try:
            ejecutar_simulacion(CONFIG_CARGA[i], modo_ejecucion, nombre_modelo)
        except Exception as e:
            print(f"Ha ocurrido un error: {e}")


if __name__ == '__main__':
    modo_ejecucion = sys.argv[1] if len(sys.argv) > 1 else 0
    nombre_modelo = sys.argv[2] if len(sys.argv) > 2 else None
    main(modo_ejecucion, nombre_modelo)