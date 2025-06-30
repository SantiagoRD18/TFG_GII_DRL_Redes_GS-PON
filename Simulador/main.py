### Importamos módulos propios
from ejecutar_simulacion import ejecutar_simulacion
from calcular_estadisticas import calcular_estadisticas
from packages.configuration.configuration import *
import time
import os
import csv


## Simulación
def main():
    for i in range(len(CONFIG_CARGA)):
        # Ejecutamos la simulacion
        try:
            ejecutar_simulacion(CONFIG_CARGA[i])
        except Exception as e:
            # Atrapar cualquier excepción
            print(f"Ha ocurrido un error: {e}")


if __name__ == '__main__':
    main()

