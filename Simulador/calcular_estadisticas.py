import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t
from scipy.stats import norm
import os
import time
import csv

from packages.configuration.parameters import *

def calcular_estadisticas():
    ## Retardos
    # Lemos el fichero
    file_path = "data"
    file_name = ''
    for file in os.listdir(file_path):
        if file.startswith("retardos"):
            # Tenemos que escoger el fichero que contenga los datos de la carga
            file_name = file
            break
    if file_name == '':
        # Si no encontramos el fichero, lo anunciamos por pantalla
        print("No file found with the specified criteria.")
    else:
        # Abrimos el fichero
        file = os.path.join(file_path, file_name)
        # Guardamos los datos en un dataframe
        df = pd.read_csv(file)

        retardos_medios = []
        intervalos_confianza_t = []
        intervalos_confianza_n = []
        for ont_id, group in df.groupby("ont_id")["retardo"]:
            retardos_medios.append(np.mean(group))
            intervalos_confianza_t.append(t.interval(0.95, len(group) - 1, loc=retardos_medios[ont_id], scale=np.std(group) / np.sqrt(len(group))))
            intervalos_confianza_n.append(norm.interval(0.95, loc=retardos_medios[ont_id], scale=np.std(group) / np.sqrt(len(group))))

        
        print("Retardos medios = ")
        for i in range(N_ONTS):
            print(f"ONT {i}: ")
            print(f"\t Retardo medio : {retardos_medios[i]:.4e}")
            print(f"\t Intervalo de confianza (t): ({intervalos_confianza_t[i][0]:.4e}, {intervalos_confianza_t[i][1]:.4e})")
            print(f"\t Intervalo de confianza (n): ({intervalos_confianza_n[i][0]:.4e}, {intervalos_confianza_n[i][1]:.4e})")

        print()
        print("Media:")
        retardo_medio_total = np.mean(df["retardo"])
        intervalo_confianza_total_t = t.interval(0.95, len(df) - 1, loc=retardo_medio_total, scale=np.std(df["retardo"]) / np.sqrt(len(df)))
        intervalo_confianza_total_n = norm.interval(0.95, loc=retardo_medio_total, scale=np.std(df["retardo"]) / np.sqrt(len(df)))
        print(f"\t Retardo medio : {retardo_medio_total:.4e}")
        print(f"\t Intervalo de confianza (t): ({intervalo_confianza_total_t[0]:.4e}, {intervalo_confianza_total_t[1]:.4e})")
        print(f"\t Intervalo de confianza (n): ({intervalo_confianza_total_n[0]:.4e}, {intervalo_confianza_total_n[1]:.4e})")
        

        ## Volcamos los datos a un csv
        start_time_str = time.strftime("%Y%m%d-%H%M", time.gmtime())
        subdirectory= "data"
        csv_retardos_summary = open(os.path.join(subdirectory, f"retardos_summary_{start_time_str}.csv"), "w")
        csv_retardos_summary_writer = csv.writer(csv_retardos_summary, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csv_retardos_summary_writer.writerow(["ont", "retardo_medio", "intervalo_confianza_t_left", "intervalo_confianza_t_right", "intervalo_confianza_n_left", "intervalo_confianza_n_right"])

        for i in range(N_ONTS):
            csv_retardos_summary_writer.writerow([i, retardos_medios[i], intervalos_confianza_t[i][0], intervalos_confianza_t[i][1], intervalos_confianza_n[i][0], intervalos_confianza_n[i][1]])
        csv_retardos_summary_writer.writerow(["total", retardo_medio_total, intervalo_confianza_total_t[0], intervalo_confianza_total_t[1], intervalo_confianza_total_n[0], intervalo_confianza_total_n[1]])
        csv_retardos_summary.close()

