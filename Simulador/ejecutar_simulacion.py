### Importamos librerías
import simpy
import sys
from datetime import datetime
import time
import cProfile
import pstats
from bitarray import bitarray
from bitarray import util
import os
import csv

### Importamos módulos propios
from packages.classes.GeneraTrafico import GeneraTrafico
from packages.classes.ONT import ONT
from packages.classes.OLT import OLT
from packages.classes.OLT_IA import OLT_IA
from packages.classes.Enlace import Enlace

from packages.configuration.parameters import *


def ejecutar_simulacion(carga):         

    start_time = time.time()
    print(f"# Comienza simulación: (carga = {carga})")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    start_time_str = time.strftime("%Y%m%d_%H%M", time.gmtime())
    subdirectory= "data"
    filename = f"summary-carga_0{carga*1000:.0f}-{start_time_str}.txt"
    file_path = os.path.join(subdirectory, filename)
    f = open(file_path, "a")

    with cProfile.Profile() as pr:
        env = simpy.Environment()

        splitter_downstream = Enlace(env, N_ONTS)
        splitter_upstream = Enlace(env)

        capas_app_ont = []
        onts = []

        for i in range(N_ONTS): 
            capas_app_ont.append(GeneraTrafico(env, i, carga, i*datetime.utcnow().microsecond // 1000))
            onts.append(ONT(env, i, capas_app_ont[i], splitter_downstream, splitter_upstream))

        olt = OLT(env, splitter_upstream, splitter_downstream)

        env.run(until=T_SIM)

        sys.stdout.write('\r' + ' ' * 50 + '\r')
        sys.stdout.flush()

        print()

        csv_retardos_summary = open(os.path.join(subdirectory, f"retardos-summary-carga_0{carga*1000:.0f}-{start_time_str}.csv"), "w")
        csv_retardos_summary_writer = csv.writer(csv_retardos_summary, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_retardos_summary_writer.writerow(["ont", "prioridad", "retardo_medio", "intervalo_confianza_left", "intervalo_confianza_right"])

        for i in range(N_ONTS):
            for j in range(N_COLAS):
                csv_retardos_summary_writer.writerow([i, j, olt.retardos_estadisticas[i][j].media, olt.retardos_estadisticas[i][j].intervalo_confianza()[0], olt.retardos_estadisticas[i][j].intervalo_confianza()[1]])


    print('# Fin de la simulación. Datos relevantes: ')


    # Preparamos la tabla con los principales datos de la simulación
    colas = []
    cargas = []
    Bytes_generados = []
    Bytes_descartados = []
    retardos_medios = []
    paquetes_generados = []
    paquetes_descartados = []
    for i in range(N_ONTS):
        colas.append(0)
        for j in range (N_COLAS):
            colas[i] += sum(paq.len for paq in onts[i].generador_trafico.colas[j])
        cargas.append(onts[i].generador_trafico.Bytes_generados*8*1e-6/T_SIM)
        Bytes_generados.append(onts[i].generador_trafico.Bytes_generados)
        Bytes_descartados.append(onts[i].generador_trafico.Bytes_descartados)
        paquetes_generados.append(onts[i].generador_trafico.paquetes_generados)
        paquetes_descartados.append(onts[i].generador_trafico.paquetes_descartados)
        retardos_medios.append(sum(olt.retardos_estadisticas[i][j].media for j in range(N_COLAS))/N_COLAS)

    end_time = time.time()
    t_ejecucion = end_time - start_time
    horas, resto = divmod(int(t_ejecucion), 3600)
    minutos, segundos = divmod(resto, 60)

    print("## 29 - RED GPON IPACT")
    f.write("## 29 - RED GPON IPACT\n")
    print(f"# Parámetros simulación")
    f.write("# Parámetros simulación\n")
    print(f"\t- Nº ONUs = {N_ONTS}")
    f.write(f"\t- Nº ONUs = {N_ONTS}\n")
    print(f"\t- Tasa de transmisión de la red = {R_tx*1e-9:,.0f} Gbps")
    f.write(f"\t- Tasa de transmisión de la red = {R_tx*1e-9:,.0f} Gbps\n")
    print(f"\t- Paquetes de {tamano_payload[0]/8} B, {tamano_payload[1]/8} B, {tamano_payload[2]/8} B")
    f.write(f"\t- Paquetes de {tamano_payload[0]/8} B, {tamano_payload[1]/8} B, {tamano_payload[2]/8} B\n")
    print(f"\t- Longitud red = {L_RED/1e3} km")
    f.write(f"\t- Longitud red = {L_RED/1e3} km\n")
    print(f"\t- Tamaño buffer = {L_BUFFER_ONTS/8*1e-6} MB")
    f.write(f"\t- Tamaño buffer = {L_BUFFER_ONTS/8*1e-6} MB\n")
    print(f"\t- Método de inserción de paquetes por prioridad de colas")
    f.write(f"\t- Método de inserción de paquetes por prioridad de colas\n")
    print(f"\t- Método de extracción de colas de prioridad")
    f.write(f"\t- Método de extracción de colas de prioridad\n")
    print(f"\t- Nº de streams = {sum(N_SOURCES)}")
    f.write(f"\t- Nº de streams = {sum(N_SOURCES)}\n")
    print(f"\t- Una sola clase de servicio")
    f.write(f"\t- Una sola clase de servicio\n")
    if(multiples_colas):
        print(f"\t- Tres colas en cada ONU")
        f.write(f"\t- Tres colas en cada ONU\n")
    else:
        print(f"\t- Una cola en cada ONU")
        f.write(f"\t- Una cola en cada ONU\n")
    print(f"\t- T_SIM \t= {(T_SIM*1e9):,.0f} ns")
    f.write(f"\t- T_SIM \t= {(T_SIM*1e9):,.0f} ns\n")
    print(f"\t- T_CICLO \t= {(T_CICLO*1e9):,.0f} ns")
    f.write(f"\t- T_CICLO \t= {(T_CICLO*1e9):,.0f} ns\n")
    print(f"\t- T_GUARDA \t= {(T_GUARDA*1e9):,.0f} ns")
    f.write(f"\t- T_GUARDA \t= {(T_GUARDA*1e9):,.0f} ns\n")
    print(f"\t- T_REPORT \t= {(T_REPORT*1e9):,.0f} ns")
    f.write(f"\t- T_REPORT \t= {(T_REPORT*1e9):,.0f} ns\n")
    print(f"\t- T_AVAILABLE \t= {(T_AVAILABLE*1e9):,.0f} ns")
    f.write(f"\t- T_AVAILABLE \t= {(T_AVAILABLE*1e9):,.0f} ns\n")
    print(f"\t- T_propagacion \t= {(T_propagacion*1e9):,.3f} ns")
    f.write(f"\t- T_propagacion \t= {(T_propagacion*1e9):,.3f} ns\n")
    print(f"\t- T_tx_gate \t= {(tamano_gate/R_tx*1e9):,.0f} ns")
    f.write(f"\t- T_tx_gate \t= {(tamano_gate/R_tx*1e9):,.0f} ns\n")
    print(f"\t- carga = {carga}")
    f.write(f"\t- carga = {carga}\n")
    

    print(f"+-------------------------------------------------------+-----------------------+")
    print(f"|  TABLA 1                                                                      |") 
    print(f"+--------+----------------------+-----------------------+-----------------------+")
    print(f"+--------+----------------------+-----------------------+-----------------------+")
    print(f"| ONT Nº | Carga (Mbps)         | Retardo medio (s)     | B_alloc medio (Bytes) |")
    print(f"+--------+----------------------+-----------------------+-----------------------+")

    f.write(f"+-------------------------------------------------------+-----------------------+\n")
    f.write(f"|  TABLA 1                                                                      |\n")
    f.write(f"+--------+----------------------+-----------------------+-----------------------+\n")
    f.write(f"+--------+----------------------+-----------------------+-----------------------+\n")
    f.write(f"| ONT Nº | Carga (Mbps)         | Retardo medio (s)     | B_alloc medio (Bytes) |\n")
    f.write(f"+--------+----------------------+-----------------------+-----------------------+\n")


    for i in range (N_ONTS):
        try:
            print(f"| ONT {i:02d} | {cargas[i]:,.3f} \t\t| {retardos_medios[i]:.4E} \t\t| {olt.B_alloc_acum[i]/olt.n_alloc[i]/8:,.0f}\t\t|")
            f.write(f"| ONT {i:02d} | {cargas[i]:,.3f} \t\t| {retardos_medios[i]:.4E} \t\t| {olt.B_alloc_acum[i]/olt.n_alloc[i]/8:,.0f}\t\t|\n")
        except IndexError:
            print(f"| ONT {i:02d} | N/A \t\t| N/A \t| N/A \t\t|N/A \t\t|")
            f.write(f"| ONT {i:02d} | N/A \t\t| N/A \t| N/A \t\t|N/A \t\t|\n")

    print(f"+-------------------------------------------------------+-----------------------+")
    f.write(f"+-------------------------------------------------------+-----------------------+\n")
    try:
        print(f"| Media  | {sum(cargas)/len(cargas):,.3f} \t\t| {sum(retardos_medios)/len(cargas):.4E}\t\t\t|")
        f.write(f"| Media  | {sum(cargas)/len(cargas):,.3f} \t\t| {sum(retardos_medios)/len(cargas):.4E}\t\t\t|\n")
    except ZeroDivisionError:
        print(f"| Media  | N/A \t\t| N/A \t\t|N/A \t\t|")
        f.write(f"| Media  | N/A \t\t| N/A \t\t|N/A \t\t|\n")
    print(f"+-------------------------------------------------------+-----------------------+")
    f.write(f"+-------------------------------------------------------+-----------------------+\n")
    print()


    print(f"+--------+---------------------------------------------------------------------------------------------------------------+")
    f.write(f"+--------+---------------------------------------------------------------------------------------------------------------+\n")
    print(f"|  TABLA 2                                                                                                               |") 
    f.write(f"|  TABLA 2                                                                                                               |\n")
    print(f"+--------+----------------------+-----------------------+-----------------------+----------------------------------------+")
    f.write(f"+--------+----------------------+-----------------------+-----------------------+----------------------------------------+\n")
    print(f"| ONT Nº | Bytes generados      | Bytes descartados     | Paquetes generados    | Paquetes descartados  | Bytes en cola  |")
    f.write(f"| ONT Nº | Bytes generados      | Bytes descartados     | Paquetes generados    | Paquetes descartados  | Bytes en cola  |\n")
    print(f"+--------+----------------------+-----------------------+-----------------------+----------------------------------------+")
    f.write(f"+--------+----------------------+-----------------------+-----------------------+----------------------------------------+\n")

    for i in range (N_ONTS):
        try:
            print(f"| ONT {i:02d} | {Bytes_generados[i]:,.0f}", end="")
            f.write(f"| ONT {i:02d} | {Bytes_generados[i]:,.0f}")
            if(Bytes_generados[i]<1000):
                print("\t", end="")
                f.write("\t")
            print(f"\t\t| {Bytes_descartados[i]:,.0f}", end="")
            f.write(f"\t\t| {Bytes_descartados[i]:,.0f}")
            if(Bytes_descartados[i]<1000):
                print("\t", end="")
                f.write("\t")
            print(f"\t\t| {paquetes_generados[i]:,.0f}", end="")
            f.write(f"\t\t| {paquetes_generados[i]:,.0f}")
            if(paquetes_generados[i]<1000):
                print("\t", end="")
                f.write("\t")
            print(f"\t\t| {paquetes_descartados[i]:,.0f}", end="")
            f.write(f"\t\t| {paquetes_descartados[i]:,.0f}")
            if(paquetes_descartados[i]/8<1000):
                print("\t", end="")
                f.write("\t")
            print(f"\t\t| {colas[i]/8:,.0f}\t\t |")
            f.write(f"\t\t| {colas[i]/8:,.0f}\t\t |\n")
        except IndexError:
            print(f"| ONT {i:02d} | N/A \t\t| N/A \t| N/A \t\t | N/A \t\t | N/A \t\t |")
            f.write(f"| ONT {i:02d} | N/A \t\t| N/A \t| N/A \t\t | N/A \t\t | N/A \t\t |\n")

    print(f"+--------+----------------------+-----------------------+-----------------------+----------------------------------------+")
    f.write(f"+--------+----------------------+-----------------------+-----------------------+----------------------------------------+\n")
    try:
        print(f"| Total  | {sum(Bytes_generados):,.0f}", end="")
        f.write(f"| Total  | {sum(Bytes_generados):,.0f}")
        if(sum(Bytes_generados)<1000):
            print("\t", end="")
            f.write("\t")
        print(f"\t\t| {sum(Bytes_descartados):,.0f}", end="")
        f.write(f"\t\t| {sum(Bytes_descartados):,.0f}")
        if(sum(Bytes_descartados)<1000):
            print("\t", end="")
            f.write("\t")
        print(f"\t\t| {sum(paquetes_generados):,.0f}", end="")
        f.write(f"\t\t| {sum(paquetes_generados):,.0f}")
        if(sum(paquetes_generados)<1000):
            print("\t      ", end="")
            f.write("\t      ")
        print(f"\t\t| {sum(paquetes_descartados):,.0f}", end="")
        f.write(f"\t\t| {sum(paquetes_descartados):,.0f}")
        if(sum(paquetes_descartados)<1000):
            print("\t", end="")
            f.write("\t")
        print(f"\t\t| {sum(colas)/8:,.0f}\t|")
        f.write(f"\t\t| {sum(colas)/8:,.0f}\t|\n")  
    except ZeroDivisionError:
        print(f"| Total  | N/A \t\t| N/A \t| N/A \t\t|")
        f.write(f"| Total  | N/A \t\t| N/A \t| N/A \t\t|\n")
    print(f"+--------+----------------------+-----------------------+-----------------------+----------------------------------------+")
    f.write(f"+--------+----------------------+-----------------------+-----------------------+----------------------------------------+\n")
    try:
        print(f"| Media  | {sum(Bytes_generados)/len(Bytes_generados):,.0f}", end="")
        f.write(f"| Media  | {sum(Bytes_generados)/len(Bytes_generados):,.0f}")
        if(Bytes_generados[i]<1000):
            print("\t", end="")
            f.write("\t")
        print(f"\t\t| {sum(Bytes_descartados)/len(Bytes_descartados):,.0f}", end="")
        f.write(f"\t\t| {sum(Bytes_descartados)/len(Bytes_descartados):,.0f}")
        if(Bytes_descartados[i]<1000):
            print("\t", end="")
            f.write("\t")
        print(f"\t\t| {sum(paquetes_generados)/len(paquetes_generados):,.0f}", end="")
        f.write(f"\t\t| {sum(paquetes_generados)/len(paquetes_generados):,.0f}")
        if(paquetes_generados[i]<1000):
            print("\t", end="")
            f.write("\t")
        print(f"\t\t| {sum(paquetes_descartados)/len(paquetes_descartados):,.0f}", end="")
        f.write(f"\t\t| {sum(paquetes_descartados)/len(paquetes_descartados):,.0f}")
        if(sum(paquetes_descartados)<1000):
            print("\t", end="")
            f.write("\t")
        print(f"\t\t| {sum(colas)/8/len(colas):,.0f}\t\t|")
        f.write(f"\t\t| {sum(colas)/8/len(colas):,.0f}\t\t|\n")
    except ZeroDivisionError:
        print(f"| Media  | N/A \t\t| N/A \t| N/A \t\t|")
        f.write(f"| Media  | N/A \t\t| N/A \t| N/A \t\t|\n")
    print(f"+--------+----------------------+-----------------------+-----------------------+----------------------------------------+")
    f.write(f"+--------+----------------------+-----------------------+-----------------------+----------------------------------------+\n")

    print(f"Tiempo total ejecución : {horas}h {minutos}m {segundos:.2f}s")
    f.write(f"Tiempo total ejecución : {horas}h {minutos}m {segundos:.2f}s\n")
    print(f"T_sim = {T_SIM*1e9:,.0f} ns")
    f.write(f"T_sim = {T_SIM*1e9:,.0f} ns\n")
    print(f"T comienzo simulación = {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    f.write(f"T comienzo simulación = {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"T fin  simulación = {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    f.write(f"T fin  simulación = {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"Paquetes que han llegado a la OLT : {olt.contador_paquetes_recibidos_olt:,.0f}")
    f.write(f"Paquetes que han llegado a la OLT : {olt.contador_paquetes_recibidos_olt:,.0f}\n")
    print(f"Bytes que han llegado a la OLT : {olt.contador_Bytes_recibidos_olt:,.0f}")
    f.write(f"Bytes que han llegado a la OLT : {olt.contador_Bytes_recibidos_olt:,.0f}\n")
    print(f"Bytes descartados por las ONTs en total: {sum(Bytes_descartados):,.0f}")
    f.write(f"Bytes descartados por las ONTs en total: {sum(Bytes_descartados):,.0f}\n")
    i=0
    b_alloc_total=0
    for b_alloc in olt.B_alloc_Mbps:
        print(f"B_alloc_Mbps[{i}]= {((b_alloc/T_SIM)/1e6):,.3f} Mbps")
        f.write(f"B_alloc_Mbps[{i}]= {((b_alloc/T_SIM)/1e6):,.3f} Mbps\n")
        b_alloc_total=b_alloc+b_alloc_total
        i=i+1
    print(f"B_alloc_Mbps_Total= {((b_alloc_total/T_SIM)/1e6):,.3f} Mbps")
    f.write(f"B_alloc_Mbps_Total= {((b_alloc_total/T_SIM)/1e6):,.3f} Mbps\n")
    print(f"B_alloc_Mbps_Media= {(((b_alloc_total/N_ONTS)/T_SIM)/1e6):,.3f} Mbps")
    f.write(f"B_alloc_Mbps_Media= {(((b_alloc_total/N_ONTS)/T_SIM)/1e6):,.3f} Mbps\n")
    

    print(f"t_ejecucion / t_sim = {t_ejecucion/T_SIM:.2f}")
    f.write(f"t_ejecucion / t_sim = {t_ejecucion/T_SIM:.2f}\n")

    f.close()


    if mostrar_profiling:
        print("\n##############################################################################")
        print("Stats de profiling")
        print("\n##############################################################################")
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()

    # Guardamos en un fichero el tamaño medio de las colas de cada onu
    filename_csv_colas = f"colas-carga_0{carga*10:.0f}-{start_time_str}.csv"
    csv_colas = open(os.path.join(subdirectory, filename_csv_colas), "w")
    csv_colas_writer = csv.writer(csv_colas, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    csv_colas_writer.writerow(["ont_id", "cola_tamano_total_bytes"])
    for i in range(N_ONTS):
        csv_colas_writer.writerow([i, colas[i]/8])
    csv_colas_writer.writerow(["media", sum(colas)/8/len(colas)])

    # Guardamos en un fichero los Mbps de cada ONU, los Mbps totales y los Mbps medios 
    filename_csv_Mbps = f"Mbps_0{carga*10:.0f}-{start_time_str}.csv"
    csv_Mbps = open(os.path.join(subdirectory,filename_csv_Mbps),"w")
    csv_Mbps_writer = csv.writer(csv_Mbps,delimiter=' ',quotechar='"',quoting=csv.QUOTE_MINIMAL)

    csv_Mbps_writer.writerow(["ont_id","Tamano_cola_Mbps"])
    for i in range (N_ONTS):
        csv_Mbps_writer.writerow([i,(olt.B_alloc_Mbps[i]/T_SIM)/1e6])
    csv_Mbps_writer.writerow(["Media", ((b_alloc_total/N_ONTS)/T_SIM)/1e6])
    csv_Mbps_writer.writerow(["Total",(b_alloc_total)/T_SIM/1e6])

    