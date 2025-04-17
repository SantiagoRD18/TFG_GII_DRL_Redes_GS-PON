import numpy as np
import matplotlib.pyplot as plt

# Dibuja la gráfica correspondiente al tráfico de entrada y salida de una ont
def grafica_es(trafico_entrada, trafico_salida, num_ont):

    valores_x = np.arange(2, 2 * len(trafico_entrada) + 1, 2)

    plt.figure(figsize=(12, 6))
    plt.ylim(0,1100)
    plt.xlim(0, 2 * len(trafico_entrada) + 1)
    plt.xlabel('Tiempo en milisegundos')
    plt.ylabel('Ancho de banda en Mbps')
    plt.plot(valores_x, trafico_entrada, label=f'Tráfico de entrada de la ONT {num_ont+1}')
    plt.plot(valores_x, trafico_salida, label=f'Tráfico de salida de la ONT {num_ont+1}')
    plt.title(f'Grafica del trafico de entrada y salida de la ONT {num_ont+1} en Mbps')
    plt.legend()
    plt.show()

# Dibuja la gráfica correspondiente al tráfico pendiente de una ont
def grafica_pendiente(trafico_pendiente, num_ont):

    valores_x = np.arange(2, 2 * len(trafico_pendiente) + 1, 2)

    plt.figure(figsize=(12, 6))
    plt.ylim(0, max(trafico_pendiente))
    plt.xlim(0, 2 * len(trafico_pendiente) + 1)
    plt.xlabel('Tiempo en milisegundos')
    plt.ylabel('Tamaño de la cola en Mbits')
    plt.plot(valores_x, trafico_pendiente, label=f'Grafica del trafico pendiente de la ONT {num_ont+1}')
    plt.title(f'Grafica del trafico pendiente de la ONT {num_ont+1} en Mbits en el ciclo determinado')
    plt.show()

# Dibuja la gráfica correspondiente a los valores de pareto de una ont
def grafica_pareto(valores_instantes, num_ont, n_ciclos):

    valoresInstantesFinales=[]
    valoresPareto=[]

    # Modificar los valores de Pareto en posiciones pares donde el instante es 0
    for i in valores_instantes:
        cont=0
        for j in i:
            if cont%2==0:
                if j==0:
                    valoresInstantesFinales.append(j)
                    valoresPareto.append(0)
                else:
                    valoresInstantesFinales.append(j)
                    valoresPareto.append(1)
            else:
                if j==0:
                    valoresInstantesFinales.append(j)
                    valoresPareto.append(0)
                else:
                    valoresInstantesFinales.append(j)
                    valoresPareto.append(0)

            cont+=1

    extended_instantes = []
    extended_pareto = []
    
    current_time = 0
    
    # Añadir valores extendidos para mantener el gráfico en el eje y hasta el siguiente instante
    for i, valor in enumerate(valoresInstantesFinales):
        current_time += valor
        # Añadir el tiempo actual y su valor correspondiente de Pareto
        extended_instantes.append(current_time)
        if i < len(valoresPareto) - 1:
            extended_pareto.append(valoresPareto[i])
            extended_instantes.append(current_time)
            extended_pareto.append(valoresPareto[i+1])
        else:
            extended_pareto.append(valoresPareto[i])
    
    plt.figure(figsize=(12, 6))
    plt.step(extended_instantes, extended_pareto, where='post', linestyle='-', color='blue')
    plt.xlabel('Tiempo Acumulado')
    plt.ylabel('Valores (Pareto)')
    plt.title(f'Gráfica de valores de Pareto de la ONT {num_ont+1}')
    plt.xlim([0,n_ciclos/5])
    plt.grid(True)
    plt.show()



# Traspone la lista de resultados y transforma la escala de bps a Mbps
def transformar_trafico(list, t_ciclo):

    array = np.array(list)
    array = array.T
    array = array / (t_ciclo*1e6) # Transformacion de bps a Mbps
    return array.tolist()

# Funcion auxiliar para el array de los instantes de los ciclos y luego calcular la grafica de barras
def calcular_instantes(array_valores, num_onts):

    x=[[] for i in range(num_onts)]

    for i, subarray in enumerate(array_valores):
        for j, sublist in enumerate(subarray):
            for k, valor in enumerate(sublist):
                x[j].append(valor)
    return x