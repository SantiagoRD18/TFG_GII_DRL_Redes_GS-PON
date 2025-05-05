import numpy as np
import matplotlib.pyplot as plt

# Dibuja la gráfica correspondiente al tráfico de entrada y salida de una ont
def plot_input_output(input_traffic, output_traffic, num_ont):

    x_values = np.arange(2, 2 * len(input_traffic) + 1, 2)

    plt.figure(figsize=(12, 6))
    plt.ylim(0,1100)
    plt.xlim(0, 2 * len(input_traffic) + 1)
    plt.xlabel('Tiempo en milisegundos')
    plt.ylabel('Ancho de banda en Mbps')
    plt.plot(x_values, input_traffic, label=f'Tráfico de entrada de la ONT {num_ont+1}')
    plt.plot(x_values, output_traffic, label=f'Tráfico de salida de la ONT {num_ont+1}')
    plt.title(f'Grafica del trafico de entrada y salida de la ONT {num_ont+1} en Mbps')
    plt.legend()
    plt.show()

# Dibuja la gráfica correspondiente al tráfico pendiente de una ont
def plot_pending(pending_traffic, num_ont):

    x_values = np.arange(2, 2 * len(pending_traffic) + 1, 2)

    plt.figure(figsize=(12, 6))
    plt.ylim(0, max(pending_traffic))
    plt.xlim(0, 2 * len(pending_traffic) + 1)
    plt.xlabel('Tiempo en milisegundos')
    plt.ylabel('Tamaño de la cola en Mbits')
    plt.plot(x_values, pending_traffic, label=f'Grafica del trafico pendiente de la ONT {num_ont+1}')
    plt.title(f'Grafica del trafico pendiente de la ONT {num_ont+1} en Mbits en el ciclo determinado')
    plt.show()

# Dibuja la gráfica correspondiente a los valores de pareto de una ont
def plot_pareto(instant_values, num_ont, n_cycles):

    final_instants_values=[]
    pareto_values=[]

    # Modificar los valores de Pareto en posiciones pares donde el instante es 0
    for i in instant_values:
        counter=0
        for j in i:
            if counter%2==0:
                if j==0:
                    final_instants_values.append(j)
                    pareto_values.append(0)
                else:
                    final_instants_values.append(j)
                    pareto_values.append(1)
            else:
                if j==0:
                    final_instants_values.append(j)
                    pareto_values.append(0)
                else:
                    final_instants_values.append(j)
                    pareto_values.append(0)

            counter+=1

    extended_instants = []
    extended_pareto = []
    
    current_time = 0
    
    # Añadir valores extendidos para mantener el gráfico en el eje y hasta el siguiente instante
    for i, value in enumerate(final_instants_values):
        current_time += value
        # Añadir el tiempo actual y su valor correspondiente de Pareto
        extended_instants.append(current_time)
        if i < len(pareto_values) - 1:
            extended_pareto.append(pareto_values[i])
            extended_instants.append(current_time)
            extended_pareto.append(pareto_values[i+1])
        else:
            extended_pareto.append(pareto_values[i])
    
    plt.figure(figsize=(12, 6))
    plt.step(extended_instants, extended_pareto, where='post', linestyle='-', color='blue')
    plt.xlabel('Tiempo Acumulado')
    plt.ylabel('Valores (Pareto)')
    plt.title(f'Gráfica de valores de Pareto de la ONT {num_ont+1}')
    plt.xlim([0,n_cycles/5])
    plt.grid(True)
    plt.show()



# Traspone la lista de resultados y transforma la escala de bps a Mbps
def process_traffic(list, t_cycle):

    array = np.array(list)
    array = array.T
    array = array / (t_cycle*1e6) # Transformacion de bps a Mbps

    return array.tolist()

# Funcion auxiliar para el array de los instantes de los ciclos y luego calcular la grafica de barras
def calculate_instants(values_array, num_onts):

    x=[[] for i in range(num_onts)]

    for i, subarray in enumerate(values_array):
        for j, sublist in enumerate(subarray):
            for k, valor in enumerate(sublist):
                x[j].append(valor)

    return x