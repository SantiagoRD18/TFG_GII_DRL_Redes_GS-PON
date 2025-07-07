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

# Traspone la lista de resultados y transforma la escala de bps a Mbps
def process_traffic(list, t_cycle):

    array = np.array(list)
    array = array.T
    array = array / (t_cycle*1e6) # Transformacion de bps a Mbps

    return array.tolist()