import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import sys
import os
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import queue

# Configurar matplotlib antes de importarlo para evitar conflictos con Tkinter
import matplotlib
matplotlib.use('Agg')  # Usar backend Agg que no requiere GUI
import matplotlib.pyplot as plt

# Importar módulos del proyecto
from agente.main_agent import main as agent_main
from simulador.main_sim import main as sim_main

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("SimDeepPON-RL")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Configurar el cierre de la ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variables para los parámetros
        self.setup_variables()
        
        # Configurar la interfaz
        self.setup_ui()
        
    def setup_variables(self):
        """Inicializar variables de la interfaz"""
        # Variables del agente
        self.num_ont = tk.IntVar(value=16)
        self.tx_rate = tk.DoubleVar(value=10e9)
        self.temp_ciclo = tk.DoubleVar(value=0.002)
        self.b_guaranteed = tk.DoubleVar(value=800e6)
        self.load = tk.DoubleVar(value=0.4)
        self.timesteps = tk.IntVar(value=2000000)
        self.nombre_modelo = tk.StringVar(value="modelo_prueba")
        self.n_ciclos = tk.IntVar(value=300)
        self.seed = tk.IntVar(value=42)
        self.num_envs = tk.IntVar(value=8)  # Nueva variable para entornos paralelos
        
        # Variables del simulador
        self.modo_simulador = tk.StringVar(value="0")  # 0: normal, 1: DRL, 2: DNN
        self.modelo_rl_nombre = tk.StringVar(value="modelo_prueba")  # Nombre del modelo para DRL
        
        # Variables de control de la interfaz
        self.current_operation_type = None  # 'train', 'eval', 'both', 'simulator'
        self.selected_sim_mode = None  # Para el modo del simulador
        
        # Variables de control de ejecución
        self.stop_event = threading.Event()
        self.current_thread = None
        self.is_running = False
        
        # Cola para comunicación segura entre hilos
        self.message_queue = queue.Queue()
        
        # Iniciar el procesamiento de mensajes
        self.process_queue()
        
    def setup_ui(self):
        """Configurar la interfaz de usuario"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar peso de las columnas y filas
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Título
        title_label = ttk.Label(main_frame, text="Sistema de Redes Ópticas", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Frame para selección principal
        selection_frame = ttk.LabelFrame(main_frame, text="Seleccionar Módulo", padding="10")
        selection_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Botones principales
        btn_agente = ttk.Button(selection_frame, text="Ejecutar Agente RL", 
                               command=self.show_agent_options, width=20)
        btn_agente.grid(row=0, column=0, padx=(0, 10))
        
        btn_simulador = ttk.Button(selection_frame, text="Ejecutar Simulador", 
                                  command=self.show_simulator_options, width=20)
        btn_simulador.grid(row=0, column=1, padx=(10, 0))
        
        # Frame para opciones específicas
        self.options_frame = ttk.LabelFrame(main_frame, text="Opciones", padding="10")
        self.options_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.options_frame.columnconfigure(1, weight=1)
        
        # Frame para parámetros
        self.params_frame = ttk.LabelFrame(main_frame, text="Parámetros", padding="10")
        self.params_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.params_frame.columnconfigure(1, weight=1)
        
        # Frame para log de salida
        log_frame = ttk.LabelFrame(main_frame, text="Log de Ejecución", padding="10")
        log_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Text area para logs
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, width=80)
        self.log_text.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame para botones del log
        log_buttons_frame = ttk.Frame(log_frame)
        log_buttons_frame.grid(row=1, column=0, columnspan=2, pady=(5, 0))
        
        # Botón para limpiar log
        btn_clear_log = ttk.Button(log_buttons_frame, text="Limpiar Log", command=self.clear_log)
        btn_clear_log.grid(row=0, column=0)
        
        # Inicialmente ocultar frames de opciones y parámetros
        self.hide_option_frames()
        
    def hide_option_frames(self):
        """Ocultar los frames de opciones y parámetros"""
        for widget in self.options_frame.winfo_children():
            widget.destroy()
        for widget in self.params_frame.winfo_children():
            widget.destroy()
            
    def show_agent_options(self):
        """Mostrar opciones del agente"""
        self.hide_option_frames()
        
        # Opciones del agente
        ttk.Label(self.options_frame, text="Tipo de Operación:").grid(row=0, column=0, sticky=tk.W)
        
        btn_train = ttk.Button(self.options_frame, text="Entrenamiento", 
                              command=lambda: self.select_agent_operation('train'), width=15)
        btn_train.grid(row=0, column=1, padx=(10, 5))
        
        btn_eval = ttk.Button(self.options_frame, text="Evaluación", 
                             command=lambda: self.select_agent_operation('eval'), width=15)
        btn_eval.grid(row=0, column=2, padx=5)
        
        btn_both = ttk.Button(self.options_frame, text="Ambos", 
                             command=lambda: self.select_agent_operation('both'), width=15)
        btn_both.grid(row=0, column=3, padx=(5, 0))
        
    def show_simulator_options(self):
        """Mostrar opciones del simulador"""
        self.hide_option_frames()
        
        # Opciones del simulador
        ttk.Label(self.options_frame, text="Modo de Simulación:").grid(row=0, column=0, sticky=tk.W)
        
        btn_normal = ttk.Button(self.options_frame, text="Simulador Normal", 
                               command=lambda: self.select_simulator_mode('0'), width=18)
        btn_normal.grid(row=0, column=1, padx=(10, 5))
        
        btn_drl = ttk.Button(self.options_frame, text="Simulador con DRL", 
                            command=lambda: self.select_simulator_mode('1'), width=18)
        btn_drl.grid(row=0, column=2, padx=5)
        
        btn_dnn = ttk.Button(self.options_frame, text="Simulador con DNN", 
                            command=lambda: self.select_simulator_mode('2'), width=18)
        btn_dnn.grid(row=0, column=3, padx=(5, 0))
        
    def execute_agent(self, tipo):
        """Ejecutar el agente en un hilo separado"""
        if self.is_running:
            messagebox.showwarning("Advertencia", "Ya hay una ejecución en curso.")
            return
            
        def run_agent():
            try:
                self.is_running = True
                self.stop_event.clear()
                
                self.log(f"Iniciando ejecución del agente - Tipo: {tipo}")
                if tipo == 'train' or tipo == 'both':
                    self.log(f"Parámetros: ONTs={self.num_ont.get()}, Load={self.load.get()}, Timesteps={self.timesteps.get()}, Entornos={self.num_envs.get()}")
                else:
                    self.log(f"Parámetros: ONTs={self.num_ont.get()}, Load={self.load.get()}, Ciclos={self.n_ciclos.get()}")
                
                # Verificar si se ha solicitado parar antes de comenzar
                if self.stop_event.is_set():
                    self.log("Ejecución cancelada por el usuario.")
                    return
                
                # Crear argumentos para el agente
                args = argparse.Namespace(
                    tipo=tipo,
                    num_ont=self.num_ont.get(),
                    TxRate=self.tx_rate.get(),
                    temp_ciclo=self.temp_ciclo.get(),
                    B_guaranteed=np.full(self.num_ont.get(), self.b_guaranteed.get()),
                    env_id='RedesOpticasEnv-v0',
                    num_envs=self.num_envs.get(),
                    algorithm='ppo',
                    load=self.load.get(),
                    timesteps=self.timesteps.get(),
                    nombre_modelo=self.nombre_modelo.get(),
                    n_ciclos=self.n_ciclos.get(),
                    seed=self.seed.get()
                )
                
                # Verificar nuevamente antes de ejecutar
                if self.stop_event.is_set():
                    self.log("Ejecución cancelada por el usuario antes de iniciarse.")
                    return
                
                # Ejecutar el agente
                self.log("Ejecutando agente... (Nota: puede tomar varios minutos)")
                agent_main(args)
                
                # Verificar si se solicitó parar durante la ejecución
                if self.stop_event.is_set():
                    self.message_queue.put({'type': 'log', 'message': "Ejecución del agente interrumpida por el usuario."})
                    return
                
                self.message_queue.put({'type': 'log', 'message': f"Ejecución del agente completada exitosamente - Tipo: {tipo}"})
                
                # Si es evaluación o ambos, preguntar si se quieren visualizar los resultados
                if tipo in ['eval', 'both'] and not self.stop_event.is_set():
                    # Enviar mensaje a la cola para procesar en el hilo principal
                    self.message_queue.put({
                        'type': 'ask_show_results', 
                        'filename': f"test-{self.nombre_modelo.get()}.csv"
                    })
                
            except Exception as e:
                if self.stop_event.is_set():
                    self.message_queue.put({'type': 'log', 'message': "Ejecución del agente interrumpida por el usuario."})
                else:
                    error_msg = f"Error en la ejecución del agente: {str(e)}"
                    self.message_queue.put({'type': 'log', 'message': error_msg})
                    self.message_queue.put({'type': 'error', 'message': error_msg})
            finally:
                self.is_running = False
        
        # Ejecutar en hilo separado para no bloquear la UI
        self.current_thread = threading.Thread(target=run_agent, daemon=False)
        self.current_thread.start()
        
    def execute_simulator(self, modo):
        """Ejecutar el simulador en un hilo separado"""
        if self.is_running:
            messagebox.showwarning("Advertencia", "Ya hay una ejecución en curso.")
            return
            
        def run_simulator():
            try:
                self.is_running = True
                self.stop_event.clear()
                
                modo_names = {"0": "Normal", "1": "DRL", "2": "DNN"}
                modo_name = modo_names.get(modo, modo)
                
                # Validaciones específicas según el modo
                if modo == "1":
                    # Validar que se ha especificado un nombre de modelo para DRL
                    modelo_nombre = self.modelo_rl_nombre.get().strip()
                    if not modelo_nombre:
                        self.message_queue.put({'type': 'error', 'message': "Debe especificar un nombre de modelo para la simulación DRL"})
                        return
                    self.log(f"Iniciando simulación - Modo: {modo_name} - Modelo: {modelo_nombre}")
                else:
                    self.log(f"Iniciando simulación - Modo: {modo_name}")
                
                # Verificar si se ha solicitado parar antes de comenzar
                if self.stop_event.is_set():
                    self.log("Simulación cancelada por el usuario.")
                    return
                
                # Verificar nuevamente antes de ejecutar
                if self.stop_event.is_set():
                    self.log("Simulación cancelada por el usuario antes de iniciarse.")
                    return
                
                # Ejecutar el simulador
                self.log("Ejecutando simulador... (Nota: puede tomar varios minutos)")
                
                # Pasar el nombre del modelo si es modo DRL
                if modo == "1":
                    sim_main(modo, self.modelo_rl_nombre.get())
                else:
                    sim_main(modo)
                
                # Verificar si se solicitó parar durante la ejecución
                if self.stop_event.is_set():
                    self.message_queue.put({'type': 'log', 'message': "Simulación interrumpida por el usuario."})
                    return
                
                self.message_queue.put({'type': 'log', 'message': f"Simulación completada exitosamente - Modo: {modo_name}"})
                
            except Exception as e:
                if self.stop_event.is_set():
                    self.message_queue.put({'type': 'log', 'message': "Simulación interrumpida por el usuario."})
                else:
                    error_msg = f"Error en la simulación: {str(e)}"
                    self.message_queue.put({'type': 'log', 'message': error_msg})
                    self.message_queue.put({'type': 'error', 'message': error_msg})
            finally:
                self.is_running = False
        
        # Ejecutar en hilo separado para no bloquear la UI
        self.current_thread = threading.Thread(target=run_simulator, daemon=False)
        self.current_thread.start()
        
    def force_quit(self):
        """Forzar la terminación completa de la aplicación"""
        try:
            import sys
            import os
            self.log("Terminando aplicación...")
            self.root.quit()
            self.root.destroy()
            os._exit(0)  # Forzar salida completa
        except:
            sys.exit(0)
    
    def log(self, message):
        """Añadir mensaje al log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, formatted_message)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def clear_log(self):
        """Limpiar el log"""
        self.log_text.delete(1.0, tk.END)
        
    def show_evaluation_results(self, filename):
        """Mostrar ventana de selección de gráficas"""
        from pathlib import Path
        try:
            # Obtener la ruta correcta según si estamos en ejecutable o código fuente
            if getattr(sys, 'frozen', False):
                # Estamos en un ejecutable de PyInstaller
                base_path = Path(__file__).resolve().parent
                csv_path = os.path.join(base_path, "agente", "logs", filename)
            else:
                # Estamos ejecutando desde código fuente
                csv_path = os.path.join("./agente/logs", filename)
            
            if not os.path.exists(csv_path):
                self.log(f"No se encontró el archivo de resultados: {csv_path}")
                messagebox.showerror("Error", f"No se encontró el archivo de resultados: {csv_path}")
                return
            
            # Leer el CSV
            df = pd.read_csv(csv_path)
            self.log(f"Leyendo resultados desde: {filename}")
            
            # Extraer datos
            onts = df['onts'].tolist()
            trafico_entrada = df['trafico_entrada'].apply(eval).tolist()  # eval para convertir string a lista
            trafico_salida = df['trafico_salida'].apply(eval).tolist()
            colas = df['colas'].apply(eval).tolist()
            
            # Crear ventana de selección de gráficas
            self.create_graph_selector_window(onts, trafico_entrada, trafico_salida, colas)
            
        except Exception as e:
            self.log(f"Error al mostrar los resultados: {str(e)}")
            messagebox.showerror("Error", f"Error al mostrar los resultados: {str(e)}")
    
    def plot_input_output(self, input_traffic, output_traffic, num_ont):
        """Mostrar gráfica de tráfico de entrada y salida"""
        x_values = np.arange(2, 2 * len(input_traffic) + 1, 2)

        plt.figure(figsize=(12, 6))
        plt.ylim(0, 1100)
        plt.xlim(0, 2 * len(input_traffic) + 1)
        plt.xlabel('Tiempo en milisegundos')
        plt.ylabel('Ancho de banda en Mbps')
        plt.plot(x_values, input_traffic, label=f'Tráfico de entrada de la ONT {num_ont+1}')
        plt.plot(x_values, output_traffic, label=f'Tráfico de salida de la ONT {num_ont+1}')
        plt.title(f'Gráfica del tráfico de entrada y salida de la ONT {num_ont+1} en Mbps')
        plt.legend()
        
        # Guardar y mostrar la imagen
        temp_file = f"temp_input_output_ont_{num_ont}.png"
        plt.savefig(temp_file, dpi=100, bbox_inches='tight')
        plt.close()  # Cerrar la figura para liberar memoria
        
        # Abrir la imagen con el visor predeterminado del sistema
        try:
            os.startfile(temp_file)  # Windows
        except AttributeError:
            os.system(f"open {temp_file}")  # macOS/Linux

    def plot_pending(self, pending_traffic, num_ont):
        """Mostrar gráfica de tráfico pendiente"""
        x_values = np.arange(2, 2 * len(pending_traffic) + 1, 2)

        plt.figure(figsize=(12, 6))
        plt.ylim(0, max(pending_traffic) if pending_traffic else 1)
        plt.xlim(0, 2 * len(pending_traffic) + 1)
        plt.xlabel('Tiempo en milisegundos')
        plt.ylabel('Tamaño de la cola en Mbits')
        plt.plot(x_values, pending_traffic, label=f'Gráfica del tráfico pendiente de la ONT {num_ont+1}')
        plt.title(f'Gráfica del tráfico pendiente de la ONT {num_ont+1} en Mbits en el ciclo determinado')
        plt.legend()
        
        # Guardar y mostrar la imagen
        temp_file = f"temp_pending_ont_{num_ont}.png"
        plt.savefig(temp_file, dpi=100, bbox_inches='tight')
        plt.close()  # Cerrar la figura para liberar memoria
        
        # Abrir la imagen con el visor predeterminado del sistema
        try:
            os.startfile(temp_file)  # Windows
        except AttributeError:
            os.system(f"open {temp_file}")  # macOS/Linux

    def ask_show_results(self, filename):
        """Preguntar al usuario si quiere visualizar los resultados"""
        try:
            result = messagebox.askyesno(
                "Visualizar Resultados", 
                "La evaluación ha terminado. ¿Desea visualizar los resultados gráficos?",
                icon='question'
            )
            
            if result:
                self.log("Mostrando resultados gráficos...")
                # Ejecutar directamente en el hilo principal
                self.show_evaluation_results(filename)
            else:
                self.log("Resultados guardados. Las gráficas no se mostrarán.")
                
        except Exception as e:
            self.log(f"Error al preguntar sobre resultados: {str(e)}")
    
    def process_queue(self):
        """Procesar mensajes de la cola de forma segura en el hilo principal"""
        try:
            while True:
                message = self.message_queue.get_nowait()
                message_type = message.get('type')
                
                if message_type == 'ask_show_results':
                    filename = message.get('filename')
                    self.ask_show_results(filename)
                elif message_type == 'show_results':
                    filename = message.get('filename')
                    self.show_evaluation_results(filename)
                elif message_type == 'log':
                    message_text = message.get('message')
                    self.log(message_text)
                elif message_type == 'error':
                    message_text = message.get('message')
                    messagebox.showerror("Error", message_text)
                    
        except queue.Empty:
            pass
        
        # Programar la siguiente verificación de la cola
        self.root.after(100, self.process_queue)
    
    def cleanup_temp_files(self):
        """Limpiar archivos temporales de gráficas"""
        try:
            import glob
            temp_files = glob.glob("temp_*.png")
            for file in temp_files:
                try:
                    os.remove(file)
                except:
                    pass  # Ignorar errores si el archivo está en uso
        except Exception as e:
            self.log(f"Error al limpiar archivos temporales: {str(e)}")
    
    def create_graph_selector_window(self, onts, trafico_entrada, trafico_salida, colas):
        """Crear ventana para seleccionar qué gráfica mostrar"""
        # Crear nueva ventana
        selector_window = tk.Toplevel(self.root)
        selector_window.title("Selector de Gráficas de Resultados")
        selector_window.geometry("450x450")  # Aumentado de 350 a 450 para que quepa todo
        selector_window.resizable(False, False)
        
        # Hacer que la ventana sea modal
        selector_window.transient(self.root)
        selector_window.grab_set()
        
        # Frame principal
        main_frame = ttk.Frame(selector_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        title_label = ttk.Label(main_frame, text="Seleccionar Gráfica a Mostrar", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Frame para selección de ONT
        ont_frame = ttk.LabelFrame(main_frame, text="Seleccionar ONT", padding="10")
        ont_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(ont_frame, text="Elegir ONT:").pack(anchor=tk.W)
        ont_var = tk.StringVar()
        ont_combo = ttk.Combobox(ont_frame, textvariable=ont_var, state="readonly", width=35)
        ont_combo['values'] = [f"ONT {ont_id + 1}" for ont_id in onts]
        ont_combo.current(0)  # Seleccionar la primera por defecto
        ont_combo.pack(fill=tk.X, pady=(5, 0))
        
        # Frame para selección de tipo de gráfica
        graph_frame = ttk.LabelFrame(main_frame, text="Tipo de Gráfica", padding="10")
        graph_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(graph_frame, text="Seleccionar tipo de gráfica:").pack(anchor=tk.W, pady=(0, 5))
        graph_var = tk.StringVar(value="entrada_salida")
        
        # Radio button para entrada y salida
        radio1 = ttk.Radiobutton(graph_frame, text="Tráfico de Entrada y Salida", 
                                variable=graph_var, value="entrada_salida")
        radio1.pack(anchor=tk.W, pady=(5, 5))
        
        # Radio button para colas
        radio2 = ttk.Radiobutton(graph_frame, text="Tráfico Pendiente (Colas)", 
                                variable=graph_var, value="colas")
        radio2.pack(anchor=tk.W, pady=(0, 5))
        
        # Frame para botones de acción
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(15, 0))
        
        def show_selected_graph():
            """Mostrar la gráfica seleccionada"""
            try:
                # Obtener selecciones
                ont_index = ont_combo.current()
                graph_type = graph_var.get()
                
                if ont_index < 0:
                    messagebox.showwarning("Advertencia", "Por favor seleccione una ONT.")
                    return
                
                ont_id = onts[ont_index]
                
                # Mostrar la gráfica seleccionada
                if graph_type == "entrada_salida":
                    self.plot_input_output(trafico_entrada[ont_index], trafico_salida[ont_index], ont_id)
                    self.log(f"Mostrando gráfica de entrada/salida para ONT {ont_id + 1}")
                elif graph_type == "colas":
                    self.plot_pending(colas[ont_index], ont_id)
                    self.log(f"Mostrando gráfica de colas para ONT {ont_id + 1}")
                
                # Programar limpieza de archivos temporales
                self.root.after(30000, self.cleanup_temp_files)
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al mostrar la gráfica: {str(e)}")
        
        def show_all_graphs():
            """Mostrar todas las gráficas (comportamiento anterior)"""
            try:
                for i, ont_id in enumerate(onts):
                    self.plot_input_output(trafico_entrada[i], trafico_salida[i], ont_id)
                    self.plot_pending(colas[i], ont_id)
                
                self.log(f"Mostrando todas las gráficas para {len(onts)} ONTs")
                # Programar limpieza de archivos temporales
                self.root.after(30000, self.cleanup_temp_files)
                selector_window.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al mostrar las gráficas: {str(e)}")
        
        def close_window():
            """Cerrar la ventana sin mostrar gráficas"""
            self.log("Selector de gráficas cerrado sin mostrar resultados.")
            selector_window.destroy()
        
        # Botones principales
        btn_show = ttk.Button(button_frame, text="Mostrar Gráfica Seleccionada", 
                             command=show_selected_graph, width=30)
        btn_show.pack(pady=(0, 8))
        
        # Frame para botones secundarios
        secondary_buttons = ttk.Frame(button_frame)
        secondary_buttons.pack(fill=tk.X)
        
        btn_show_all = ttk.Button(secondary_buttons, text="Mostrar Todas", 
                                 command=show_all_graphs, width=18)
        btn_show_all.pack(side=tk.LEFT, padx=(0, 5))
        
        btn_close = ttk.Button(secondary_buttons, text="Cerrar", 
                              command=close_window, width=12)
        btn_close.pack(side=tk.RIGHT)
        
        # Centrar la ventana
        selector_window.update_idletasks()
        x = (selector_window.winfo_screenwidth() // 2) - (selector_window.winfo_width() // 2)
        y = (selector_window.winfo_screenheight() // 2) - (selector_window.winfo_height() // 2)
        selector_window.geometry(f"+{x}+{y}")
        
        # Establecer foco en el primer elemento
        ont_combo.focus_set()
        
        # Manejar el cierre de la ventana
        selector_window.protocol("WM_DELETE_WINDOW", close_window)

    def on_closing(self):
        """Manejar el cierre de la ventana principal"""
        if self.is_running:
            result = messagebox.askyesno(
                "Cerrar Aplicación", 
                "Hay una ejecución en curso.\n\n"
                "¿Desea cerrar la aplicación de todas formas?\n"
                "Esto terminará todos los procesos.",
                icon='warning'
            )
            
            if result:
                self.force_quit()
        else:
            self.root.quit()
            self.root.destroy()

    def select_agent_operation(self, operation_type):
        """Seleccionar tipo de operación del agente y mostrar parámetros correspondientes"""
        self.current_operation_type = operation_type
        self.show_agent_params_for_operation(operation_type)
        
    def select_simulator_mode(self, mode):
        """Seleccionar modo del simulador y mostrar información correspondiente"""
        self.selected_sim_mode = mode
        self.show_simulator_params_for_mode(mode)
        
    def show_agent_params_for_operation(self, operation_type):
        """Mostrar parámetros específicos según el tipo de operación del agente"""
        # Limpiar frame de parámetros
        for widget in self.params_frame.winfo_children():
            widget.destroy()
            
        # Parámetros comunes
        common_params = [
            ("Número de ONTs:", self.num_ont, 1, 100),
            ("TxRate (bps):", self.tx_rate, 1e9, 100e9),
            ("Tiempo de Ciclo (s):", self.temp_ciclo, 0.001, 0.01),
            ("B Guaranteed (bps):", self.b_guaranteed, 100e6, 10e9),
            ("Carga:", self.load, 0.1, 1.0),
            ("Seed:", self.seed, 1, 1000)
        ]
        
        # Parámetros específicos según el tipo de operación
        if operation_type == 'train':
            specific_params = [
                ("Timesteps:", self.timesteps, 100000, 10000000),
                ("N° Entornos Paralelos:", self.num_envs, 1, 16)
            ]
            button_text = "Comenzar Entrenamiento"
            button_command = lambda: self.execute_agent('train')
            
        elif operation_type == 'eval':
            specific_params = [
                ("N° Ciclos:", self.n_ciclos, 100, 1000)
            ]
            button_text = "Comenzar Evaluación"
            button_command = lambda: self.execute_agent('eval')
            
        elif operation_type == 'both':
            specific_params = [
                ("Timesteps:", self.timesteps, 100000, 10000000),
                ("N° Entornos Paralelos:", self.num_envs, 1, 16),
                ("N° Ciclos:", self.n_ciclos, 100, 1000)
            ]
            button_text = "Comenzar Entrenamiento y Evaluación"
            button_command = lambda: self.execute_agent('both')
        
        # Combinar parámetros
        all_params = common_params + specific_params
        
        # Mostrar parámetros
        for i, (label, var, min_val, max_val) in enumerate(all_params):
            row = i // 2
            col = (i % 2) * 2
            
            ttk.Label(self.params_frame, text=label).grid(row=row, column=col, sticky=tk.W, padx=(0, 5))
            
            if isinstance(var, tk.StringVar):
                entry = ttk.Entry(self.params_frame, textvariable=var, width=15)
            else:
                entry = ttk.Spinbox(self.params_frame, from_=min_val, to=max_val, 
                                   textvariable=var, width=15)
            entry.grid(row=row, column=col+1, padx=(0, 20), pady=2)
        
        # Campo para nombre del modelo (siempre presente)
        next_row = (len(all_params) + 1) // 2
        ttk.Label(self.params_frame, text="Nombre del Modelo:").grid(row=next_row, column=0, sticky=tk.W, padx=(0, 5))
        entry_modelo = ttk.Entry(self.params_frame, textvariable=self.nombre_modelo, width=15)
        entry_modelo.grid(row=next_row, column=1, padx=(0, 20), pady=2)
        
        # Botón para ejecutar
        execute_button = ttk.Button(self.params_frame, text=button_text, 
                                   command=button_command, width=30)
        execute_button.grid(row=next_row+1, column=0, columnspan=4, pady=(20, 10))
        
    def show_simulator_params_for_mode(self, mode):
        """Mostrar parámetros específicos según el modo del simulador"""
        # Limpiar frame de parámetros
        for widget in self.params_frame.winfo_children():
            widget.destroy()
            
        mode_names = {"0": "Normal", "1": "DRL", "2": "DNN"}
        mode_name = mode_names.get(mode, mode)
        
        # Información del modo seleccionado
        info_text = f"""
Modo de simulación seleccionado: {mode_name}

Descripción:"""
        
        if mode == "0":
            info_text += "\n• Simulación estándar sin algoritmos de aprendizaje"
        elif mode == "1":
            info_text += "\n• Simulación con Deep Reinforcement Learning"
        elif mode == "2":
            info_text += "\n• Simulación con Deep Neural Network"
            
        info_label = ttk.Label(self.params_frame, text=info_text, justify=tk.LEFT)
        info_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 15))
        
        # Si es modo DRL, mostrar campo para nombre del modelo
        if mode == "1":
            ttk.Label(self.params_frame, text="Nombre del Modelo RL:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
            entry_modelo_rl = ttk.Entry(self.params_frame, textvariable=self.modelo_rl_nombre, width=25)
            entry_modelo_rl.grid(row=1, column=1, sticky=tk.W, padx=(0, 20), pady=5)
            
            # Información adicional sobre el modelo
            info_modelo = ttk.Label(self.params_frame, 
                                   text="Especifica el nombre del modelo PPO entrenado (sin extensión)", 
                                   font=("Arial", 8), foreground="gray")
            info_modelo.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(0, 15))
            
            button_row = 3
        else:
            button_row = 1
        
        # Botón para ejecutar simulación
        button_text = f"Comenzar Simulación ({mode_name})"
        execute_button = ttk.Button(self.params_frame, text=button_text, 
                                   command=lambda: self.execute_simulator(mode), width=30)
        execute_button.grid(row=button_row, column=0, columnspan=2, pady=(10, 0))

def main():
    """Función principal"""
    root = tk.Tk()
    app = MainApplication(root)
    
    # Mensaje de bienvenida
    app.log("Sistema iniciado. Seleccione un módulo para comenzar.")
    
    root.mainloop()

if __name__ == "__main__":
    main()