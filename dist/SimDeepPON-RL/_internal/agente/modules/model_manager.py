from pathlib import Path
from stable_baselines3 import PPO, DQN, A2C, SAC, TD3

import os

# Constantes para el guardado de modelos
SUBDIR = "models"
MODELS = { # Algoritmos que tiene StableBaselines3
    "ppo": PPO,
    "dqn": DQN,
    "a2c": A2C,
    "sac": SAC,
    "td3": TD3
}

# Constantes para la creacion de modelos
POLICY = "MlpPolicy"
ENABLE_VERBOSE = 2
N_STEPS = 512  # Steps por actualización
BATCH_SIZE = 32  # Tamaño del mini-batch (16384 es múltiplo de 256)
LEARNING_RATE = 1e-3
GAMMA = 0.95
GAE_LAMBDA = 0.90

# Metodo que devuelve la ruta de la raiz del agente, para buscar los ficheros donde guardar o cargar los modelos
def get_project_root():
    current_dir = Path(__file__).resolve().parent
    return current_dir.parent

# Metodo para almacenar el modelo ya entrenado en un fichero, dentro del directorio especificado en las consstantes
def save_model(model, filename):
    models_dir = os.path.join(get_project_root(), SUBDIR)
    os.makedirs(models_dir, exist_ok=True)
    full_path = os.path.join(models_dir, filename)
    model.save(full_path)

    return os.path.exists(full_path)


# Metodo para cargar un modelo ya entrenado desde un fichero
# Necesita recibir el tipo de algoritmo para llamar a su metodo load()
# Necesita recibir un vector de entornos con 
def load_model(filename, algorithm):

    algorithm = algorithm.lower()
    if algorithm not in MODELS:
        raise ValueError(f"Tipo de modelo no soportado: {algorithm}")
    else:
        model_class = MODELS[algorithm]

    full_path = os.path.join(get_project_root(), SUBDIR, filename)
    model = model_class.load(full_path)
    
    return model


def create_model(vec_env, algorithm):
    
    algorithm = algorithm.lower()

    model = MODELS[algorithm](
        POLICY,
        vec_env,
        verbose=ENABLE_VERBOSE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA
    )

    return model