import argparse
import agente.custom_env.redes_opticas_env
from agente.classes.local_agent import LocalAgent
import sys
import tkinter as tk

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="Entrenamiento y evaluación del agente RL para PON")
        parser.add_argument('--tipo', type=str, required=True, choices=['train', 'eval', 'both'], help='Tipo de operación: train, eval o both')
        parser.add_argument('--num_ont', type=int, default=16)
        parser.add_argument('--TxRate', type=float, default=10e9)
        parser.add_argument('--temp_ciclo', type=float, default=0.002)
        parser.add_argument('--B_guaranteed', type=float, default=800e6)
        parser.add_argument('--env_id', type=str, default='RedesOpticasEnv-v0')
        parser.add_argument('--num_envs', type=int, default=8)
        parser.add_argument('--algorithm', type=str, default='ppo')
        parser.add_argument('--load', type=float, default=0.4)
        parser.add_argument('--timesteps', type=int, default=2000000)
        parser.add_argument('--nombre_modelo', type=str, default='modelo_prueba')
        parser.add_argument('--n_ciclos', type=int, default=300)
        parser.add_argument('--seed', type=int, default=42)
        args = parser.parse_args()

    # Parámetros comunes
    num_ont = args.num_ont
    TxRate = args.TxRate
    temp_ciclo = args.temp_ciclo
    B_guaranteed = args.B_guaranteed
    env_id = args.env_id
    num_envs = args.num_envs
    algorithm = args.algorithm
    load = args.load
    num_timesteps = args.timesteps
    nombre_modelo = args.nombre_modelo
    n_ciclos = args.n_ciclos
    seed = args.seed
    num_tests = 1  # Puedes parametrizarlo si lo necesitas

    if args.tipo == 'both':
        # TRAIN AND TEST
        agent = LocalAgent(num_ont, TxRate, temp_ciclo, B_guaranteed, seed)
        agent.create_model(env_id, num_envs, algorithm, n_ciclos, load)
        agent.train_model(num_timesteps)
        agent.save_model(nombre_modelo)
        agent.exec_simulation(n_ciclos, num_tests, env_id, load)
        agent.save_results(False, f"test-{nombre_modelo}.csv")
    elif args.tipo == 'train':
        # SOLO TRAIN
        agent = LocalAgent(num_ont, TxRate, temp_ciclo, B_guaranteed, seed)
        agent.create_model(env_id, num_envs, algorithm, n_ciclos, load)
        agent.train_model(num_timesteps)
        agent.save_model(nombre_modelo)
    elif args.tipo == 'eval':
        # SOLO TEST
        agent = LocalAgent(num_ont, TxRate, temp_ciclo, B_guaranteed, seed)
        agent.load_model(nombre_modelo, algorithm)
        agent.exec_simulation(n_ciclos, num_tests, env_id, load)
        agent.save_results(False, f"test-{nombre_modelo}.csv")
    else:
        print(f"Tipo de operación no reconocido: {args.tipo}")
        sys.exit(1)

if __name__ == "__main__":
    main()