"""
Script para evaluar y visualizar el agente entrenado
"""
import gymnasium as gym
import numpy as np
import pygame
import time
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from agent.q_learning_agent import QLearningAgent


def evaluate_agent(agent, n_episodes=100, render=False, verbose=True):
    """
    Evalúa el rendimiento del agente entrenado
    
    Args:
        agent: Agente Q-Learning entrenado
        n_episodes: Número de episodios de evaluación
        render: Si se debe visualizar el entorno
        verbose: Mostrar información detallada
        
    Returns:
        Diccionario con métricas de evaluación
    """
    render_mode = "human" if render else None
    env = gym.make('Taxi-v3', render_mode=render_mode)
    
    total_rewards = []
    total_steps = []
    success_count = 0
    
    if verbose:
        print("=" * 60)
        print("EVALUACION DEL AGENTE")
        print("=" * 60)
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            # Seleccionar mejor acción (sin exploración)
            action = agent.choose_action(state, training=False)
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if render:
                time.sleep(0.1)  # Pausa para visualización
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
        # Episodio exitoso si entregó al pasajero
        if episode_reward > -100:  # Umbral razonable de éxito
            success_count += 1
        
        if verbose and (episode + 1) % 20 == 0:
            print(f"Episodio {episode + 1}/{n_episodes} - "
                  f"Recompensa: {episode_reward}, Pasos: {steps}")
    
    env.close()
    
    # Calcular métricas
    metrics = {
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'avg_steps': np.mean(total_steps),
        'std_steps': np.std(total_steps),
        'success_rate': (success_count / n_episodes) * 100,
        'min_reward': np.min(total_rewards),
        'max_reward': np.max(total_rewards),
        'min_steps': np.min(total_steps),
        'max_steps': np.max(total_steps)
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("RESULTADOS DE EVALUACION")
        print("=" * 60)
        print(f"Episodios evaluados: {n_episodes}")
        print(f"Recompensa promedio: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Pasos promedio: {metrics['avg_steps']:.2f} ± {metrics['std_steps']:.2f}")
        print(f"Tasa de éxito: {metrics['success_rate']:.1f}%")
        print(f"Rango de recompensas: [{metrics['min_reward']}, {metrics['max_reward']}]")
        print(f"Rango de pasos: [{metrics['min_steps']}, {metrics['max_steps']}]")
        print("=" * 60)
    
    return metrics


def visualize_agent(agent, n_episodes=5, delay=0.3):
    """
    Visualiza el agente entrenado en acción
    
    Args:
        agent: Agente Q-Learning entrenado
        n_episodes: Número de episodios a visualizar
        delay: Tiempo de espera entre acciones (segundos)
    """
    env = gym.make('Taxi-v3', render_mode="human")
    
    print("=" * 60)
    print("VISUALIZACION DEL AGENTE ENTRENADO")
    print("=" * 60)
    print("Acciones:")
    print("  0: Mover hacia abajo")
    print("  1: Mover hacia arriba")
    print("  2: Mover hacia la derecha")
    print("  3: Mover hacia la izquierda")
    print("  4: Recoger pasajero")
    print("  5: Dejar pasajero")
    print("=" * 60)
    print("Presiona la 'X' de la ventana para cerrar\n")
    
    action_names = {
        0: "Abajo",
        1: "Arriba",
        2: "Derecha",
        3: "Izquierda",
        4: "Recoger",
        5: "Dejar"
    }
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\nEpisodio {episode + 1}/{n_episodes}")
        print("-" * 40)
        
        ejecutando = True
        
        while ejecutando:
            # Detectar cierre de ventana
            if env.unwrapped.render_mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        ejecutando = False
                        break
            
            if not ejecutando:
                break
            
            # Seleccionar mejor acción
            action = agent.choose_action(state, training=False)
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            print(f"  Paso {steps}: {action_names[action]} | "
                  f"Recompensa: {reward:+3.0f} | "
                  f"Total: {episode_reward:+4.0f}")
            
            time.sleep(delay)
            
            if done:
                if episode_reward > 0:
                    print(f"\nExito! Pasajero entregado en {steps} pasos")
                else:
                    print(f"\nEpisodio terminado. Recompensa: {episode_reward}")
                print(f"   Recompensa total: {episode_reward}")
                print(f"   Pasos totales: {steps}")
                break
        
        if not ejecutando:
            print("\nVisualizacion interrumpida por el usuario")
            break
        
        if episode < n_episodes - 1:
            print("\nPresiona Ctrl+C para detener, o espera 2 segundos...")
            time.sleep(2)
    
    env.close()
    pygame.quit()
    print("\nVisualizacion completada")


def compare_with_random(agent, n_episodes=100):
    """
    Compara el agente entrenado con una política aleatoria
    
    Args:
        agent: Agente Q-Learning entrenado
        n_episodes: Número de episodios para comparación
    """
    print("=" * 60)
    print("COMPARACION: AGENTE ENTRENADO vs POLITICA ALEATORIA")
    print("=" * 60)
    
    # Evaluar agente entrenado
    print("\n1. Evaluando agente entrenado...")
    env = gym.make('Taxi-v3')
    trained_rewards = []
    trained_steps = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            action = agent.choose_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        trained_rewards.append(episode_reward)
        trained_steps.append(steps)
    
    # Evaluar política aleatoria
    print("2. Evaluando politica aleatoria...")
    random_rewards = []
    random_steps = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        max_steps = 100
        
        for _ in range(max_steps):
            action = env.action_space.sample()  # Acción aleatoria
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        random_rewards.append(episode_reward)
        random_steps.append(steps)
    
    env.close()
    
    # Mostrar comparación
    print("\n" + "=" * 60)
    print("RESULTADOS DE LA COMPARACION")
    print("=" * 60)
    print(f"{'Métrica':<30} {'Entrenado':<15} {'Aleatorio':<15} {'Mejora'}")
    print("-" * 60)
    
    trained_avg_reward = np.mean(trained_rewards)
    random_avg_reward = np.mean(random_rewards)
    reward_improvement = ((trained_avg_reward - random_avg_reward) / abs(random_avg_reward)) * 100
    
    trained_avg_steps = np.mean(trained_steps)
    random_avg_steps = np.mean(random_steps)
    steps_improvement = ((random_avg_steps - trained_avg_steps) / random_avg_steps) * 100
    
    print(f"{'Recompensa promedio':<30} {trained_avg_reward:<15.2f} {random_avg_reward:<15.2f} {reward_improvement:+.1f}%")
    print(f"{'Pasos promedio':<30} {trained_avg_steps:<15.2f} {random_avg_steps:<15.2f} {steps_improvement:+.1f}%")
    print(f"{'Desv. estándar recompensa':<30} {np.std(trained_rewards):<15.2f} {np.std(random_rewards):<15.2f}")
    print(f"{'Desv. estándar pasos':<30} {np.std(trained_steps):<15.2f} {np.std(random_steps):<15.2f}")
    print("=" * 60)
    
    return {
        'trained': {'rewards': trained_rewards, 'steps': trained_steps},
        'random': {'rewards': random_rewards, 'steps': random_steps}
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluar agente Q-Learning')
    parser.add_argument('--model', type=str, default='models/q_learning_taxi.pkl',
                       help='Ruta del modelo guardado')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Número de episodios para evaluación')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualizar el agente en acción')
    parser.add_argument('--compare', action='store_true',
                       help='Comparar con política aleatoria')
    parser.add_argument('--visual-episodes', type=int, default=5,
                       help='Número de episodios a visualizar')
    parser.add_argument('--delay', type=float, default=0.3,
                       help='Delay entre acciones en visualización (segundos)')
    
    args = parser.parse_args()
    
    # Cargar agente
    print("Cargando modelo...")
    try:
        agent = QLearningAgent.load(args.model)
    except FileNotFoundError:
        print(f"Error: No se encontro el modelo en {args.model}")
        print("   Ejecuta primero: python train.py")
        sys.exit(1)
    
    # Evaluación
    if not args.visualize:
        evaluate_agent(agent, n_episodes=args.episodes, verbose=True)
    
    # Comparación
    if args.compare:
        compare_with_random(agent, n_episodes=args.episodes)
    
    # Visualización
    if args.visualize:
        visualize_agent(agent, n_episodes=args.visual_episodes, delay=args.delay)
