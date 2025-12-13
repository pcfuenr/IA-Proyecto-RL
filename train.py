import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from agent.q_learning_agent import QLearningAgent


def train_agent(
    n_episodes: int = 10000,
    max_steps: int = 100,
    learning_rate: float = 0.1,
    discount_factor: float = 0.95,
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    save_model: bool = True,
    model_path: str = 'models/q_learning_taxi.pkl',
    verbose: bool = True
):
    """
    Entrena el agente Q-Learning
    
    Args:
        n_episodes: Número de episodios de entrenamiento
        max_steps: Máximo de pasos por episodio
        learning_rate: Tasa de aprendizaje (alpha)
        discount_factor: Factor de descuento (gamma)
        epsilon: Probabilidad inicial de exploración
        epsilon_min: Valor mínimo de epsilon
        epsilon_decay: Tasa de decaimiento de epsilon
        save_model: Si se debe guardar el modelo
        model_path: Ruta donde guardar el modelo
        verbose: Mostrar información durante el entrenamiento
    """
    
    # Crear entorno
    env = gym.make('Taxi-v3')
    
    # Crear agente
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay
    )
    
    # Métricas de entrenamiento
    rewards_per_episode = []
    steps_per_episode = []
    epsilon_history = []
    
    # Métricas para promedios móviles
    window_size = 100
    avg_rewards = []
    
    if verbose:
        print("=" * 60)
        print("ENTRENAMIENTO DEL AGENTE Q-LEARNING - TAXI-V3")
        print("=" * 60)
        print(f"Episodios: {n_episodes}")
        print(f"Learning Rate (α): {learning_rate}")
        print(f"Discount Factor (γ): {discount_factor}")
        print(f"Epsilon inicial: {epsilon}")
        print(f"Epsilon final: {epsilon_min}")
        print("=" * 60)
    
    # Barra de progreso
    progress_bar = tqdm(range(n_episodes), desc="Entrenando")
    
    for episode in progress_bar:
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Seleccionar acción
            action = agent.choose_action(state, training=True)
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Actualizar Q-table
            agent.update(state, action, reward, next_state, done)
            
            # Actualizar estado y métricas
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Decaer epsilon
        agent.decay_epsilon()
        
        # Guardar métricas
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        epsilon_history.append(agent.epsilon)
        
        # Calcular promedio móvil
        if len(rewards_per_episode) >= window_size:
            avg_reward = np.mean(rewards_per_episode[-window_size:])
            avg_rewards.append(avg_reward)
        
        # Actualizar barra de progreso
        if episode % 100 == 0 and episode > 0:
            avg_last_100 = np.mean(rewards_per_episode[-100:])
            progress_bar.set_postfix({
                'Avg Reward (100)': f'{avg_last_100:.2f}',
                'Epsilon': f'{agent.epsilon:.3f}'
            })
    
    env.close()
    
    # Mostrar estadísticas finales
    if verbose:
        print("\n" + "=" * 60)
        print("ESTADISTICAS DE ENTRENAMIENTO")
        print("=" * 60)
        print(f"Recompensa promedio (últimos 100): {np.mean(rewards_per_episode[-100:]):.2f}")
        print(f"Recompensa promedio (últimos 1000): {np.mean(rewards_per_episode[-1000:]):.2f}")
        print(f"Pasos promedio (últimos 100): {np.mean(steps_per_episode[-100:]):.2f}")
        print(f"Epsilon final: {agent.epsilon:.4f}")
        
        stats = agent.get_stats()
        print(f"\nQ-Table Stats:")
        print(f"  - Valores no-cero: {stats['non_zero_values']}/{agent.n_states * agent.n_actions}")
        print(f"  - Media: {stats['q_table_mean']:.3f}")
        print(f"  - Desviación estándar: {stats['q_table_std']:.3f}")
        print(f"  - Máximo: {stats['q_table_max']:.3f}")
        print(f"  - Mínimo: {stats['q_table_min']:.3f}")
        print("=" * 60)
    
    # Guardar modelo
    if save_model:
        agent.save(model_path)
    
    # Generar gráficas
    plot_training_results(
        rewards_per_episode,
        steps_per_episode,
        epsilon_history,
        window_size
    )
    
    return agent, rewards_per_episode, steps_per_episode


def plot_training_results(rewards, steps, epsilon_history, window_size=1):
    """Genera gráficas del entrenamiento"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Resultados del Entrenamiento - Q-Learning Taxi-v3', fontsize=14)
    
    # 1. Recompensas por episodio
    ax1 = axes[0, 0]
    ax1.plot(rewards, alpha=0.3, label='Recompensa')
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(rewards)), moving_avg, 
                label=f'Media móvil ({window_size})', linewidth=2)
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa Total')
    ax1.set_title('Recompensas durante el Entrenamiento')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Pasos por episodio
    ax2 = axes[0, 1]
    ax2.plot(steps, alpha=0.3, label='Pasos')
    if len(steps) >= window_size:
        moving_avg_steps = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(window_size-1, len(steps)), moving_avg_steps,
                label=f'Media móvil ({window_size})', linewidth=2)
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Número de Pasos')
    ax2.set_title('Pasos por Episodio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Epsilon decay
    ax3 = axes[1, 0]
    ax3.plot(epsilon_history, color='orange', linewidth=2)
    ax3.set_xlabel('Episodio')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Decaimiento de Epsilon (Exploración vs Explotación)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribución de recompensas
    ax4 = axes[1, 1]
    ax4.hist(rewards, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(np.mean(rewards), color='red', linestyle='--', 
                linewidth=2, label=f'Media: {np.mean(rewards):.2f}')
    ax4.set_xlabel('Recompensa')
    ax4.set_ylabel('Frecuencia')
    ax4.set_title('Distribución de Recompensas')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar gráfica
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_results.png', dpi=300, bbox_inches='tight')
    print(f"\nGraficas guardadas en: results/training_results.png")
    
    plt.show()


if __name__ == "__main__":
    # Configuración de entrenamiento
    agent, rewards, steps = train_agent(
        n_episodes=3000,
        max_steps=100,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        save_model=True,
        verbose=True
    )
    
    print("\nEntrenamiento completado!")
