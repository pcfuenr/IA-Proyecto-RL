"""
Agente Q-Learning para el entorno Taxi-v3
Este agente aprende la política óptima mediante el algoritmo Q-Learning
"""
import numpy as np
import pickle
import os


class QLearningAgent:
    """
    Agente que aprende usando Q-Learning
    
    Parámetros:
        n_states: Número de estados en el entorno
        n_actions: Número de acciones posibles
        learning_rate: Tasa de aprendizaje (alpha)
        discount_factor: Factor de descuento (gamma)
        epsilon: Probabilidad de exploración inicial
        epsilon_min: Valor mínimo de epsilon
        epsilon_decay: Tasa de decaimiento de epsilon
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate  # alpha
        self.discount_factor = discount_factor  # gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Inicializar Q-table con ceros
        self.q_table = np.zeros((n_states, n_actions))
        
    def choose_action(self, state: int, training: bool = True) -> int:
        """
        Selecciona una acción usando política epsilon-greedy
        
        Args:
            state: Estado actual
            training: Si es True, usa exploración; si es False, solo explotación
            
        Returns:
            Acción seleccionada
        """
        if training and np.random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return np.random.randint(self.n_actions)
        else:
            # Explotación: mejor acción según Q-table
            return np.argmax(self.q_table[state])
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ):
        """
        Actualiza la Q-table usando la ecuación de Q-Learning
        
        Q(s,a) = Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: Estado actual
            action: Acción tomada
            reward: Recompensa recibida
            next_state: Siguiente estado
            done: Si el episodio terminó
        """
        # Valor actual de Q(s,a)
        current_q = self.q_table[state, action]
        
        # Mejor valor Q en el siguiente estado
        if done:
            max_next_q = 0  # No hay estado futuro
        else:
            max_next_q = np.max(self.q_table[next_state])
        
        # Actualización Q-Learning
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state, action] = new_q
    
    def decay_epsilon(self):
        """Reduce epsilon después de cada episodio"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Guarda el modelo (Q-table y parámetros)"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'n_states': self.n_states,
            'n_actions': self.n_actions
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Modelo guardado en: {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Carga un modelo guardado"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Crear agente con los parámetros guardados
        agent = cls(
            n_states=model_data['n_states'],
            n_actions=model_data['n_actions'],
            learning_rate=model_data['learning_rate'],
            discount_factor=model_data['discount_factor'],
            epsilon=model_data['epsilon'],
            epsilon_min=model_data['epsilon_min'],
            epsilon_decay=model_data['epsilon_decay']
        )
        
        # Restaurar Q-table
        agent.q_table = model_data['q_table']
        
        print(f"Modelo cargado desde: {filepath}")
        return agent
    
    def get_stats(self) -> dict:
        """Retorna estadísticas del agente"""
        return {
            'epsilon': self.epsilon,
            'q_table_mean': np.mean(self.q_table),
            'q_table_std': np.std(self.q_table),
            'q_table_max': np.max(self.q_table),
            'q_table_min': np.min(self.q_table),
            'non_zero_values': np.count_nonzero(self.q_table)
        }
