# IA-Proyecto-RL: Q-Learning para Taxi-v3

Este proyecto implementa un agente de **Aprendizaje por Refuerzo** usando el algoritmo **Q-Learning** para resolver el entorno **Taxi-v3** de Gymnasium.

## Estructura del Proyecto

```
IA-Proyecto-RL/
├── agent/
│   ├── __init__.py
│   └── q_learning_agent.py    # Implementación del agente Q-Learning
├── enviroment/
│   └── env.py                 # Entorno básico (visualización aleatoria)
├── models/                    # Modelos entrenados guardados (se crea automáticamente)
├── results/                   # Gráficas y resultados (se crea automáticamente)
├── train.py                   # Script de entrenamiento
├── evaluate.py                # Script de evaluación y visualización
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Este archivo
```

## Instalación

### 1. Instalar dependencias

```powershell
pip install gymnasium pygame numpy matplotlib tqdm
```

O usando el archivo de requisitos:

```powershell
pip install -r requirements.txt
```

## Uso

### 1. Entrenar el agente

Entrena un nuevo agente Q-Learning:

```powershell
python train.py
```

Esto generará:
- Un modelo entrenado en `models/q_learning_taxi.pkl`
- Gráficas de entrenamiento en `results/training_results.png`

### 2. Evaluar el agente

Evalúa el rendimiento del agente entrenado:

```powershell
python evaluate.py
```

**Opciones disponibles:**

```powershell
# Evaluar con 200 episodios
python evaluate.py --episodes 200

# Visualizar el agente en acción
python evaluate.py --visualize

# Comparar con política aleatoria
python evaluate.py --compare

# Visualizar 3 episodios con delay de 0.5 segundos
python evaluate.py --visualize --visual-episodes 3 --delay 0.5

# Usar modelo específico
python evaluate.py --model models/mi_modelo.pkl
```

### 3. Visualización simple (ambiente aleatorio)

Para ver el entorno con comportamiento aleatorio:

```powershell
python enviroment/env.py
```

## Sobre el Algoritmo Q-Learning

### Ecuación de actualización

```
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
```

Donde:
- **s**: Estado actual
- **a**: Acción tomada
- **r**: Recompensa recibida
- **s'**: Siguiente estado
- **α** (alpha): Tasa de aprendizaje (0.1)
- **γ** (gamma): Factor de descuento (0.95)

### Estrategia Epsilon-Greedy

El agente balancea **exploración** y **explotación**:
- Con probabilidad **ε**: acción aleatoria (exploración)
- Con probabilidad **1-ε**: mejor acción según Q-table (explotación)
- **ε** decae de 1.0 a 0.01 durante el entrenamiento

## Entorno Taxi-v3

### Descripción
- **Cuadrícula**: 5x5
- **Objetivo**: Recoger un pasajero y llevarlo a su destino
- **Acciones**: 6 posibles
  - 0: Mover hacia abajo
  - 1: Mover hacia arriba
  - 2: Mover hacia la derecha
  - 3: Mover hacia la izquierda
  - 4: Recoger pasajero
  - 5: Dejar pasajero

### Recompensas
- **+20**: Entregar pasajero en el destino correcto
- **-10**: Recoger/dejar pasajero incorrectamente
- **-1**: Por cada paso

### Estados
- **500 estados** posibles (25 posiciones × 5 pasajeros × 4 destinos)

## Métricas de Evaluación

El script de evaluación proporciona:
- **Recompensa promedio** y desviación estándar
- **Número de pasos promedio**
- **Tasa de éxito** (% de episodios exitosos)
- **Comparación** con política aleatoria

## Personalización

### Modificar hiperparámetros

Edita en `train.py`:

```python
agent, rewards, steps = train_agent(
    n_episodes=10000,        # Número de episodios
    learning_rate=0.1,       # Tasa de aprendizaje (α)
    discount_factor=0.95,    # Factor de descuento (γ)
    epsilon=1.0,             # Exploración inicial
    epsilon_min=0.01,        # Exploración final
    epsilon_decay=0.995      # Velocidad de decaimiento
)
```

## Resultados Esperados

Después del entrenamiento:
- **Recompensa promedio**: ~7-9 (últimos 1000 episodios)
- **Pasos promedio**: ~13-15
- **Tasa de éxito**: >95%

## Dependencias

- `gymnasium>=0.29.0` - Entornos de RL
- `pygame>=2.5.0` - Renderizado gráfico
- `numpy>=1.24.0` - Operaciones numéricas
- `matplotlib>=3.7.0` - Visualización de gráficas
- `tqdm>=4.65.0` - Barras de progreso

## Notas

- El entrenamiento completo toma ~5-10 minutos
- Los modelos guardados incluyen la Q-table y todos los parámetros
- Las gráficas se actualizan automáticamente después del entrenamiento
- Para mejores resultados, entrena con al menos 10,000 episodios

## Contribuciones

Este proyecto es parte de un trabajo académico de Inteligencia Artificial sobre Aprendizaje por Refuerzo.

## Referencias

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Taxi-v3 Environment](https://gymnasium.farama.org/environments/toy_text/taxi/)
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.
