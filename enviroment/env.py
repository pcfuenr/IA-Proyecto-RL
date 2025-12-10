import gymnasium as gym
import pygame  # Necesario para detectar el clic en la X
import time

# 1. Crear entorno
env = gym.make("Taxi-v3", render_mode="human")
observation, info = env.reset()
ejecutando = True

while ejecutando:
    if env.unwrapped.render_mode == "human":
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                ejecutando = False
                break
    
    if not ejecutando:
        break

    action = env.action_space.sample()  # Acción aleatoria
    observation, reward, terminated, truncated, info = env.step(action)
    
    time.sleep(0.3)

    # Reiniciar si termina el episodio
    if terminated or truncated:
        observation, info = env.reset()

env.close()
pygame.quit()