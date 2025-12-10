import gymnasium as gym
import numpy
import random
from os import system, name
from time import sleep

# Define function to clear console window.
def clear(): 
  
    # Clear on Windows.
    if name == 'nt': 
        _ = system('cls')
  
    # Clear on Mac and Linux. (os.name is 'posix') 
    else: 
        _ = system('clear')
clear()
env = gym.make("Taxi-v3").env