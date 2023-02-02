from env import TicTacToe
from utils import play_human, simulate_from_state
from model import Agent
import torch

# Just testing some manual states

env = TicTacToe()
a = Agent()

# Uncomment to load model from a checkpoint
a.load_from_file('out/ckpt.pt')

# Uncomment to play against model policy. Currently human is always player one, see utils.py.
# play_human(a)

state = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32).view(3, 3).unsqueeze(0)
print(state)
y = a(state)
print(y)
print("-------------------\n")

state = torch.tensor([1, 1, 0, -1, -1, 0, 0, 0, 0], dtype=torch.int32).view(3, 3).unsqueeze(0)
print(state)
y = a(state)
print(y)
print("-------------------\n")

state = torch.tensor([1, -1, 0, 1, -1, 0, 0, 0, 0], dtype=torch.int32).view(3, 3).unsqueeze(0)
print(state)
y = a(state)
print(y)
print("-------------------\n")

state = torch.tensor([-1, -1, 0, 1, 0, 0, 1, 0, 0], dtype=torch.int32).view(3, 3).unsqueeze(0)
print(state)
y = a(state)
print(y)
print("-------------------\n")

state = torch.tensor([[ -1,  -1, 1], [1,  -1,  -1], [0,  0, 1]], dtype=torch.int32).unsqueeze(0)
print(state)
y = a(state)
print(y)
print("-------------------\n")

state = torch.tensor([[-1, 0, -1,], [0, 0, 0], [1, 0, 0]], dtype=torch.int32).unsqueeze(0)
print(state)
y = a(state)
print(y)
print("-------------------\n")

# Uncomment to test simulation
# print(simulate_from_state(state.squeeze(0) * -1, 500))
