from env import TicTacToe
import torch
import matplotlib.pyplot as plt
from utils import simulate_from_state, test_vs_random, learning_rate_decay
from model import Agent
import time
import os
import random

num_epochs = 500
num_iterations = 100 # how often to train using batch after each iteration
num_simulations = 100 # Number of simulations per tested action
buffer_size = 5000 # Number of total samples to store in buffer
buffer_warmup = 500 # begin training after this many samples
batch_size = 10
lr_init = 1e-3
lr_min = 1e-4
lr_decay_step = 10
out_dir = 'out'
resume_from = None # i.e. ckpt.pt, will search for file in out_dir, None means training from scratch

env = TicTacToe()
agent = Agent()
optim = torch.optim.Adam(agent.parameters(), lr=lr_init)
cross_entropy = torch.nn.CrossEntropyLoss()
lr = learning_rate_decay(lr_init, num_epochs, lr_decay_step, lr_min) # array of learning rates for each epoch
buffer = []

# If resume_from not none set model and optimizer from file
if resume_from is not None:
    agent.load_from_file(os.path.join(out_dir, resume_from))
    optim.load_state_dict(torch.load(os.path.join(out_dir, resume_from))['optimizer'])

state = env.init_state()
avg_rewards = []

t0 = time.time()

for i in range(num_epochs):
    print(f"Iteration: {i}/{num_epochs}, lr: {lr[i]}")

    # Learning rate decay
    for param_group in optim.param_groups:
        param_group['lr'] = lr[i]

    state = env.init_state()
    done = False
    num_steps = 0

    # Take random actions, simulate a ton of games then take another random action and so on, if done start over
    while True:
        if num_steps > 0:
            state, _, done = env.step(state, env.random_action(state))
            if done:
                break

        target_probs = simulate_from_state(state=state, num_simulations=num_simulations)

        # Model should always see state as if its player 1 turn
        buffer_state = torch.clone(state)

        # Save both normal sample and rotated sample to get more diversity and more samples I guess?
        buffer.insert(0, (buffer_state, target_probs))
        buffer.insert(0, (torch.rot90(buffer_state), torch.rot90(target_probs.view(3, 3)).reshape((1, 9))))

        if len(buffer) > buffer_size:
            buffer.pop()
            buffer.pop()

        num_steps += 1

    # If gathered enough data start optimizing
    if len(buffer) > buffer_warmup:
        for _ in range(num_iterations):
            random.shuffle(buffer)
            state_batch = torch.stack([s[0] for s in buffer[0:batch_size]], dim=0)
            target_probs = torch.stack([s[1][0] for s in buffer[0:batch_size]], dim=0)

            # Train
            optim.zero_grad()
            pred_probs = agent(state_batch)

            loss = cross_entropy(pred_probs, target_probs)
            loss.backward()
            optim.step()

    # Evaluation
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    reward = test_vs_random(agent, 500)
    avg_rewards.append(reward)
    # use hacked stability measuremnt of model probs when state is 0, min to max, if differs by much then unstable
    init_probs = agent(env.init_state().unsqueeze(0))
    prob_diff = torch.max(init_probs) - torch.min(init_probs)
    print(f"- Duration: {dt:.2f}s, avg reward: {reward:.2f}, prob diff: {prob_diff:.2f}")

    checkpoint = {
        'model': agent.state_dict(),
        'optimizer': optim.state_dict(),
        'num_epoch': i,
        'num_simulations': num_simulations,
        'avg_reward': reward
    }
    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

plt.plot(avg_rewards)
plt.show()
