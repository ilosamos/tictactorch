from env import TicTacToe
from random import random
import numpy as np
import torch
import torch.nn.functional as F


def simulate_from_state(state, num_simulations, epsilon=1.0, model=None):
    env = TicTacToe()
    target_probs = torch.zeros(9)
    valid_actions = TicTacToe.valid_actions(state)
    mask = torch.zeros(9)
    mask[valid_actions] = 1
    player = env.get_turn(state)

    for action in valid_actions:
        next_state, r, d = env.step(state, action)
        avg_r = 0

        # special case if only one action left
        if d:
            # if draw, set reward without simulation since its only action left
            if env.is_full(next_state):
                target_probs[action] = r
                continue

            # if there is an immediate win ridiculously scale up reward to basically ignore other actions
            target_probs[action] += num_simulations * r
            continue

        for _ in range(num_simulations):
            x, r, p = simulate_game(state=next_state, epsilon=epsilon, model=model)
            if r == env.r_draw or p == player:
                avg_r += r
            else:
                avg_r += r * -1

        avg_r /= num_simulations
        target_probs[action] += avg_r

    target_probs /= valid_actions.size(dim=0)
    target_probs[target_probs == 0] = float("-Inf") # Mask minus infinity for softmax to be 0
    return F.softmax(target_probs, dim=-1).unsqueeze(0) # Add batch dimension


def simulate_game(epsilon=1.0, model=None, state=None):
    env = TicTacToe()
    state = env.init_state() if state is None else torch.clone(state)
    done = False
    r = 0
    player = 0

    seq = [state]

    while not done:
        p = random()
        if p < epsilon or model is None:
            action = env.random_action(state)
        else:
            action = model.sample(state)
        player = env.get_turn(state)
        state, r, done = env.step(state, action)
        seq.append(state)

    return seq, r, player if r != env.r_draw else 0


def test_vs_random(model, total_games):
    # Test as player 1 for now
    rewards = []
    agent_player = 1
    env = TicTacToe()

    for i in range(total_games):
        state = env.init_state()

        while True:
            player = env.get_turn(state)
            if player == agent_player:
                action = model.sample(state)
                state, r, done = env.step(state, action)
            else:
                action = env.random_action(state)
                state, r, done = env.step(state, action)

            if done:
                if r == env.r_draw or player == agent_player:
                    rewards.append(r)
                else:
                    rewards.append(r * -1)
                break

        # Test model for both players
        agent_player *= -1

    return np.mean(rewards)


# written by chatGPT lol
def learning_rate_decay(lr_initial, epochs, lr_step_size, lr_min):
    lr = lr_initial
    lr_list = []
    for i in range(epochs):
        if (i+1) % lr_step_size == 0:
            lr = max(lr*0.9, lr_min)
        lr_list.append(lr)
    return np.array(lr_list)


def play_human(model):
    env = TicTacToe()
    state = env.init_state()
    done = False
    r = 0

    while not done:
        env.render(state)
        action = int(input('Enter action: '))
        state, r, done = env.step(state, action)

        if done:
            break

        # if agent is player 2 flip state
        action = model.sample(state)
        state, r, done = env.step(state, action)

    env.render(state)
    print('Reward: ', r)
