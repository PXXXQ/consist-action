import gym
import torch
import torch.nn as nn
import numpy as np
from models.mlp_policy_disc import DiscretePolicy


class Agent_Disc(nn.Module):
    def __init__(self, state_dim, action_num, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        self.activation = torch.tanh
        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        value_hidden_num = int(last_dim / 8)
        action_hidden_num = int(last_dim / 8)

        self.value_hidden = nn.Linear(last_dim, value_hidden_num)
        self.action_hidden = nn.Linear(last_dim, action_hidden_num)

        self.value_head = nn.Linear(value_hidden_num, 1)
        self.action_head = nn.Linear(action_hidden_num, action_num)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_hidden = self.activation(self.action_hidden(x))
        value_hidden = self.activation(self.action_hidden(x))

        action_prob = torch.softmax(self.action_head(action_hidden), dim=1)
        value_hidden = self.activation(self.value_head(value_hidden))

        return action_prob, value_hidden

    def select_action(self, x):
        action_prob, _ = self.forward(x)
        action = action_prob.multinomial(1)
        return action

    def get_log_prob(self, x, actions):
        action_prob, _ = self.forward(x)
        actions = actions.long()
        actions = actions.unsqueeze(1)

        return torch.log(action_prob.gather(1, actions))

    def estimate_advantage(self):
        pass


if __name__ == "__main__":

    dtype = torch.float64
    torch.set_default_dtype(dtype)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    env = gym.make("MountainCar-v0")

    action_dim = 3
    state_dim = 2

    Agent = Agent_Disc(state_dim, action_dim, hidden_size=(128, 128))
    # Agent = DiscretePolicy(action_dim, state_dim, hidden_size=(128, 128))

    state = env.reset()
    memory = dict(state=[], action=[], reward=[], next_state=[], done=[])

    while True:

        action_tensor = Agent.select_action(torch.tensor([state]))
        action = action_tensor[0].tolist()[0]
        observation, reward, done, _ = env.step(action)
        print("action: ", action_tensor[0].tolist()[0], "reward: ", reward)

        memory["state"].append(state)
        memory["action"].append(action)
        memory["reward"].append(reward)
        memory["next_state"].append(observation)
        memory["done"].append(0 if done else 1)

        state = observation

        # env.render()
        if done:
            state = env.reset()
            memory_len = len(memory["state"])
            if memory_len > 320:
                print(memory_len)
                break

    # memory to batch
    states = torch.from_numpy(np.array(memory["state"])).to(dtype).to(device)
    actions = torch.from_numpy(np.array(memory["action"])).to(dtype).to(device)
    rewards = torch.from_numpy(np.array(memory["reward"])).to(dtype).to(device)
    masks = torch.from_numpy(np.array(memory["next_state"])).to(dtype).to(device)
    targets = torch.from_numpy(np.array(memory["done"])).to(dtype).to(device)

    Agent.to(device)
    action_probs = Agent.get_log_prob(states, actions)

    print("AAA")





