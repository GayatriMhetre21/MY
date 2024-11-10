import numpy as np
import random

# Define constants for the environment
GRID_SIZE = 5
PASSENGER_LOCATIONS = [(0, 0), (0, 4), (4, 0), (4, 4)]
DESTINATION_LOCATIONS = [(0, 4), (0, 0), (4, 4), (4, 0)]
ACTIONS = ['N', 'S', 'E', 'W', 'PICKUP', 'DROPOFF']

class TaxiEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.passenger_locations = PASSENGER_LOCATIONS
        self.destination_locations = DESTINATION_LOCATIONS
        self.reset()

    def reset(self):
        # Randomly place taxi and passenger in the grid
        self.taxi_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
        self.passenger_pos = random.choice(self.passenger_locations)
        self.destination_pos = random.choice(self.destination_locations)
        self.passenger_on_board = False
        return self._get_state()

    def _get_state(self):
        return (self.taxi_pos[0], self.taxi_pos[1], 
                self.passenger_pos[0], self.passenger_pos[1], 
                int(self.passenger_on_board))

    def step(self, action):
        if action == 'N':
            new_pos = (max(self.taxi_pos[0] - 1, 0), self.taxi_pos[1])
        elif action == 'S':
            new_pos = (min(self.taxi_pos[0] + 1, self.grid_size - 1), self.taxi_pos[1])
        elif action == 'E':
            new_pos = (self.taxi_pos[0], min(self.taxi_pos[1] + 1, self.grid_size - 1))
        elif action == 'W':
            new_pos = (self.taxi_pos[0], max(self.taxi_pos[1] - 1, 0))
        else:
            new_pos = self.taxi_pos
        
        # Check if taxi picks up or drops off a passenger
        reward = -1  # Small negative reward for each step taken
        
        if action == 'PICKUP':
            if new_pos == self.passenger_pos and not self.passenger_on_board:
                reward += 20  # Reward for picking up the passenger
                self.passenger_on_board = True
                # Randomly choose a new destination for the passenger
                self.destination_pos = random.choice(self.destination_locations)
        
        elif action == 'DROPOFF':
            if new_pos == self.destination_pos and self.passenger_on_board:
                reward += 20  # Reward for dropping off the passenger
                self.passenger_on_board = False
                # Reset passenger position to pick up again in next episode
                self.passenger_pos = random.choice(self.passenger_locations)

        # Update taxi position only if not picking up or dropping off
        if action not in ['PICKUP', 'DROPOFF']:
            self.taxi_pos = new_pos
        
        return self._get_state(), reward

class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, GRID_SIZE, 2, len(ACTIONS)))  # State-action values
        self.alpha = 0.1   # Learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon = 0.2 # Exploration rate

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(ACTIONS)  # Explore: choose random action
        else:
            return ACTIONS[np.argmax(self.q_table[state])]

    def learn(self, state, action, reward, next_state):
        action_index = ACTIONS.index(action)
        
        best_next_action_index = np.argmax(self.q_table[next_state])
        
        # Update Q-value using the Bellman equation
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action_index]
        td_delta = td_target - self.q_table[state][action_index]
        
        # Update Q-value for current state-action pair
        self.q_table[state][action_index] += self.alpha * td_delta

def train_agent(env, agent, episodes):
    for episode in range(episodes):
        state = env.reset()
        
        while True:
            action = agent.choose_action(state)
            next_state, reward = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            
            if reward > 0:  # End of episode when we get a positive reward for drop-off or pickup
                break

# Initialize environment and agent
taxi_env = TaxiEnv()
agent = QLearningAgent(taxi_env)

# Train the agent for a number of episodes
train_agent(taxi_env, agent, episodes=10000)

# Test the trained agent by visualizing its actions in the environment
def test_agent(env, agent):
    state = env.reset()
    total_reward = 0
    
    while True:
        action_index = np.argmax(agent.q_table[state])
        action = ACTIONS[action_index]
        
        next_state, reward = env.step(action)
        
        print(f'Taxi Position: {env.taxi_pos}, Passenger Position: {env.passenger_pos}, '
              f'Destination Position: {env.destination_pos}, Action Taken: {action}, Reward: {reward}')
        
        total_reward += reward
        
        state = next_state
        
        if reward > 0:  # End of episode when we get a positive reward for drop-off or pickup
            break
            
    print(f'Total Reward: {total_reward}')

# Test the trained agent after training completion
test_agent(taxi_env, agent)


