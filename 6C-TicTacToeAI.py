import numpy as np
import random

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3))  # 0 for empty, 1 for X, -1 for O
        self.current_player = 1  # Player X starts

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1
        return self.board.flatten()

    def available_actions(self):
        return np.argwhere(self.board.flatten() == 0).flatten()

    def make_move(self, action):
        if self.board.flatten()[action] == 0:
            self.board.flatten()[action] = self.current_player
            self.current_player *= -1  # Switch player
            return True
        return False

    def check_winner(self):
        for i in range(3):
            if abs(np.sum(self.board[i, :])) == 3:  # Check rows
                return np.sign(np.sum(self.board[i, :]))
            if abs(np.sum(self.board[:, i])) == 3:  # Check columns
                return np.sign(np.sum(self.board[:, i]))
        if abs(np.sum(np.diag(self.board))) == 3:  # Check diagonal
            return np.sign(np.sum(np.diag(self.board)))
        if abs(np.sum(np.diag(np.fliplr(self.board)))) == 3:  # Check anti-diagonal
            return np.sign(np.sum(np.diag(np.fliplr(self.board))))
        if len(self.available_actions()) == 0:  # Draw condition
            return 0
        return None

class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.1   # Learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon = 0.2 # Exploration rate

    def get_state_key(self, board):
        return str(board)

    def choose_action(self, board):
        state_key = self.get_state_key(board)
        
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(board.available_actions()))  # Explore
        else:
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(9)  # Initialize Q-values for new state
            
            available_actions = board.available_actions()
            return available_actions[np.argmax(self.q_table[state_key][available_actions])]  # Exploit

    def learn(self, board, action, reward, next_board):
        state_key = self.get_state_key(board)
        next_state_key = self.get_state_key(next_board)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(9)  # Initialize Q-values for new state
        
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(9)  # Initialize Q-values for new state
        
        best_next_action_value = np.max(self.q_table[next_state_key])
        
        td_target = reward + self.gamma * best_next_action_value
        td_delta = td_target - self.q_table[state_key][action]
        
        # Update Q-value for current state-action pair
        self.q_table[state_key][action] += self.alpha * td_delta

def train_agent(agent, episodes):
    for _ in range(episodes):
        board = TicTacToe()
        
        while True:
            action = agent.choose_action(board)
            board.make_move(action)
            
            winner = board.check_winner()
            if winner is not None: 
                reward = winner if winner != -1 else -0.5   # Reward for winning or draw (for O)
                agent.learn(board.board.flatten(), action, reward, board.board.flatten())
                break
            
            next_action = random.choice(board.available_actions())   # Random move for opponent (O)
            board.make_move(next_action)
            
            winner = board.check_winner()
            if winner is not None:
                reward = winner if winner != 1 else -0.5   # Reward for winning or draw (for X)
                agent.learn(board.board.flatten(), action, reward, board.board.flatten())
                break
            
            agent.learn(board.board.flatten(), action, -0.1, board.board.flatten())   # Small penalty for each step

def test_agent(agent):
    board = TicTacToe()
    
    while True:
        print("Current Board:")
        print(board.board)
        
        action = agent.choose_action(board)
        print(f"Agent chooses action: {action}")
        
        board.make_move(action)
        
        winner = board.check_winner()
        if winner is not None:
            print("Final Board:")
            print(board.board)
            
            if winner == 1:
                print("Agent wins!")
            elif winner == -1:
                print("Opponent wins!")
            else:
                print("It's a draw!")
            break
        
        next_action = random.choice(board.available_actions())   # Random move for opponent (O)
        print(f"Opponent chooses action: {next_action}")
        
        board.make_move(next_action)

# Initialize the agent and train it
agent = QLearningAgent()
train_agent(agent, episodes=5000)

# Test the trained agent against a random opponent
test_agent(agent)
