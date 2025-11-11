class GridMDP:
    def __init__(self, rows=3, cols=4, block=(2, 2), terminal_states={(4, 3): 1, (4, 2): -1}, prob_intended = 0.8, prob_perpendicular = 0.1, default_reward=-0.04,gamma = 0.9):
        self.rows = rows
        self.cols = cols
        self.block = block
        self.terminal_states = terminal_states
        self.default_reward = default_reward
        self.states = [(i, j) for j in range(1, rows + 1) for i in range(1, cols + 1) if (i, j) != block]        
        self.actions = ['U', 'D', 'L', 'R']
        self.prob_intended = prob_intended
        self.prob_perpendicular = prob_perpendicular
        self.transition_probabilities = self.build_transition_probabilities()
        self.rewards = self.build_rewards()
        self.gamma = gamma

    def build_transition_probabilities(self):
        transition_probabilities = {}
        for state in self.states:
            # In terminal state, any action leads to the same state with probability 1
            if state in self.terminal_states:
                transition_probabilities[state] = {action: [(state,1.0)] for action in self.actions}
            else:
                transition_probabilities[state] = {}
                for action in self.actions:
                    transition_probabilities[state][action] = self.calculate_transition_probabilities(state, action)
        return transition_probabilities
    
    def calculate_transition_probabilities(self, state, action):
        i, j = state
        probs = []
        def is_valid(state):
            x, y = state
            return 1 <= x <= self.cols and 1 <= y <= self.rows and state != self.block
        move = {
            'U': (i, j + 1), 'D': (i, j - 1),
            'L': (i - 1, j), 'R': (i + 1, j)
        }
        perpendicular_moves = {
            'U': [(i - 1, j), (i + 1, j)],  # Left, Right
            'D': [(i - 1, j), (i + 1, j)],  # Left, Right
            'L': [(i, j + 1), (i, j - 1)],  # Up, Down
            'R': [(i, j + 1), (i, j - 1)]   # Up, Down
        }
        intended_move = move[action]
        if is_valid(intended_move):
            probs.append((self.prob_intended, intended_move))
        else:
            probs.append((self.prob_intended, state))

        for perp_move in perpendicular_moves[action]:
            if is_valid(perp_move):
                probs.append((self.prob_perpendicular, perp_move))
            else:
                probs.append((self.prob_perpendicular, state))

        # Adjust probabilities to sum to 1
        new_probs = {}
        for prob, next_state in probs:
            if next_state in new_probs:
                new_probs[next_state] += prob
            else:
                new_probs[next_state] = prob
        return list(new_probs.items())

    def build_rewards(self):
        '''
                Terminal States (4,2) and (4,3):
                    Reward is zero for all actions
                
                Non-terminal states:
                    If an action leads to terminal state (4,2), the reward is -1 and if the action leads to the terminal state (4,3), reward is +1
                    For all other transitions reward is default_reward

        '''

        rewards = {}
        for state in self.states:
            rewards[state] = {}
            # In terminal state, any action leads to the same state with reward 0
            if state in self.terminal_states:
                for action in self.actions:
                    rewards[state][action] = {}
                    rewards[state][action][state] = 0
                continue
            # For non-terminal states, if the transition is to a terminal state, reward is either +1 (to (4,3)) or -1 (to (4,2))            
            for action in self.actions:
                rewards[state][action] = {}
                for next_state, _ in self.transition_probabilities[state][action]:
                    if next_state in self.terminal_states:
                        reward = self.terminal_states[next_state]
                    else:
                        # For non-terminal states, if the transition is to a non-terminal state, reward is default_reward
                        reward = self.default_reward
                    rewards[state][action][next_state] = reward
        return rewards
    
    def get_transition_probabilities(self, state, action):
        return self.transition_probabilities[state][action]
    
    def get_reward(self, state, action, next_state):
        '''
                This function returns reward for a given state, action, next_state triplet, i.e., it returns R(s,a,s')
        '''
        return self.rewards[state][action][next_state]
    



def policy_evaluation(mdp, policy, theta=0.001):
    # implements iterative algorithm to solve a set of linear equations    
    # U_old stores the utilities of previous iteration
    # U stores the updated utilities from current iteration     
    U = {state: 0 for state in mdp.states}
    U_old = U.copy()
    while True:
        delta = 0
        U_old = U.copy()
        #U = {state: 0 for state in mdp.states}
        for state in mdp.states:
            if state in mdp.terminal_states:
                continue # Utilities of terminal states are zero           
            action = policy[state]

            #U(s) = Q(s,a) for a fixed policy           
            U[state] = calculate_q_value(mdp, state, action, U_old)

            delta = max(delta, abs(U[state] - U_old[state]))
        
        #Checking for the convergence of utilities
        if delta < theta:
            break
    return U
    
#Calculate Q values
def calculate_q_value(mdp, state, action, U):
    q_value = 0    
    for next_state, prob in mdp.get_transition_probabilities(state, action):        
        #q_value = Q(s,a)
        #prob = p(s'|s,a)        
        #reward = R(s,a,s')
        reward = mdp.get_reward(state, action, next_state)                                
        q_value += prob * (reward + mdp.gamma * U[next_state])
    return q_value

#Extract policy from utilities
def extract_policy(mdp, U):
    policy = {}
    for state in mdp.states:
        #Terminal states have no policy 
        if state in mdp.terminal_states:
            continue
        best_action = None
        max_value = float('-inf')
        # Calculate Q(s,a) for all actions and select the action a with largest Q(s,a)
        for action in mdp.actions:
            q_value = calculate_q_value(mdp, state, action, U)
            if q_value > max_value:
                max_value = q_value
                best_action = action
        policy[state] = best_action
    return policy
    