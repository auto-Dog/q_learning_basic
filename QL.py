import pandas as pd
import random
import time

# [o o <-(o)-> o o x] Target: control the robot <-( )-> until it is on 'x'
# This is a version thet fit the standard framework of RL, pseudocode seen at https://www.eecs.tufts.edu/~mguama01/post/q-learning/qlearning.png
action_space = ['left','right']     # cause state change
state_space = range(10)
# state: a value in state_space, is not the environment itself
# Q-map: state(index) | action0 score | action1 score

# Q-function based RL, some code from https://blog.csdn.net/zjl0409/article/details/121867048
class Robot:
    def __init__(self) -> None:
        super(Robot,self).__init__()
        global action_space,state_space
        self.epsilon = 0.9  # Greddy rate
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.8  # Reward decline rate 
        self.q_table = pd.DataFrame(data=[[0 for _ in action_space] for _ in state_space], index=state_space, columns=action_space)
        self.action_record = None
        self.last_state = None
        

    def Q_function(self,state,action,reward):
        # update Q map
        next_state_q_values = self.q_table.loc[state, self.get_valid_actions(state)]
        self.q_table.loc[self.last_state, action] += self.alpha * (
            reward + self.gamma * next_state_q_values.max() - self.q_table.loc[self.last_state, action])

    def get_valid_actions(self,state):
        global action_space,state_space  # ['left', 'right']
        valid_actions = set(action_space)
        if state == state_space[-1]:  # No space for turn right
            valid_actions -= set(['right'])  
        if state == state_space[0]:  # No space for turn left
            valid_actions -= set(['left'])  
        return list(valid_actions)

    def choose_action(self,state,reward,count):
        if(count!=0):
        # 3. update Q_table (In this code style, the last try will not be learned)
            self.Q_function(state,self.action_record,reward)

        global action_space,state_space
        # 1. choose a from s using policy derived from Q (ε-greedy)
        if (random.uniform(0, 1) > self.epsilon) or ((self.q_table.loc[state] == 0).all()):  
            current_action = random.choice(self.get_valid_actions(state))
        else:
            current_action = self.q_table.loc[state].idxmax()
        self.action_record = current_action
        self.last_state = state
        return current_action
        # 2. take action a, observe r, s'
            # (has been finished in main loop)

# Q-function based Env
class Env:
    def __init__(self,state_space) -> None:
        super(Env,self).__init__()    
        self.state = state_space[0]
        # self.rewards = [-1, -0.5, 0, 0.8, -0.5, 1, -0.2, 1, 0.5, 0.1] # A improper case, reward not match target cause exploration-exploitation
        self.rewards = [-1, -0.5, -0.5, -0.5, -0.5, 1, -0.2, 0, 0, 0]  # reward
        self.target = 5

    def reset(self):
        global action_space,state_space
        self.state = state_space[0]
        return self.state

    def step(self,action):
        global action_space,state_space

        if action == 'right' and self.state != state_space[-1]:  
            next_state = self.state + 1
        elif action == 'left' and self.state != state_space[0]:  
            next_state = self.state - 1
        else:
            next_state = self.state

        # Take action
        self.state = next_state
        if self.state == self.target:
            done = True
        else:
            done = False

        return self.state, self.rewards[self.state], done

    def render(self):
        v_map = [0,0,0,0,0,0,0,0,0,0]
        v_map[self.target] = 'x'
        v_map[self.state] = 'A'
        print('\r',v_map,end=' ')
        time.sleep(0.01)


# RL structure (Generic)
env = Env(state_space)
agent = Robot()
num_episodes = 10
reward = 0
for i in range(num_episodes):
    state = env.reset()
    RL_count = 0
    while True:
        action = agent.choose_action(state,reward,RL_count)
        state, reward, done = env.step(action)
        env.render()
        RL_count += 1
        if done:
            break
    print('\n----- Total cost:',RL_count,' Times-----')
# For using a pretrained model: 
# 1. load Q table (or other model)
# 2. set init state, then run code after line 'while True:' to get an agent plan
    