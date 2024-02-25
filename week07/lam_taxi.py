# Load OpenAI Gym and other necessary packages
import gym
import numpy as np


# Environment
env = gym.make("Taxi-v3", render_mode='ansi')
state_size = env.observation_space.n
action_size = env.action_space.n

# Training parameters for Q learning
alpha = 0.9 # Learning rate
gamma = 0.9 # Future reward discount factor
num_of_episodes = 1000
num_of_steps = 500 # per each episode
epsilon = 0.1 # Exploration probability
qtable = np.zeros([state_size, action_size])

# Function for evaluating the policy
def eval_policy(qtable_, num_of_episodes_, max_steps_):
    rewards = []
    num_of_interactions = []
    for episode in range(num_of_episodes_): # This is out loop over num of episodes
        state = env.reset()[0]
        total_reward = 0
        total_interactions = 0
        for step in range(max_steps_):
            action = np.argmax(qtable_[state,:])
            new_state, reward, done, truncated, info = env.step(action)
            total_interactions += 1
            total_reward += reward
            if done:
                break
            else:
                state = new_state
        rewards.append(total_reward)
        num_of_interactions.append(total_interactions)
        env.close()
    return sum(rewards)/num_of_episodes_, sum(num_of_interactions)/num_of_episodes_

# Q-learning to modify policy
for episode in range(num_of_episodes):
    states = []
    actions = []
    rewards = []
    state = env.reset()[0]
    # Choose action (epsilon-greedy)
    if np.random.uniform() < epsilon:
        action = np.argmax(qtable[state,:])
    else:
        action = np.random.randint(0,action_size)
    # Run episode
    for step in range(num_of_steps):        
        new_state, reward, done, truncated, info = env.step(action)
        # Choose action (epsilon-greedy)
        if np.random.uniform() < epsilon:
            new_action = np.argmax(qtable[state,:])
        else:
            new_action = np.random.randint(0,action_size)
        # Update rule
        if done:
            qtable[state,action] = reward 
            break
        else:
            qtable[state,action] = qtable[state,action] + alpha*(reward+gamma*np.max(qtable[new_state,:])-qtable[state,action]) 
            state = new_state
            action = new_action

eval_rewards = []
eval_interactions = []
num_of_evals = 50
# Running the test 10 times
for repeat in range(10):
    eval_reward, interaction = eval_policy(qtable,num_of_evals*10,100)
    eval_rewards.append(eval_reward)
    eval_interactions.append(interaction)

# Getting the mean values and print them
eval_reward_q_learning_mean = np.mean(np.asarray(eval_rewards))
eval_interaction_q_learning_mean = np.mean(np.asarray(eval_interactions))
print(f"Q-learning reward average {eval_reward_q_learning_mean:.2f} and interactions average {eval_interaction_q_learning_mean:.2f}")