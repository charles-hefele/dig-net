import gym
import gym_digger
import numpy as np
import random
from gym import envs
from IPython.display import clear_output


# env = gym.make('Digger-v0')
env = gym.make('DiggerDiscrete-v0')

# print all envs
# envids = [spec.id for spec in envs.registry.all()]
# for envid in sorted(envids):
#     print(envid)

q_table = np.zeros((env.observation_space.n, env.action_space.n))
# print(q_table)

num_episodes = 10000
max_steps_per_episode = 150

learning_rate = 0.1  # 0 means only use old knowledge, 1 means only use new knowledge)
discount_rate = 0.99  # 0 means prioritize short-term reward, 1 means prioritize long-term reward

exploration_rate = 1  # 0 means never take a random step, 1 means always take a random step
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []

max_step_count = 0
maxed_out_count = 0
done_count = 0
first_done_episode = 0

for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):

        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])  # choose from policy
        else:
            action = env.action_space.sample()  # choose randomly

        # take a new step
        new_state, reward, done, info = env.step(action)

        # update all reward
        rewards_current_episode += reward

        # update Q-table for Q(s, a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        # check episode end criteria
        if done:
            if done_count == 0:
                first_done_episode = episode
            done_count += 1
            print(f'done in {step} steps in episode {episode}')

            if step > max_step_count:
                max_step_count = step

            break

        elif step == max_steps_per_episode - 1:
            maxed_out_count += 1
            print(f'reached max step count of {max_steps_per_episode} in episode {episode}')

        # transition to the new state
        state = new_state

    # Exploration rate decay, but only after finding the goal for the first time
    if done_count > 0:
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    rewards_all_episodes.append(rewards_current_episode)

# print max steps taken
print(f'done_count: {done_count}')
print(f'first_done_episode: {first_done_episode}')
print(f'max_step_count: {max_step_count}')
print(f'maxed_out_count: {maxed_out_count}')

# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
count = 1000
print("*******Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r / 1000)))
    count += 1000

# Print updated Q-table
print("\n\n**********Q-table**********\n")
print(q_table)


# Animate it
env = gym.make('DiggerDiscrete-v0')
max_steps_per_episode = 150

for episode in range(10):
    # initialize new episode params
    state = env.reset()
    done = False
    print("*****EPISODE ", episode+1, "*****\n\n\n\n")
    # time.sleep(1)

    for step in range(max_steps_per_episode):
        # Show current state of environment on screen
        # Choose action with highest Q-value for current state
        # Take new action

        clear_output(wait=True)
        env.render()
        print(f'Step: {step}')
        # time.sleep(0.1)

        action = np.argmax(q_table[state,:])
        new_state, reward, done, info = env.step(action)

        if done:
            clear_output(wait=True)
            env.render()
            print(f'All nutrients have been dug!')
            # time.sleep(3)
            clear_output(wait=True)
            break

        elif step == max_steps_per_episode - 1:
            print(f"You reached the maximum step count of {max_steps_per_episode}")
            # time.sleep(3)
            clear_output(wait=True)

        state = new_state

env.close()