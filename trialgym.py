import gym
env = gym.make('CartPole-v0')

class Agent():
    def __init__(self, env):
        self.action_size = env.action_space.n
        print("Action size:", self.action_size)
        
    def get_action(self, state):
        pole_angle = state[2]
        action = 0 if pole_angle < 0 else 1
        return action

agent = Agent(env)
state = env.reset()

for _ in range(1000):
    action = agent.get_action(state)
    state, reward, done, info = env.step(action)
    env.render()

env.close()