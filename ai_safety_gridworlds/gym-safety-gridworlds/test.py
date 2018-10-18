from gym_safety_gridworlds.envs.island_navigation import IslandNavigation

env = IslandNavigation()

s = env.reset()
print(s)

ns, reward, done, _ = env.step(1)
print(ns)
print(reward)
print(done)
