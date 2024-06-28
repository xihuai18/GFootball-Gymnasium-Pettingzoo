from pettingzoo.test import parallel_api_test

from gfootball import gfootball_pettingzoo_v1

# API Tests
env = gfootball_pettingzoo_v1.parallel_env(
    "academy_3_vs_1_with_keeper", representation="simplev1", number_of_left_players_agent_controls=1
)
parallel_api_test(env, 400)

env = gfootball_pettingzoo_v1.parallel_env(
    "academy_3_vs_1_with_keeper", representation="simplev1", number_of_left_players_agent_controls=2
)
parallel_api_test(env, 400)

# Seed Tests
env1 = gfootball_pettingzoo_v1.parallel_env(
    "academy_3_vs_1_with_keeper", representation="simplev1", number_of_left_players_agent_controls=2
)

obs1_list = []
obs1, info1 = env1.reset(seed=42)
obs1_list.append(obs1)

while True:
    obs1, _, terminated1, _, _ = env1.step({agent: env1.action_space(agent).sample() for agent in env1.agents})
    obs1_list.append(obs1)

    if any(terminated1.values()):
        break

env2 = gfootball_pettingzoo_v1.parallel_env(
    "academy_3_vs_1_with_keeper", representation="simplev1", number_of_left_players_agent_controls=2
)

obs2_list = []
obs2, info2 = env2.reset(seed=42)
obs2_list.append(obs2)

while True:
    obs2, _, terminated2, _, _ = env2.step({agent: env2.action_space(agent).sample() for agent in env2.agents})
    obs2_list.append(obs2)

    if any(terminated2.values()):
        break

for i, (obs1, obs2) in enumerate(zip(obs1_list, obs2_list)):
    assert all(obs1[agent] == obs2[agent] for agent in env1.agents), f"Observations at step {i} differ: {obs1} {obs2}"

print("Seed test passed!")

# Wrapper Tests
from co_mas.wrappers import AutoResetParallelEnvWrapper, OrderForcingParallelEnvWrapper

env = gfootball_pettingzoo_v1.parallel_env(
    "academy_3_vs_1_with_keeper",
    representation="simplev1",
    number_of_left_players_agent_controls=2,
    additional_wrappers=[OrderForcingParallelEnvWrapper],
)

try:
    env.step({agent: env.action_space(agent).sample() for agent in env.agents})
except Exception as e:
    assert str(e) == "Environment must be reset before stepping", e
    print("Order Forcing Test Passed!")
else:
    raise AssertionError("Expected reset error")


env = gfootball_pettingzoo_v1.parallel_env(
    "academy_3_vs_1_with_keeper",
    representation="simplev1",
    number_of_left_players_agent_controls=2,
    additional_wrappers=[OrderForcingParallelEnvWrapper, AutoResetParallelEnvWrapper],
)

env.reset(seed=42)

while True:
    _, _, terminated, _, _ = env.step({agent: env.action_space(agent).sample() for agent in env.agents})

    if all(terminated.values()):
        break

_, _, terminated, _, _ = env.step({agent: env.action_space(agent).sample() for agent in env.agents})

assert terminated != {agent: True for agent in env.agents}

print("Auto Reset Test Passed!")

print("Wrapper Test Passed!")
