import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            env.render()
            break
    return np.array(V)


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: Lambda discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        V = policy_eval_fn(policy, env)
        policy_stable = True

        for s in range(env.nS):
            old_action = np.argmax(policy[s])
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            max_action = np.argmax(action_values)
            if old_action != max_action:
                policy_stable = False
            policy[s] = np.eye(env.nA)[max_action]

        if policy_stable:
            return policy, V

def value_iteration(env, discount_factor=1.0, theta=0.00001):
    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])

    while True:
        delta = 0
        for s in range(env.nS):
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            v = max(action_values)
            max_action = np.argmax(action_values)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
            policy[s] = np.eye(env.nA)[max_action]
        if delta < theta:
            env.render()
            break

    return policy, V

# policy evaluation
# random_policy = np.ones([env.nS, env.nA]) / env.nA
# v = policy_eval(random_policy, env)
#
# print("Value Function:")
# print(v)
# print("")
#
# print("Reshaped Grid Value Function:")
# print(v.reshape(env.shape))
# print("")
#
# # Test: Make sure the evaluated policy is what we expected
# expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
# np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
#
# # policy iteration
# policy, v = policy_improvement(env)
# print("Policy Probability Distribution:")
# print(policy)
# print("")
#
# print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
# print(np.reshape(np.argmax(policy, axis=1), env.shape))
# print("")
#
# print("Value Function:")
# print(v)
# print("")
#
# print("Reshaped Grid Value Function:")
# print(v.reshape(env.shape))
# print("")
#
# expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
# np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

# value iteration
policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
