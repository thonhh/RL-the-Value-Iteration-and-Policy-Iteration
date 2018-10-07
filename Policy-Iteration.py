from time import time
import numpy as np
import gym

theta=0.0001


def one_step_lookahead(env,discount_factor,state, V):
    """
    Helper function to calculate the value for all action in a given state.

    Args:
        state: The state to consider (int)
        V: The value to use as an estimator, Vector of length env.nS

    Returns:
        A vector of length env.nA containing the expected value of each action.
    """
    A = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[state][a]:
            A[a] += prob * (reward + discount_factor * V[next_state])
    return A

def policy_eval(policy, env, discount_factor):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):#https://daynhauhoc.com/t/hoi-ve-enumerate/29931/4
                # For each action, look at the possible next states...
                for prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)

#4--------------------------------------------------------------------
'''
def getPolicy(env, stateValues, discount_factor):
    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(env,discount_factor,s, stateValues)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0

    return policy
def the_expected_value_of_each_action(env,discount_factor, selectBest):


    V = np.zeros(env.nS)

    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(env.nS):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(env,discount_factor,s, V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10.
            V[s] = best_action_value
            # Check if we can stop
        if delta < theta:
            break
    return V
'''
def policy_improvement(env, discount_factor, policy_need_improve, policy_eval_fn=policy_eval):
    policy=policy_need_improve
    while True:
        # Evaluate the current policy
        V = policy_eval_fn(policy, env, discount_factor)

        # Will be set to false if we make any changes to the policy
        policy_stable = True

        # For each state...
        for s in range(env.nS):
            # The best action we would take under the currect policy
            chosen_a = np.argmax(policy[s])

            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = one_step_lookahead(env,discount_factor,s, V)
            best_a = np.argmax(action_values)

            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]

        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V


#3--------------------------------------------------------------------


'''
def valueIteration(env, gamma):
  stateValues = the_expected_value_of_each_action(env, gamma, selectBest=True)#computeStateValues(env, gamma, selectBest=True)
  policy = getPolicy(env, stateValues, gamma)#constructGreedyPolicy(env, stateValues, gamma)
  return policy
'''




def policyIteration(env, gamma):
  #stateValues = the_expected_value_of_each_action(env, gamma, selectBest=True)#computeStateValues(env, gamma, selectBest=True)
  #policy = getPolicy(env, stateValues, gamma)#constructGreedyPolicy(env, stateValues, gamma)

  # Start with a random policy
  policy_need_improve = np.ones([env.nS, env.nA]) / env.nA
  policy, v = policy_improvement(env,gamma,policy_need_improve)
  return policy









#2--------------------------------------------------------------------
evaluateIterations = 2#1000

def solveEnv(env, methods, envName):
    print(f'Solving environment {envName}')
    for method in methods:
        name, f, gamma = method
        tstart = time()
        policy = f(env, gamma)
        tend = time()
        print(f'It took {tend - tstart} seconds to compute a policy using "{name}" with gamma={gamma}')

        #score = evaluatePolicy(env, policy, evaluateIterations)
        #print(f'Policy average reward is {score}')

methods = [
    #('Value Iteration', valueIteration, 0.9),
    ('Policy Iteration', policyIteration, 0.9),
]


#1--------------------------------------------------------------------
frozenLake4 = gym.make('FrozenLake-v0')
solveEnv(frozenLake4.env, methods, 'Frozen Lake 4x4')