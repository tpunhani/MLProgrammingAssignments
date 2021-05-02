# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import random

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # Find all the possible states in the environment
        all_states = self.mdp.getStates()

        # run a loop equal to number of iterations
        for iter in range(self.iterations):
            new_values = self.values.copy()    #initialize variable with 0

            # Run a loop for every state and calculate their values
            for state in all_states:

                # check if the state is the end state
                if self.mdp.isTerminal(state):
                    new_values[state] = 0
                    continue

                maxQValue = -1000   # Minimum value for initial comparison

                # find all possible transitions of the state
                state_actions = self.mdp.getPossibleActions(state)

                # calculate the maximum qvalue action from possible actions
                for action in state_actions:
                    q_action = self.computeQValueFromValues(state, action)
                    if q_action > maxQValue:
                        maxQValue = q_action


                # update the best qvalue of a state
                new_values[state] = maxQValue

            # Update global values variable
            self.values = new_values



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        # Initially check if the state is the end state than no need to calculate qvalue
        if self.mdp.isTerminal(state):
            return 0

        final_q_value = 0

        # loop through every possible transition and update final q value of the state
        for transition_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            final_q_value = final_q_value + prob * (self.mdp.getReward(state, action, transition_state) + self.discount * self.getValue((transition_state)))

        return final_q_value
        util.raiseNotDefined()


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # check if it's a legal action
        if self.mdp.isTerminal(state):
            return None

        # get all the possible actions from a state
        all_actions = self.mdp.getPossibleActions(state)

        # Choose best q value as the minimum for initialization
        best_q_value = -100000

        best_possible_actions = [] # there can be more than one possible action


        # check every action q value and compare it with the best q value till now
        for possible_action in all_actions:
            q_action_value = self.computeQValueFromValues(state, possible_action)

            if q_action_value > best_q_value:
                best_possible_actions = []
                best_q_value = q_action_value
            if q_action_value == best_q_value:
                best_possible_actions.append(possible_action)


        # return the random of equal values best actions
        return random.choice(best_possible_actions)
        util.raiseNotDefined()



    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

