## Description

This environment provides a discrete-action, continuous-observation space for Partially Observable Markov Decision Process (POMDPs). It supports single-objective or multi-objective rewards.

**Easy to Use**  
This environment is straightforward and only requires matrix input for transitions, observations, and rewards.

**Origin and Multi-Objective Integration**  
This environment is adapted from the [matrix-mdp-gym](https://github.com/Paul-543NA/matrix-mdp-gym) repository. Its multi-objective environment setting is based on [MO-Gymnasium](https://github.com/Farama-Foundation/MO-Gymnasium).

## Starting State

• `p_0` is a 1D numpy array (shape (num_states,)) defining the initial probability over states.  
• The environment samples the starting state from `p_0`.

## Action Space

Defined by the first dimension of `p` (a 3D array with shape (num_actions, num_states, num_states)):  
• Dimension 0 → selected action  
• Dimension 1 → current state  
• Dimension 2 → next state

## Observation Space

Observations are belief states (1D arrays with shape (num_states,)), updated each step using an observation matrix `o` (shape (num_actions, num_states, num_observations)).  
• The belief is initialized to `p_0`.  
• It updates with the chosen action and received observation.

## Rewards

• Single-objective: A scalar from the dot product of the belief state with `r` (shape (num_states, num_actions)).  
• Multi-objective: A vector from the dot product of the belief state with `r` (shape (num_objectives, num_states, num_actions)).  
• The optional `true_reward` parameter bypasses the belief and uses the actual state (only for evaluation or debugging).

## Episode Truncation

Episodes end at terminal states, which have zero transition probabilities for all actions. These states are identified at initialization.

## Arguments

• `p_0`: Initial state distribution (shape (num_states,))  
• `p`: Transition probability matrix (shape (num_actions, num_states, num_states))  
• `o`: Observation matrix (shape (num_actions, num_states, num_observations))  
• `r`: Reward matrix (single-objective or multi-objective)  
• `render_mode`: Optional rendering mode  
• `multi_objective`: Boolean for multi-objective rewards  
• `true_reward`: Boolean to return the state-based reward

## Usage

Create a `MatrixPOMDPEnv` instance with the required parameters. Refer to the examples folder for Jupyter-based usage.
Examples implemented in this repository draw on the POMDP-based methodologies discussed in:
Corotis, R. B., Ellis, J. H., & Jiang, M. (2005). Modeling of risk-based inspection, maintenance and life-cycle cost with partially observable Markov decision processes. Structure and Infrastructure Engineering, 1(1), 75–84. https://doi.org/10.1080/15732470412331289305
