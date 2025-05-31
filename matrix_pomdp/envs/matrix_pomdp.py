__credits__ = ["A. Saleh Mteran"]

from os import path
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces


class MatrixPOMDPEnv(gym.Env):
    metadata = {
        "render_modes": ["human", ],
    }

    def __init__(self, p_0, p, o,  r, render_mode: Optional[str] = None, multi_objective=False, true_reward=False):
        """
        Initialize the POMDP environment.
        Parameters
        ----------
        p_0 : np.ndarray
            Initial state distribution, which must sum to 1.
        p : np.ndarray
            Transition probability matrix with shape (num_actions, num_states, num_states).
        o : np.ndarray
            Observation matrix with the same shape as p but (num_actions, num_states, num_observations). Each action must have a corresponding observation.
        r : np.ndarray
            Reward matrix, which can be either:
                - shape (num_states, num_actions) for single-objective,
                - shape (num_objectives, num_states, num_actions) for multi-objective.
        render_mode : Optional[str], default=None
            Rendering option for the environment.
        multi_objective : bool, default=False
            Indicates if the reward is multi-objective.
        true_reward : bool, default=False
            If True returns the true reward for the agent based on the current state.
            rather than the belief state. This provides the agent with the exact reward for its action in the 
            current state, bypassing the uncertainty inherent in the belief-based reward computation.
            Warning: Enabling this parameter reveals the true state to the agent, which can compromise the 
            partially observable nature of the environment. It should only be used during evaluation or debugging 
            to assess the agent's performance under ideal conditions.
        Raises
        ------
        ValueError
            If the initial probabilities do not sum to 1, 
            if transition probabilities are invalid, 
            if the observation matrix does not match the shape of the transition matrix,
            or if the reward matrix has an incorrect shape for single-objective or multi-objective use.
        """
        self.p_0 = p_0
        self.p = p
        self.o = o
        self.r = r
        self.true_reward = true_reward

        self.action_space = spaces.Discrete(p.shape[0])
        self.observation_space = spaces.Box(low=np.zeros(p.shape[1]), high=np.ones(p.shape[1]), dtype=np.float64)
        self.states_array = np.arange(p.shape[1])

        # Check that initial probabilities sum to 1   
        if np.around(p_0.sum(), decimals=6) != 1:
            raise ValueError("The provided initial probabilities (p_0) do not sum to 1.")
        
        # Check that transition probabilities sum to 1 or are 0 for terminal states.
        for s in self.states_array:
            for a in np.arange(self.p.shape[0]):
                if p[a, s, :].sum() != 0 and np.around(p[a, s, :].sum(), decimals=6) != 1:
                    raise ValueError(
                        "The transition probabilities must sum to 1 for every row in the transition matrix. " +
                        f"Transition probabilities for state {s} and action {a} do not sum to 1."
                    )
                
        # Check that observation probabilities sum to 1
        for s in self.states_array:
            for a in np.arange(self.o.shape[0]):
                if np.around(o[a, s, :].sum(), decimals=6) != 1:
                    raise ValueError(
                        "The observation probabilities must sum to 1 for every row in the observation matrix. " +
                        f"Observation probabilities for state {s} and action {a} do not sum to 1."
                    )
                
        # Check that observation matrix has the correct shape
        if o.shape[0] != p.shape[0]:
            raise ValueError("The observation matrix's first dimension (num_actions) must match the transition matrix's first dimension. " +
                             f"Observation matrix actions: {o.shape[0]}, Transition matrix actions: {p.shape[0]}")
        if o.shape[1] != p.shape[1]:
            raise ValueError("The observation matrix's second dimension (num_states) must match the transition matrix's second dimension. " +
                             f"Observation matrix states: {o.shape[1]}, Transition matrix states: {p.shape[1]}")
        
        # Check if the reward matrix is single-objective or multi-objective
        if multi_objective:
            if r.ndim != 3:
                raise ValueError("The multi-objective reward matrix must have 3 dimensions of shape (num_objectives, num_states, num_actions).")
            if r.shape[1] != p.shape[1]:
                raise ValueError(f"The reward matrix's second dimension (num_states) must match the transition matrix's second dimension. Reward matrix states: {r.shape[1]}, Transition matrix states: {p.shape[1]}")
            if r.shape[2] != p.shape[0]:
                raise ValueError("The reward matrix's third dimension (num_actions) must match the transition matrix's first dimension. " +
                                 f"Reward matrix actions: {r.shape[2]}, Transition matrix actions: {p.shape[0]}")
    
            self.step = self.step_multiobjective

            # Replace -np.inf and np.inf with specific bounds if they are known.
            self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.r.shape[0],), dtype=np.float32)

        else:
            if r.ndim != 2:
                raise ValueError("The reward matrix must have 2 dimensions of shape (num_states, num_actions). Or set multi_objective=True for multi-objective rewards.")
            if r.shape[0] != p.shape[1]:
                raise ValueError("The reward matrix's first dimension (num_states) must match the transition matrix's second dimension.",f"Reward matrix states: {r.shape[0]}, Transition matrix states: {p.shape[1]}")
            if r.shape[1] != p.shape[0]:
                raise ValueError("The reward matrix's second dimension (num_actions) must match the transition matrix's first dimension. " +
                                 f"Reward matrix actions: {r.shape[1]}, Transition matrix actions: {p.shape[0]}")
            
            self.step = self.step_singleobjective

        if p_0.shape[0] != p.shape[1]:
            raise ValueError("The initial state distribution (p_0) must have the same number of states as the transition matrix (p). " +
                             f"Initial state distribution states: {p_0.shape[0]}, Transition matrix states: {p.shape[1]}")
                
        self.render_mode = render_mode

        # Find terminal states where the sum of transition probabilities is 0 for all actions
        self.terminal_states = [s for s in self.states_array if self.p[:, s, :].sum() == 0]

        # Initialize the first state
        self.state = np.random.choice(self.states_array, p=self.p_0)

        # Initialize first observation
        self.observation = np.random.choice(self.states_array, p=self.p_0)
        
        # Initialize the belief state equals to the p_0
        self.belief = p_0

    def belief_update(self, b, a, ob):
        """
        Updates the belief distribution based on the previous belief, chosen action, and received observation.
        Args:
            b (np.ndarray): Prior belief distribution over states.
            a (int): Action index.
            ob (int): Observation index.
        Returns:
            np.ndarray: The updated belief distribution over states.
        """
        b_new = (self.p[a].T @ b) * self.o[a, :, ob]
        # Normalize the belief to sum to 1
        if b_new.sum() == 0:
            raise ValueError("Belief normalization failed due to zero sum.")
        b_new /= b_new.sum()
        return b_new
    
    # The step function for the single objective environment
    def step_singleobjective(self, action: np.ndarray):

        # Check if the action is valid in the current state
        if self.p[action, self.state, :].sum() == 0:
            if self.render_mode == "human":
                print(f"/!\\ The action {action} is invalid in state {self.state}" +\
                      f"as p[action={action}, :, state={self.state}].sum() == 0 /!\\")
            new_state = None
            reward = None
            done = None  
        
        else:
            new_state = np.random.choice(self.states_array, p=self.p[action, self.state, :])
            
            self.observation = np.random.choice(self.states_array, p=self.o[action, new_state, :])
            
            self.belief = self.belief_update(b=self.belief, a=action, ob=self.observation)

            if np.around(self.belief.sum(), decimals=6) != 1:
                raise ValueError("The provided belief probabilities do not sum to 1." + f"For action {action}, state {self.state} and observation {self.observation}.")

            # Compute the reward based on the belief and the reward matrix
            if self.true_reward:
                reward = self.r[self.state, action]
            else:
                reward = np.dot(self.belief, self.r[:, action])

            done = (new_state in self.terminal_states)

        self.state = new_state

        if self.render_mode == "human":
            self.render()
        
        return self._get_obs(), reward, done, done, {}
    
    # The step function for the multi-objective environment
    def step_multiobjective(self, action: np.ndarray):

        if self.p[action, self.state, :].sum() == 0:
            if self.render_mode == "human":
                print(f"/!\\ The action {action} is invalid in state {self.state}" +\
                      f"as p[action={action}, :, state={self.state}].sum() == 0 /!\\")
            new_state = None
            reward = None
            done = None  
        
        else:
            new_state = np.random.choice(self.states_array, p=self.p[action, self.state, :])
            
            self.observation = np.random.choice(self.states_array, p=self.o[action, new_state, :])

            self.belief = self.belief_update(b=self.belief, a=action, ob=self.observation)

            if np.around(self.belief.sum(), decimals=6) != 1:
                raise ValueError("The provided belief probabilities do not sum to 1.")

            
            if self.true_reward:
                reward = np.zeros(2) 
                reward[0] = self.r[0, self.state, action]
                reward[1] = self.r[1, self.state, action]
            
            else:
                reward = np.array([
                    np.dot(self.belief, self.r[i, :, action]) for i in range(self.r.shape[0])
                    ])
            
            done = (new_state in self.terminal_states)

        self.state = new_state

        if self.render_mode == "human":
            self.render()
        
        return self._get_obs(), reward, done, done, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment, draws the state from the initial state distribution matrix.

        Args:
            seed (int?): Seed for random number generator
            options (dict): To override the random state initialisation, set the start state by giving:
                options={"start_state": start_state} (e.g. options={"start_state": 0} sets the start state to S0)

        Returns:
            obs (int): New initial state
            info (dict): Always empty, for compatibility with other gymnasium environments
        """
        super().reset(seed=seed)

        start_state = np.random.choice(self.states_array, p=self.p_0)
        self.belief = self.p_0
        
        if options is not None:
            start_state = options.get("start_state") if "start_state" in options else start_state

        self.state = start_state
        self.observation = np.random.choice(self.states_array, p=self.p_0)

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}        
        

    def _get_obs(self):
        """
        Returns the current observation, which is the belief state.
        Returns:
            np.ndarray: The current belief state, a 1D numpy array with shape (num_states,).

        """
        return self.belief
    
    def render(self):
        """
        Prints environment information in the console.

        Returns:

        """
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "human":
            print(f"Current state: {self.state}")

    def close(self):
        pass