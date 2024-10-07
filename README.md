This is explanation for the model.py file

DQN parameters:
- epsilon: Picking epsilon is picking the percentage of the time that the model chooses the known best action. 
        For example if epsilon = 0.9 (90%), if a random number we choose is below 90%, we take the best action. 
        10% of the time is left to explore other options in case there is a better one.
- epsilon_decay: This can be set to a very very low number, and as we progress through epsiodes

