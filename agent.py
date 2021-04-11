import numpy as np
import utils
import random

HIGH_BOUND = 520
LOW_BOUND = 0
BLOCK_SIZE = 40

class Agent:

    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

        self.reset()

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''
        # print(state)
        # get rewards
        reward = self.get_reward(points, dead)
        # snake head in food pellet position
        if reward is 1:
            self.points = points
        new_discrete_state = discretize(state)
        # print(new_discrete_state)
        if self._train:
            if self.s is not None:
                # Update q table with best action that maximizes Q
                curr_q = self.Q[self.s + (self.a,)]
                self.Q[self.s + (self.a,)] = self.get_q(curr_q, reward, new_discrete_state)
            # Compute next action with exploration
            maximum_action = float('-inf')
            max_idx = 0
            for i in range(len(self.actions)):
                each_action = self.explore(new_discrete_state, i)
                if each_action >= maximum_action:
                    maximum_action = each_action
                    max_idx = i
            self.a = max_idx
            # Update N table
            if not dead:
                self.N[new_discrete_state + (self.a,)] += 1
            self.s = new_discrete_state
            if dead:
                self.reset()
        else:
            self.a = np.argmax(self.Q[new_discrete_state])

        return self.a


    def get_q(self, q_val, reward, s_prime):
        lr = self.C / (self.C + self.N[self.s + (self.a,)])
        new_q = q_val + lr * (reward + (self.gamma * np.max(self.Q[s_prime]) - q_val))
        return new_q

    def explore(self, s, i):
        if self.N[s + (i,)] < self.Ne:
            return 1
        else:
            return self.Q[s + (i,)]

    def get_reward(self, points, dead):
        if points > self.points:
            return 1
        elif dead:
            return -1
        else:
            return -0.1


def discretize(state):
    '''
    state: [200, 200, [], 80, 80] ---discretize--->
    (adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
    '''
    mdp_state = np.zeros(8, dtype=int)
    # set adjoining_wall_x
    if state[0] == LOW_BOUND or state[0] == HIGH_BOUND:
        mdp_state[0] = 0
    elif (state[0] + BLOCK_SIZE) == HIGH_BOUND:
        mdp_state[0] = 2
    elif (state[0] - BLOCK_SIZE) == LOW_BOUND:
        mdp_state[0] = 1
    else:
        mdp_state[0] = 0

    # set adjoining_wall_y
    if state[1] == LOW_BOUND or state[1] == HIGH_BOUND:
        mdp_state[1] = 0
    elif (state[1] + BLOCK_SIZE) == HIGH_BOUND:
        mdp_state[1] = 2
    elif (state[1] - BLOCK_SIZE) == LOW_BOUND:
        mdp_state[1] = 1
    else:
        mdp_state[1] = 0
    # set food_dir_x
    if state[0] == state[3]:
        mdp_state[2] = 0
    elif state[0] > state[3]:
        mdp_state[2] = 1
    else:
        mdp_state[2] = 2
    # set food_dir_y
    if state[1] == state[4]:
        mdp_state[3] = 0
    elif state[1] > state[4]:
        mdp_state[3] = 1
    else:
        mdp_state[3] = 2

    mdp_state[4] = 0
    mdp_state[5] = 0
    mdp_state[6] = 0
    mdp_state[7] = 0

    if len(state[2]) > 0:
        # store body list into a hashtable for faster computation
        body_hash = {}
        for item in state[2]:
            body_hash[item] = 1

        # set adjoining_body_top
        if (state[0], state[1] - BLOCK_SIZE) in body_hash.keys():
            mdp_state[4] = 1
        # set adjoining_body_bottom
        if (state[0], state[1] + BLOCK_SIZE) in body_hash.keys():
            mdp_state[5] = 1
        # set adjoining_body_left
        if (state[0] - BLOCK_SIZE, state[1]) in body_hash.keys():
            mdp_state[6] = 1
        # set adjoining_body_right
        if (state[0] + BLOCK_SIZE, state[1]) in body_hash.keys():
            mdp_state[7] = 1
        # print(mdp_state)
        # print(state)
    return tuple(mdp_state)
