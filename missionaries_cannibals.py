# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 18:09:23 2020

@author: sebastien.delandtshe
"""

# From Wikipedia:
# In the missionaries and cannibals problem, three missionaries and three cannibals 
# must cross a river using a boat which can carry at most two people, 
# under the constraint that, for both banks, if there are missionaries present 
# on the bank, they cannot be outnumbered by cannibals 
# (if they were, the cannibals would eat the missionaries). 
# The boat cannot cross the river by itself with no people on board.

# Solution 1: stochastic search

import random
from operator import add

# Def: Missionaries = M, Cannibals = C
# initial state of the system: #Mleft, #Cleft, #Mright, #Cright
initial_state = [3, 3, 0, 0]
final_state = [0, 0, 3, 3]

# Boat actions:
boat_left_to_right = [[-2, 0, 2, 0], [0, -2, 0, 2], [-1, 0, 1, 0], [0, -1, 0, 1], [-1, -1, 1, 1]]
boat_right_to_left = [[2, 0, -2, 0], [0, 2, 0, -2], [1, 0, -1, 0], [0, 1, 0, -1], [1, 1, -1, -1]]


def is_move_possible(current_state, boat_action):
    from operator import add
    next_state = list(map(add, current_state, boat_action))
    # test negative people
    if any([i < 0 for i in next_state]):
        return False
    # test cannibalism
    elif 0 < next_state[0] < next_state[1]:
        return False
    elif 0 < next_state[2] < next_state[3]:
        return False
    else:
        return True


iter_number = 0
best = 1000
max_iter = 10000
solution = []

while iter_number < max_iter:
    iter_number = iter_number + 1
    #    print('iteration number %d' % iter_number)
    list_moves = []
    list_states = []
    boat_left = True
    depth = 0
    current_state = initial_state
    list_states.append(current_state)
    while depth < best:
        if boat_left:
            boat_action = random.choice(boat_left_to_right)
        else:
            boat_action = random.choice(boat_right_to_left)

        if is_move_possible(current_state, boat_action):
            current_state = list(map(add, current_state, boat_action))
            depth = depth + 1
            boat_left = not boat_left
            list_moves.append(boat_action)
            list_states.append(current_state)

        if current_state == final_state:
            best = depth
            solution = [list_moves, list_states]
            print('final state reached after %d steps' % best)
            break
print('moves: ' + str(best))
print(str(solution[0]))
print('states: ')
print(str(solution[1]))


#################################
# Solution 2: breath-first search


class Queue:  # list with put and get methods First-In-First-Out
    def __init__(self):
        self.queue = []

    def __repr__(self):
        return f'Queue object containing {self.queue}'

    def get(self):
        first = self.queue[0]
        self.queue = self.queue[1::]
        return first

    def put(self, element):
        self.queue.append(element)
        return self


class State:
    def __init__(self, missionaries_left: int = 3,
                 missionaries_right: int = 0,
                 cannibals_left: int = 3,
                 cannibals_right: int = 0,
                 boat: str = 'left'):

        self.mis_left = missionaries_left
        self.mis_right = missionaries_right
        self.can_left = cannibals_left
        self.can_right = cannibals_right
        self.boat = boat
        self.moves = []

    def __str__(self):
        if self.boat == 'left':
            mid_str = ' B   |||||     '
        else:
            mid_str = '     |||||   B '
        return 'M' * self.mis_left + 'C' * self.can_left + mid_str + 'M' * self.mis_right + 'C' * self.can_right

    def __repr__(self):
        if self.boat == 'left':
            mid_str = ' B   |||||     '
        else:
            mid_str = '     |||||   B '
        return 'M' * self.mis_left + 'C' * self.can_left + mid_str + 'M' * self.mis_right + 'C' * self.can_right

    def short(self):
        return [self.mis_left, self.can_left, self.mis_right, self.can_right, self.boat == 'left']

    def make_move(self, move):
        # flatten list
        while any(isinstance(x, list) for x in move):
            move = move[0]
        self.mis_left = self.mis_left + move[0]
        self.can_left = self.can_left + move[1]
        self.mis_right = self.mis_right + move[2]
        self.can_right = self.can_right + move[3]
        if self.boat == 'left':
            self.boat = 'right'
        else:
            self.boat = 'left'
        self.moves.append(move)

    def make_moves(self, movelist):
        if isinstance(movelist, list):  # if several moves
            for move in movelist:
                self.make_move(move)
        else:  # if only one move
            self.make_move(movelist)
        return self

    def is_valid(self):
        if self.mis_left > 0:
            if self.can_left > self.mis_left:
                return False
        if self.mis_right > 0:
            if self.can_right > self.mis_right:
                return False
        return True

    def is_goal(self):
        return self.is_valid and (self.can_left == 0) and (self.mis_left == 0)

    def next_moves(self):
        next_moves = []
        if self.boat == 'left':
            if self.mis_left > 1:
                next_moves.append([-2, 0, 2, 0])
            if self.mis_left > 0:
                next_moves.append([-1, 0, 1, 0])
            if self.can_left > 1:
                next_moves.append([0, -2, 0, 2])
            if self.can_right > 0:
                next_moves.append([0, -1, 0, 1])
            if self.mis_left > 0 and self.can_left > 0:
                next_moves.append([-1, -1, 1, 1])
        else:
            if self.mis_right > 1:
                next_moves.append([2, 0, -2, 0])
            if self.mis_right > 0:
                next_moves.append([1, 0, -1, 0])
            if self.can_right > 1:
                next_moves.append([0, 2, 0, -2])
            if self.can_right > 0:
                next_moves.append([0, 1, 0, -1])
            if self.mis_right > 0 and self.can_right > 0:
                next_moves.append([1, 1, -1, -1])

        return next_moves

# create initial object

state = State()
states_list = []
queue = Queue()
current_path = []
possible_moves = state.next_moves()  # start first move
for move in possible_moves:
    if isinstance(move, list):
        c = current_path.copy()
        c.append([move])
        queue = queue.put(c)
    else:  # only one move
        c = current_path.copy()
        queue = queue.put(c.append([possible_moves]))

while True:
    state = State()
    path = queue.get()
    state = state.make_moves(path)  # realize this solution
    if state.is_valid():
        if state.short() not in states_list:  # not been there yet
            states_list.append(state.short())
            current_path = path
            if not state.is_goal():  # while not at the final state
                possible_moves = state.next_moves()
                for move in possible_moves:
                    if isinstance(move, list):
                        c = current_path.copy()
                        c.append([move])
                        queue = queue.put(c)
                    else:  # only one move
                        c = current_path.copy()
                        queue = queue.put(c.append([possible_moves]))
            else:
                break
print('Solution:')
final = State()
print(f'0:                      {final}')
for i, move in enumerate(state.moves):
    final.make_move(move)
    print(f'{i + 1}: {move}:       {final}')



































