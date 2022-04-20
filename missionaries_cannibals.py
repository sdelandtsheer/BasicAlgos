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
	elif next_state[0] > 0 and next_state[0] < next_state[1]:
			return False
	elif next_state[2] > 0 and next_state[2] < next_state[3]:
			return False
	else:
		return True

iter_number = 0
best = 1000
max_iter = 10000
solution = []

while iter_number < max_iter:
	
	iter_number = iter_number + 1
# 	print('iteration number %d' % iter_number)
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
			boat_left =  not boat_left
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





































