# The eight-queens problem

# Chess composer Max Bezzel published the eight queens puzzle in 1848
# Place eight queens on a chessboard (8*8) so that no two queens attack each other

# Solution 1: naive solution by solving for rooks and filter solutions that are attacking

# generate all permutations of numbers 1 to 8, and use this as indices to place queens
import itertools

trials = list(itertools.permutations(list(range(8))))
solutions = []

for idx, trial in enumerate(trials):
    valid = True
    for i, j in enumerate(trial):
        if valid:
            for i2, j2 in enumerate(trial):
                if valid and i2 != i:
                    # check if these two attack to the right
                    if abs(i2 - i) == abs(j2 - j):
                        valid = False
    if valid:
        solutions.append(trial)

for solution in solutions:
    print(solution)


###########################
# Solution 2: recursive bfs in a generator


def queens(sol, r, queens_to_place, a1, a2):
    if queens_to_place:  # a is not empty
        for c in queens_to_place:
            if r + c not in a1 and r - c not in a2:
                yield from queens(sol + [c], i + 1, queens_to_place - {c}, a1 | {r + c}, a2 | {r - c})
    else:
        yield sol


for solution in queens([], 0, set(range(8)), set(), set()):
    print(solution)
















