### The towers of Hanoi:
# Three vertical 'towers' called A, B, and C can be stacked with discs with varying widths.
# Larger discs cannot be placed on top of smaller ones. In the starting position, all 3 discs
# are on top of tower A, with the widest (disc 1) at the bottom, disk 2 on top of disk 1, and
# disc 3 on top of disc 2.
# The goal is to move all discs from tower A to tower C, touching only one at a time.

# First we need a stack for the towers:
class Stack:
    def __init__(self):
        self.s = []

    def push(self, item):
        self.s.append(item)  # append at the top

    def pop(self):
        return self.s.pop()  # remove from top

    def __repr__(self):
        return f"Stack: {self.s}"


# define the recursive solving:
def solve_hanoi(start, end, temp, n_discs):
    if n_discs == 1:
        end.push(start.pop()) # if only one disc, move it
    else:
        solve_hanoi(start, temp, end, n_discs - 1)  # put n-1 discs (not the top one) to temp location
        solve_hanoi(start, end, temp, 1)  # move the bottom disc to final location
        solve_hanoi(temp, end, start, n_discs - 1)  # put discs remaining on final location


# define the towers:
a = Stack()
b = Stack()
c = Stack()
for disc in range(1, 4):
    a.push(disc)

solve_hanoi(a, c, b, 3)






