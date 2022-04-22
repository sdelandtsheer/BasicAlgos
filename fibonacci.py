### Examples of the Fibo function
# The Fibonacci sequence is the sequence of integers, starting with [0, 1] where every number
# is the sum of the two preceding ones.
# The sequence goes 0, 1, 1, 2, 3, 5, 8, 13, 21, etc.

# 1) Full naive recursivity (beginner):
def fib1(n: int) -> int:
    return fib1(n - 1) + fib1(n - 2)
# this will result in 'infinite recursivity' because there is no base case


# 2) Adding base cases (advanced):
def fib2(n: int) -> int:
    if n < 2:
        return n
    else:
        return fib2(n - 1) + fib2(n - 2)
# this works, but is inefficient: every call to fib2 results in two more calls.
# for fib2(4), there are 9 calls in total. There are 21891 calls for fib2(20).


# 3) Memoization with a dictionary (pro):
def fib3(n: int) -> int:
    memo = {0: 0, 1: 1}
    if n not in memo:
        memo[n] = fib3(n - 1) + fib3(n - 2)
    else:
        return memo[n]
# of note, the 'lru_cache' decorator that does that automatically


# 4) Iterative approach (old-school):
def fib4(n: int) -> int:
    if n == 0:
        return n
    last_n = 0
    next_n = 1
    for _ in range(1, n):
        last_n, next_n = next_n, last_n + next_n
    return next_n


# 5) Using a generator (fancy):
def fib5(n: int):
    yield 0
    if n > 0:
        yield 1
    last_n = 0
    next_n = 1
    for _ in range(1, n):
        last_n, next_n = next_n, last_n + next_n
        yield next_n

