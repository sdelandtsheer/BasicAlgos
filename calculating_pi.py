### Calculating pi
# It is possible to generate the decimals of 'pi' to arbitrary precision using the
# Leibniz formula: pi = 4 - 4/3 + 4/5 - 4/7 + 4/9 - 4/11 etc.

def calculate_pi(n_terms: int):
    denominator = 1.0
    operation = 1.0
    pi = 0.0
    for _ in range(n_terms):
        pi += operation * (4 / denominator)
        denominator += 2.0
        operation *= -1.0
    return pi

# x = calculate_pi(1000000)