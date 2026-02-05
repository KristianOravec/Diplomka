import random
import matplotlib.pyplot as plt

def simulate_guessing(iterations=100, range_limit=1000):
    guesses = []
    min_so_far_list = []
    current_min = float('inf')

    for _ in range(iterations):
        # Generate a random guess
        guess = random.randint(0, range_limit)
        guesses.append(guess)

        # Update the minimal number found so far
        if guess < current_min:
            current_min = guess

        min_so_far_list.append(current_min)

    return min_so_far_list

# Run the simulation
data = simulate_guessing()

# Visualization
plt.figure(figsize=(10, 6))
plt.step(range(len(data)), data, where='post', color='red', linewidth=2, label='Minimum Found')
plt.title('Minimal Number Found So Far (0-1000)')
plt.xlabel('Number of Guesses')
plt.ylabel('Minimal Value')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()