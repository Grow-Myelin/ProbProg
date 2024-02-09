import jax.numpy as jnp
from jax.scipy.stats import binom
import time


def calculate_hit_probabilities_jax(dice_probabilities):
    """
    Function to calculate the probabilities of getting any number of hits using JAX,
    given a dictionary of hit probabilities for different dice.

    Parameters:
    - dice_probabilities: A dictionary where keys are the hit probabilities for a single die,
      and values are the number of such dice.

    Returns:
    - A list where the index represents the number of hits, and the value at each index is
      the probability of getting that many hits.
    """
    # Total number of dice
    total_dice = sum(dice_probabilities.values())

    # Initialize a JAX numpy array to hold the probabilities for each number of hits
    probabilities = jnp.zeros(total_dice + 1)
    probabilities = probabilities.at[0].set(1.0)

    # Iterate through each type of dice and update the probabilities array
    for hit_probability, num_dice in dice_probabilities.items():
        # Calculate the distribution for the current type of dice
        distribution = jnp.array([binom.pmf(k, num_dice, hit_probability) for k in range(num_dice + 1)])
        # Convolve the current distribution with the existing probabilities
        probabilities = jnp.convolve(probabilities, distribution)

    return probabilities.tolist()

# Example usage with JAX:
dice_probabilities_jax_example = {
    0.4: 2,  # Probability of 0.4 (hit on 7, 8, 9, or 10) for 2 dice
    0.2: 3,  # Probability of 0.2 (hit on 9 or 10) for 3 dice
}

start = time.time()
num_dice = sum(dice_probabilities_jax_example.values())
end = time.time()
print(calculate_hit_probabilities_jax(dice_probabilities_jax_example)[:num_dice+1])
print(end-start)