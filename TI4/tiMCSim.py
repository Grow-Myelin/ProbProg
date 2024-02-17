import jax.numpy as jnp
from jax import random
import os
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import seaborn as sns


KeyType = Tuple[int, float]
SideType = Dict[KeyType, int]
RNGKey = Any

def apply_hits(side: SideType, hits_scored: int, rng_key: RNGKey) -> SideType:
    """
    Apply hits to a side based on hit probabilities, updating the state of the side.
    
    Parameters:
        side (SideType): The current state of the side's units.
        hits_scored (int): The number of hits to apply to the side.
        rng_key (RNGKey): A random key for JAX's random number generation.
    
    Returns:
        SideType: The updated state of the side's units after applying hits.
    """
    sorted_keys = sorted(side.keys(), key=lambda x: (x[1], x[0]))
    for key in sorted_keys:
        if hits_scored <= 0:
            break
        hits_to_eliminate, hit_probability = key
        count = side[key]
        while count > 0 and hits_scored > 0:
            count -= 1
            hits_scored -= 1
            if hits_to_eliminate > 1:
                new_key = (hits_to_eliminate - 1, hit_probability)
                side[new_key] = side.get(new_key, 0) + 1
        if count > 0:
            side[key] = count
        else:
            del side[key]
    return side

def simulate_combat_round(side_a: SideType, side_b: SideType, rng_key: RNGKey) -> Tuple[SideType, SideType]:
    """
    Simulates a single round of combat between two sides.
    
    Parameters:
        side_a (SideType): The initial state of side A.
        side_b (SideType): The initial state of side B.
        rng_key (RNGKey): A random key for JAX's random number generation.
    
    Returns:
        Tuple[SideType, SideType]: The updated states of side A and side B after the combat round.
    """
    rng_key_a, rng_key_b = random.split(rng_key)
    hits_scored_a = sum(random.binomial(rng_key_a, n=count, p=hit_probability).item()
                        for (hits_to_eliminate, hit_probability), count in side_a.items())
    hits_scored_b = sum(random.binomial(rng_key_b, n=count, p=hit_probability).item()
                        for (hits_to_eliminate, hit_probability), count in side_b.items())
    side_a_updated = apply_hits(side_a.copy(), hits_scored_b, rng_key_a)
    side_b_updated = apply_hits(side_b.copy(), hits_scored_a, rng_key_b)
    return side_a_updated, side_b_updated

def run_combat_until_elimination_modified(side_a: SideType, side_b: SideType, rng_key: RNGKey, max_rounds: int = 1000) -> Tuple[SideType, SideType, int, SideType, SideType]:
    """
    Runs combat rounds between two sides until one side is eliminated or a maximum
    number of rounds is reached, also returns the initial states.

    Parameters:
    - side_a (SideType): Initial state of side A.
    - side_b (SideType): Initial state of side B.
    - rng_key (RNGKey): JAX random key for generating random numbers.
    - max_rounds (int, optional): Maximum number of rounds to simulate. Defaults to 1000.

    Returns:
    - Tuple[SideType, SideType, int, SideType, SideType]: Final states of side A and side B,
      the number of rounds simulated, and copies of the initial states of side A and side B.
    """
    for round_number in range(1, max_rounds + 1):
        rng_key, round_key = random.split(rng_key)
        side_a, side_b = simulate_combat_round(side_a, side_b, round_key)
        if not side_a or not side_b:
            break
    return side_a, side_b, round_number, side_a.copy(), side_b.copy()  # Return final states as well


def state_to_string(state: SideType) -> str:
    """
    Converts a side's state to a string representation for consistent key usage.

    This function sorts the units in the side by their hit probabilities and health
    before converting to string, ensuring consistent string representation for
    identical states.

    Parameters:
    - state (SideType): The current state of the side's units, represented as a
      dictionary with keys as (unit health, hit probability) and values as unit counts.

    Returns:
    - str: A string representation of the state, sorted by unit properties.
    """
    sorted_items = sorted(state.items(), key=lambda x: (x[0][1], x[0][0], x[1]))
    return str(sorted_items)

def calculate_health(state_str: str) -> int:
    """
    Calculates the total health score based on a side's state string representation.

    This function interprets the state string, extracting unit health and count
    to compute the overall health score of the side.

    Parameters:
    - state_str (str): A string representation of a side's state, typically
      obtained from `state_to_string` function.

    Returns:
    - int: The total health score calculated from the state.
    """
    if not state_str:
        return 0
    # Convert string back to list of tuples
    state = eval(state_str)
    health_score = sum(a * c for (a, _), c in state)
    return health_score


def mc(initial_side_a,initial_side_b,num_simulations):
    """
    Conducts a Monte Carlo simulation to estimate the outcome probabilities of combat between two sides.
    
    Parameters:
        initial_side_a (SideType): The initial state of side A.
        initial_side_b (SideType): The initial state of side B.
        num_simulations (int): The number of simulations to run.
    
    Returns:
        Tuple[List[str], List[int]]: The outcomes (as labels) and their counts.
    """
    unique_outcomes_side_a = defaultdict(int)
    unique_outcomes_side_b = defaultdict(int)
    draws_count = 0
    for _ in range(num_simulations):
        seed = int.from_bytes(os.urandom(4), 'big')
        rng_key = random.PRNGKey(seed)
        _, _, _, final_side_a, final_side_b = run_combat_until_elimination_modified(initial_side_a, initial_side_b, rng_key)
        if not final_side_a and not final_side_b:
            draws_count += 1
        # Convert final states to string keys
        final_state_a_str = state_to_string(final_side_a)
        final_state_b_str = state_to_string(final_side_b)
        
        # Increment the count for each unique outcome
        unique_outcomes_side_a[final_state_a_str] += 1
        unique_outcomes_side_b[final_state_b_str] += 1
    # Sort outcomes by health score
    sorted_outcomes_side_a = sorted(unique_outcomes_side_a.items(), key=lambda x: calculate_health(x[0]), reverse=True)
    sorted_outcomes_side_b = sorted(unique_outcomes_side_b.items(), key=lambda x: calculate_health(x[0]), reverse=True)

    # Combine lists for Side A and Side B, adding a label to identify the side
    combined_outcomes = [(label, count, 'A') for label, count in sorted_outcomes_side_a] + \
                        [(label, count, 'B') for label, count in sorted_outcomes_side_b]

    # Sort Side A by decreasing health, Side B by increasing health
    sorted_outcomes_a = sorted([(label, count) for label, count, side in combined_outcomes if side == 'A'], key=lambda x: calculate_health(x[0]), reverse=True)
    sorted_outcomes_b = sorted([(label, count) for label, count, side in combined_outcomes if side == 'B'], key=lambda x: calculate_health(x[0]))

    # Assuming draws is the sum of counts for '[]' from both sides
    draws = ('draws', draws_count)

    # Combine all outcomes with draws in the middle

    # Filter out 0 health outcomes for Side A and Side B
    filtered_outcomes_a = [(label, count) for label, count in sorted_outcomes_a if calculate_health(label) > 0]
    filtered_outcomes_b = [(label, count) for label, count in sorted_outcomes_b if calculate_health(label) > 0]

    # Adjusting labels for filtered outcomes
    labels_with_health_a = [f"{calculate_health(label)} (A)" for label, _ in filtered_outcomes_a]
    labels_with_health_b = [f"{calculate_health(label)} (B)" for label, _ in filtered_outcomes_b]

    # Labels for draws
    labels_with_health_draws = ["Draws"]

    # Combine all labels
    all_labels = labels_with_health_a + labels_with_health_draws + labels_with_health_b

    # Combine all values while maintaining the correct order and filtering
    all_values = [count for _, count in filtered_outcomes_a] + [draws[1]] + [count for _, count in filtered_outcomes_b]
    return all_labels,all_values

def normalize(all_labels,all_values):
    """
    Normalizes the simulation counts to probabilities
    
    Parameters:
        all_labels (List[str]): The labels for each outcome.
        all_values (List[int]): The counts for each outcome.
        filename (str): The file path to save the plot.
    """
    total_simulations = sum(all_values)

    # Convert frequencies to probabilities
    probabilities = [value / total_simulations for value in all_values]

    # Create a DataFrame
    data = pd.DataFrame({
        'Outcome': all_labels,
        'Probability': probabilities
    })
    return data

# Example usage
initial_side_a = {(2, 0.6): 3}
initial_side_b = {(1, 0.2): 6}
num_simulations = 10000

def plot_and_simulate_data(initial_side_a,initial_side_b,num_simulations):
    """
    Runs Monte Carlo simulations to estimate combat outcomes between two sides and plots the results.

    This function orchestrates the simulation process by calling the appropriate functions to
    simulate combat, calculate outcome probabilities, and plot the probabilities in a bar plot.

    Parameters:
    - initial_side_a (SideType): The initial state of side A.
    - initial_side_b (SideType): The initial state of side B.
    - num_simulations (int): The number of simulations to run for estimating outcomes.

    The function does not return a value but generates and saves a plot visualizing the
    probability of different outcomes based on the simulations.
    """
    all_labels,all_values = mc(initial_side_a,initial_side_b,num_simulations)
    data = normalize(all_labels,all_values)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))
    barplot = sns.barplot(x='Outcome', y='Probability', data=data, palette='coolwarm', errorbar=None)
    barplot.set_title('Probability of Outcomes (Side A → Draws → Side B)', fontsize=16)
    barplot.set_xlabel('Outcome', fontsize=14)
    barplot.set_ylabel('Probability', fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('plots/ti4_seaborn.png')

plot_and_simulate_data(initial_side_a,initial_side_b,num_simulations)