import jax
import jax.numpy as jnp
from jax import random
import os
from typing import Dict, Tuple, Any

KeyType = Tuple[int, float]
SideType = Dict[KeyType, int]
RNGKey = Any


def apply_hits_jax(side: SideType, hits_scored: int, rng_key: RNGKey) -> SideType:
    """
    Apply hits to a side with JAX, prioritizing dice with lower hit probabilities
    and lower health. Incorporates randomness in selecting dice within the same
    priority level to take hits. Accepts an RNG key for reproducible randomness.

    Parameters:
        side (SideType): Dictionary representing the side's units and their stats.
        hits_scored (int): Number of hits to apply to the side.
        rng_key (RNGKey): JAX random key for generating random numbers.

    Returns:
        SideType: Updated side after applying hits.
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


def simulate_combat_round_jax(side_a: SideType, side_b: SideType, rng_key: RNGKey) -> Tuple[SideType, SideType]:
    """
    Simulates a single round of combat between two sides using JAX.

    Parameters:
        side_a (SideType): First combatant side.
        side_b (SideType): Second combatant side.
        rng_key (RNGKey): JAX random key for generating random numbers.

    Returns:
        Tuple[SideType, SideType]: Updated states of side_a and side_b after combat.
    """
    rng_key_a, rng_key_b = random.split(rng_key)
    hits_scored_a = sum(random.binomial(rng_key_a, n=count, p=hit_probability).item()
                        for (hits_to_eliminate, hit_probability), count in side_a.items())
    hits_scored_b = sum(random.binomial(rng_key_b, n=count, p=hit_probability).item()
                        for (hits_to_eliminate, hit_probability), count in side_b.items())

    side_a_updated = apply_hits_jax(side_a.copy(), hits_scored_b, rng_key_a)
    side_b_updated = apply_hits_jax(side_b.copy(), hits_scored_a, rng_key_b)

    return side_a_updated, side_b_updated


def run_combat_until_elimination(side_a: SideType, side_b: SideType, rng_key: RNGKey, max_rounds: int = 1000) -> Tuple[SideType, SideType, int]:
    """
    Runs combat rounds recursively until one side is eliminated or a maximum
    number of rounds is reached.

    Parameters:
        side_a (SideType): First combatant side.
        side_b (SideType): Second combatant side.
        rng_key (RNGKey): JAX random key for generating random numbers.
        max_rounds (int): Maximum number of rounds to simulate.

    Returns:
        Tuple[SideType, SideType, int]: Final states of side_a and side_b, and the number of rounds simulated.
    """
    for round_number in range(1, max_rounds + 1):
        rng_key, round_key = random.split(rng_key)
        side_a, side_b = simulate_combat_round_jax(side_a, side_b, round_key)
        if not side_a or not side_b:
            break

    return side_a, side_b, round_number


def monte_carlo_combat_simulation(initial_side_a: SideType, initial_side_b: SideType, num_simulations: int = 1000) -> Dict[str, float]:
    """
    Performs a Monte Carlo simulation of combat between two sides over a specified
    number of simulations to estimate outcome probabilities.

    Parameters:
        initial_side_a (SideType): Initial state of side A.
        initial_side_b (SideType): Initial state of side B.
        num_simulations (int): Number of simulations to run.

    Returns:
        Dict[str, float]: Probabilities of different outcomes.
    """
    outcomes = {"side_a_wins": 0, "side_b_wins": 0, "draws": 0}
    
    for _ in range(num_simulations):
        seed = int.from_bytes(os.urandom(4), 'big')
        rng_key = random.PRNGKey(seed)
        final_side_a, final_side_b, _ = run_combat_until_elimination(initial_side_a.copy(), initial_side_b.copy(), rng_key)

        if not final_side_a and not final_side_b:
            outcomes["draws"] += 1
        elif not final_side_a:
            outcomes["side_b_wins"] += 1
        elif not final_side_b:
            outcomes["side_a_wins"] += 1

    probabilities = {outcome: count / num_simulations for outcome, count in outcomes.items()}

    return probabilities


# Example usage
initial_side_a = {(2, 0.6): 3}
initial_side_b = {(1, 0.2): 6}
num_simulations = 10000

probabilities = monte_carlo_combat_simulation(initial_side_a, initial_side_b, num_simulations)
print(probabilities)