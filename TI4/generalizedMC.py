from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import random
import os
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import seaborn as sns


class Ability(ABC):
    def __init__(self, ability_type,ability_prob, ability_num_dice,ability_initiative):
        self.ability_type = ability_type
        self.ability_prob = ability_prob
        self.ability_num_dice = ability_num_dice
        self.ability_initiative = ability_initiative

    @abstractmethod
    def check_initiative(self) -> bool:
        """
        Check if the ability can be used this turn.
        """
        pass

class Unit(ABC):
    def __init__(self, health, hit_probability,num_dice,initiative,ability_type):
        self.health = health
        self.hit_probability = hit_probability
        self.initiative = initiative
        self.ability_type = ability_type
        self.num_dice = num_dice
    @abstractmethod
    def apply_hit(self) -> None:
        """
        Apply a hit to the unit. This should update the unit's health or state.
        """
        pass

    @abstractmethod
    def is_alive(self) -> bool:
        """
        Check if the unit is alive.
        """
        pass

    @abstractmethod
    def __name__(self) -> str:
        """
        Return the name of the unit.
        """
        pass

class Ship(Unit):
    def __init__(self, health, hit_probability,num_dice,initiative,ability_type):
        super().__init__(health, hit_probability,num_dice,initiative,ability_type)
    def apply_hit(self):
        self.health -= 1

    def is_alive(self):
        return self.health > 0
    
    def __name__(self):
        return self.__class__.__name__

class AFB(Ability):
    def __init__(self, ability_type,ability_prob, ability_num_dice,ability_initiative):
        super().__init__(ability_type,ability_prob, ability_num_dice,ability_initiative)
    def check_initiative(self,initiative) -> bool:
        return self.ability_initiative == initiative
    def return_ability(self):
        return (self.ability_prob,self.ability_num_dice)
    def return_targets(self):
        return "Light Air"

class Bombardment(Ability):
    def __init__(self, ability_type,ability_prob, ability_num_dice,ability_initiative):
        super().__init__(ability_type,ability_prob, ability_num_dice,ability_initiative)
    def check_initiative(self,initiative) -> bool:
        return self.ability_initiative == initiative
    def return_ability(self):
        return (self.ability_prob,self.ability_num_dice)
    def return_target(self):
        return "Ground Force"
    
class Destroyer(Ship):
    def __init__(self):
        super().__init__(health=1, hit_probability=0.2,num_dice=1,initiative=1,ability_type="AFB")
        self.ability = AFB(ability_type="AFB",ability_prob=0.2,ability_num_dice=2,ability_initiative=0)

class Cruiser(Ship):
    def __init__(self):
        super().__init__(health=1, hit_probability=0.3,num_dice=1,initiative=1,ability_type=None)

class Dreadnought(Ship):
    def __init__(self):
        super().__init__(health=2, hit_probability=0.5,num_dice=1,initiative=1,ability_type="Bombardment")

class Carrier(Ship):
    def __init__(self):
        super().__init__(health=1, hit_probability=0.2,num_dice=1,initiative=1,ability_type=None)

class Fighter(Ship):
    def __init__(self):
        super().__init__(health=1, hit_probability=0.2,num_dice=1,initiative=1,ability_type=None)

class WarSun(Ship):
    def __init__(self):
        super().__init__(health=2, hit_probability=0.7,num_dice=3,initiative=1,ability_type="Bombardment")

class Flagship(Ship):
    def __init__(self,flagship_type):
        super().__init__(health=flagships[flagship_type]["health"],
                        hit_probability=flagships[flagship_type]['hit_probability'],
                        num_dice=flagships[flagship_type]["num_dice"],
                        initiative=flagships[flagship_type]['initiative'],
                        ability_type=flagships[flagship_type]['ability'])


def initialize_side(side: Dict[str, int]) -> Dict[str, Any]:
    side_objects = {}
    for keys,values in side.items():
        units_arr = []
        for i in range(values):
            units_arr.append(units[keys]())
        side_objects[keys] = units_arr
    return side_objects

def print_side(side: Dict[str, Any]) -> None:
    for keys,values in side.items():
        for i in values:
            print(i.__name__() + " " + str(i.health))
    print("\n")

def gather_hit_dict(side_objects: Dict[str, Any]) -> Dict[float,int]:
    hit_dict = defaultdict(int)
    for keys,values in side_objects.items():
        for i in values:
            hit_dict[i.hit_probability] += 1
    return hit_dict

def calculate_hits(side_objects: Dict[str, Any],rng_key) -> int:
    hit_dict = gather_hit_dict(side_objects)
    return sum(random.binomial(rng_key, n=count, p=hit_probability).item()
                        for (hit_probability), count in hit_dict.items())

def apply_hits(side_objects: Dict[str, Any],hits: int) -> Dict[str, Any]:
    hit_queue = []
    while hits > 0:
        for unit in sustain_units:
            if unit in side_objects:
                for i in side_objects[unit]:
                    if hits == 0:
                        break
                    elif i.is_alive() and i.health > 1:
                        hit_queue.append(i)
                        hits -= 1
        for unit in hit_order:
            if unit in side_objects:
                for i in side_objects[unit]:
                    if hits == 0:
                        break
                    elif i.health > 0:
                        hit_queue.append(i)
                        hits -= 1
    print(hit_queue)
    for i in hit_queue:
        i.apply_hit()

    return side_objects
flagships = {
    "Sol": {"health": 2,"hit_probability": 0.5,"num_dice": 1,"initiative": 1,"ability": None},
    "Jol-Nar": {"health": 2,"hit_probability": 0.5,"num_dice": 1,"initiative": 1,"ability": None},
    "Yin": {"health": 2,"hit_probability": 0.5,"num_dice": 1,"initiative": 1,"ability": None}
}
unit_types = {
    "Light Air": ["Fighter"],
    "Ground Force": ["Infantry","Mech"],
    "Heavy Air": ["Cruiser","Destroyer","Carrier","Dreadnought","WarSun","FlagShip"],
    "Space Dock": ["Space Dock"],
    "PDS": ["PDS"]
}

units = {
    "Fighter":Fighter,
    "Destroyer":Destroyer,
    "Cruiser":Cruiser,
    "Dreadnought":Dreadnought
}

sideA = {"Fighter":3,
         "Destroyer":2}

sideA_objects = initialize_side(sideA)

sideB = {"Fighter":1,
         "Dreadnought":2}

hit_order = ['Fighter','Destroyer','Carrier','Cruiser','Dreadnought','Flagship','WarSun']
sustain_units = ['Dreadnought','WarSun','Flagship']
sideB_objects = initialize_side(sideB)
rng_key = random.PRNGKey(2)
# print_side(sideA_objects)
# print_side(sideB_objects)
# print_side(sideA_objects)
# print_side(sideB_objects)
def check_health(side_objects: Dict[str, Any]) -> int:
    total_health = 0
    for keys,values in side_objects.items():
        for i in values:
            total_health += i.health
    return total_health

def simulate_combat_round(sideA_objects: Dict[str, Any], sideB_objects: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    seed = int.from_bytes(os.urandom(4), 'big')
    rng_key = random.PRNGKey(seed)
    rng_key_a, rng_key_b = random.split(rng_key)
    hitsA = calculate_hits(sideA_objects,rng_key_a)
    hitsB = calculate_hits(sideB_objects,rng_key_b)
    sideA_objects = apply_hits(sideA_objects,hitsA)
    sideB_objects = apply_hits(sideB_objects,hitsB)
    return sideA_objects,sideB_objects

def run_combat_until_elimination(sideA_objects: Dict[str, Any], sideB_objects: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    while True:
        sideA_objects,sideB_objects = simulate_combat_round(sideA_objects,sideB_objects)
        if check_health(sideA_objects) == 0 or check_health(sideB_objects) == 0:
            break
    return sideA_objects,sideB_objects

def mc(num_simulations: int, sideA_objects: Dict[str, Any], sideB_objects: Dict[str, Any]) -> list[float]:
    A_wins = 0
    B_wins = 0
    draws = 0
    for i in range(num_simulations):
        print(i)
        print(sideA_objects)
        print(sideB_objects)
        sideA_objects,sideB_objects = run_combat_until_elimination(sideA_objects,sideB_objects)
        print('check')
        sideA_health = check_health(sideA_objects)
        sideB_health = check_health(sideB_objects)
        if sideA_health > sideB_health:
            A_wins += 1
        elif sideA_health < sideB_health:
            B_wins += 1
        else:
            draws += 1
    return [A_wins/num_simulations,B_wins/num_simulations,draws/num_simulations]

print(mc(1,sideA_objects,sideB_objects))
#sideA_objects,sideB_objects = run_combat_until_elimination(sideA_objects,sideB_objects)
print_side(sideA_objects)
print_side(sideB_objects)