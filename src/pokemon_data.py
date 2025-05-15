import numpy as np

def get_win_probabilities():
    probabilities = {
        # Format: (pokemon_a, pokemon_b): probability of pokemon_a winning
        ('primarina', 'primarina'): 0.5,
        ('primarina', 'sylveon'): 0.09583,
        ('primarina', 'rillaboom'): 0.0252,
        ('primarina', 'heatran'): 1.0,
        ('primarina', 'urshifu'): 1.0,
        ('primarina', 'zapdos'): 0.0,  # Zapdos wins 100% against Primarina

        ('sylveon', 'primarina'): 0.90417,
        ('sylveon', 'sylveon'): 0.5,
        ('sylveon', 'rillaboom'): 0.387,
        ('sylveon', 'heatran'): 0.0,
        ('sylveon', 'urshifu'): 1.0,
        ('sylveon', 'zapdos'): 0.99375,  # CORRECTED: Sylveon wins 99.375% against Zapdos

        ('rillaboom', 'primarina'): 0.9748,
        ('rillaboom', 'sylveon'): 0.613,
        ('rillaboom', 'rillaboom'): 0.5,
        ('rillaboom', 'heatran'): 0.95,
        ('rillaboom', 'urshifu'): 0.6954,
        ('rillaboom', 'zapdos'): 0.0165,  # Zapdos wins 98.35% against Rillaboom

        ('heatran', 'primarina'): 0.0,
        ('heatran', 'sylveon'): 1.0,
        ('heatran', 'rillaboom'): 0.05,
        ('heatran', 'heatran'): 0.5,
        ('heatran', 'urshifu'): 0.0,
        ('heatran', 'zapdos'): 0.0,  # Zapdos wins 100% against Heatran

        ('urshifu', 'primarina'): 0.0,
        ('urshifu', 'sylveon'): 0.0,
        ('urshifu', 'rillaboom'): 0.3046,
        ('urshifu', 'heatran'): 1.0,
        ('urshifu', 'urshifu'): 0.5,
        ('urshifu', 'zapdos'): 0.3,  # Zapdos wins 70% against Urshifu

        ('zapdos', 'primarina'): 1.0,
        ('zapdos', 'sylveon'): 0.00625,  # CORRECTED: Now matches inverse of sylveon vs zapdos
        ('zapdos', 'rillaboom'): 0.9835,
        ('zapdos', 'heatran'): 1.0,
        ('zapdos', 'urshifu'): 0.7,
        ('zapdos', 'zapdos'): 0.5,

        ('garchomp', 'zapdos'): 1.0,
        ('zapdos', 'garchomp'): 0.0,
        ('garchomp', 'primarina'): 0.04167,
        ('primarina', 'garchomp'): 0.95833,
        ('garchomp', 'sylveon'): 0.4012,
        ('sylveon', 'garchomp'): 0.5988,
        ('garchomp', 'heatran'): 0.1831,
        ('heatran', 'garchomp'): 0.8169,
        ('garchomp', 'rillaboom'): 0.9676,
        ('rillaboom', 'garchomp'): 0.0324,
        ('garchomp', 'urshifu'): 0.04167,
        ('urshifu', 'garchomp'): 0.9583,
        ('garchomp', 'garchomp'): 0.5,
    }
    return probabilities
