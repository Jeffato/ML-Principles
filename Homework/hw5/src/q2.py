import numpy as np
import pandas as pd
import random

# CPTs
cloudy_cpt = {
    True : 0.5,
    False : 0.5
}

# Cloudy
sprinker_cpt = {
    True : 0.1,
    False : 0.5
}

# Cloudy
rain_cpt = {
    True: {True: 0.8, False: 0.2},   # P(R | C)
    False: {True: 0.2, False: 0.8}
}

# Sprinkler, Rain
wet_grass_cpt = {
    (True, True): 0.99,
    (True, False): 0.9,
    (False, True): 0.9,
    (False, False): 0.0,

}

# States
cloudy = True
rainy = True
sprinker = True
wet_grass = True

samples = [] 
iterations = 1000000

for _ in range(iterations):
    # Cloudy : Markov blanket: Sprinkler, Rain
    p_cloudy = cloudy_cpt[True] * sprinker_cpt[True] * rain_cpt[True][rainy]
    p_neg_cloudy = cloudy_cpt[False] * sprinker_cpt[False] * rain_cpt[False][rainy]
    cloud_norm = p_cloudy + p_neg_cloudy
    cloudy = random.choices([True, False], weights=[p_cloudy / cloud_norm, p_neg_cloudy / cloud_norm], k=1)[0]

    # Rain : Markov Blanket: Cloud, sprinkler, Rain
    p_rainy = rain_cpt[cloudy][True] * wet_grass_cpt[sprinker, True]
    p_neg_rainy = rain_cpt[cloudy][False] * wet_grass_cpt[sprinker, False]
    rainy_norm = p_rainy + p_neg_rainy
    rainy = random.choices([True, False], weights=[p_rainy / rainy_norm, p_neg_rainy / rainy_norm], k=1)[0]

    samples.append({'Cloudy': cloudy, 'Rainy': rainy})

sample_df = pd.DataFrame(samples)

print(sample_df)
print(sample_df.shape)

print(f'Posterior: {sample_df['Cloudy'].mean()}')