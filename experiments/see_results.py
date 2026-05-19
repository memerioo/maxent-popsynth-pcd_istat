import numpy as np
import sys
import os

# 1. Path Setup: Point to the 'src' directory just like before
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# 2. Import the metadata dictionary and attribute names
from istat.attr_meta_ISTAT import ATTR_NAMES_SYNTH, ATTR_META

def read_profiles(filename="test_2_pop200000.npy", num_to_print=5):
    # Load the synthetic population
    print(f"Loading population from {filename}...\n")
    pop = np.load(filename)
    
    total_people, total_attrs = pop.shape
    print(f"Total individuals in file: {total_people}\n")
    
    # Loop through the requested number of individuals
    for i in range(min(num_to_print, total_people)):
        traits = []
        
        # Look at each attribute column for this person
        for col_idx, attr_name in enumerate(ATTR_NAMES_SYNTH):
            # Get the raw integer
            val_idx = pop[i, col_idx]
            
            # Translate the integer back to the string value using the metadata
            val_str = ATTR_META[attr_name]['vals'][val_idx]
            
            traits.append(f"{attr_name}: {val_str}")
            
        # Format it nicely: profile1: sex: F ; age: 25-34 ; ...
        profile_string = f"Profile {i+1}:\n" + " ; ".join(traits)
        print(profile_string)
        print("-" * 80)

if __name__ == "__main__":
    # Change 'num_to_print' to see more or fewer people!
    read_profiles("test_2_pop200000.npy", num_to_print=10)