import os
from cyclonedds.internal import load_cyclonedds
try:
    lib = load_cyclonedds()
    print("Loaded cyclonedds library successfully!")
    print("Library path/object:", lib)
except Exception as e:
    print("Failed to load:", e)
