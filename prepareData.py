import json
import PIL
from PIL import Image
import numpy as np
import random
import glob, os

results = {}
results["X"] = []
results["y"] = []
print(results)

os.chdir("images/human_small")

for file in glob.glob("*.jpg"): #train humans
    image = np.array(Image.open(file))
    print(image[4:])
    results["X"].append(np.swapaxes(np.swapaxes(image, 1, 2), 0, 1))
    results["y"].append(0)

os.chdir("images/penguins_small")

for file in glob.glob("*.jpg"): #train penguins
    image = np.array(Image.open(file))
    results["X"].append(np.swapaxes(np.swapaxes(image, 1, 2), 0, 1))
    results["y"].append(1)

os.chdir("../../data")

with open('data.json', 'w') as outfile:
    json.dump(data, outfile)