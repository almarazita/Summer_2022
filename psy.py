# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 17:20:19 2022

@author: almar

Chautauqua Institution Special Studies 2022, Week 2
Psychological detectives connecting with our natural world

Original study investigating the impact of connection to nature on mental
wellness, controlling for age, sex, self-control, and cell phone compulsion.
"""

from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

# Read in data by line to create list of 130-length strings
with open('psy.txt') as f:
    data = f.readlines()


# Parse to named tuples used to make custom subject class
Subject = namedtuple("Subject", ['age', 'sex', 'nature', 'control', 'affect',
                                 'smartphone'])


# Create list of Subject objects from each line, removing whitespace
# Convert age to an integer and sex to a boolean
sample = [Subject(int(line[128:130]), bool(int(line[130])), line[0:28].split(),
                  line[28:54].split(), line[54:95].split(),
                  line[96:127].split()) for line in data]


def convert(sequence):
    """ Changes elements in 2D list from strings to floats. """
    
    new_sequence = []
    
    for x in sequence:
        conversion = [float(y) for y in x]
        new_sequence.append(conversion) # Creates list of lists
    
    return new_sequence


# Create list for each attribute/questionnaire as proper data type
ages = [x.age for x in sample]
sexes = [x.sex for x in sample]
natures = convert([x.nature for x in sample])
controls = convert([x.control for x in sample])
affects = convert([x.affect for x in sample])
smartphones = convert([x.smartphone for x in sample])


def reverseScore(sequence, *items):
    """ Reverse scores (5-point scale) elements in 2D list that are in
    items. """
    
    for row, x in enumerate(sequence):
        for col, y in enumerate(x):
            if col + 1 in items:
                sequence[row][col] = 6 - y


# Calculate scores for each questionnaire according to past studies
reverseScore(natures, 4, 12, 14)
nature_scores = [sum(x) for x in natures]

reverseScore(controls, 2, 3, 4, 5, 7, 9, 10, 12, 13)
control_scores = [sum(x) for x in controls]

affect_scores = [sum(x[0::2]) - sum(x[1::2]) for x in affects]

smartphone_scores = [sum(x) for x in smartphones]


# Update sample to be list of Subject objects using processed and scored data
sample = [Subject(ages[i], sexes[i], nature_scores[i],
                  control_scores[i], affect_scores[i],
                  smartphone_scores[i]) for i in range(0, len(data))]
print(sample)


# Plot
plt.scatter(nature_scores, affect_scores)
plt.axis([14, 70, -40, 40])
plt.xlabel("Connection to Nautre")
plt.ylabel("Mental Wellbeing")
plt.suptitle("Affect and Nature Connectedness")
coef = np.polyfit(np.array(nature_scores), np.array(affect_scores), deg=1)
m = coef[0]
b = coef[1]
x = np.arange(14, 70, 0.01)
plt.plot(x, m * x + b)
plt.show()


# Write results to a new text file for further processing in Excel/SPSS
with open('results.txt', "w") as f:
    for i in range(len(data)):
        f.write(str(ages[i]) + ' ')
        f.write(str(sexes[i]) + ' ')
        f.write(str(sample[i].nature) + ' ')
        f.write(str(sample[i].control) + ' ')
        f.write(str(sample[i].affect) + ' ')
        f.write(str(sample[i].smartphone) + '\n')