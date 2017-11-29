""" Solves a Sudoku puzzle using a genetic algorithm.

@author: Christian Thomas Jacobs
"""

import numpy
import random

Ng = 9

class Population(object):
    def __init__(self):
        self.candidates = []
        return
    
    def seed(self):
        
        return

class Grid(object):
    def __init__(self):
        self.values = numpy.zeros((Ng, Ng))
    
        return
        
class Sudoku(object):
    def __init__(self):
    
        self.given = Grid()
        return
    
    def load(self, path):
        # Load a configuration to solve.
        with open(path, "r") as f:
            values = numpy.loadtxt(f).reshape((Ng, Ng))
        self.given.values = values
        return
    
s = Sudoku()
s.load("puzzle.txt")
