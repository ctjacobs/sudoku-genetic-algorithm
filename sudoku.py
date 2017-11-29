""" Solves a Sudoku puzzle using a genetic algorithm. This is based on a piece of coursework produced as part of the CS3M6 Evolutionary Computation module at the University of Reading.

@author: Christian Thomas Jacobs
"""

import numpy
import random

Ng = 9

class Population(object):
    def __init__(self, Np, given):
        self.candidates = []  # The candidate solutions (also known as the chromosomes) in the population.

        # Determine the legal values that each square can take.
        helper = Grid()
        helper.values = [[[] for j in range(0, Ng)] for i in range(0, Ng)]
        for row in range(0, Ng):
            for column in range(0, Ng):
                for value in range(1, 10):
                    if((given.values[row][column] == 0) and not (given.is_column_duplicate(column, value) or given.is_block_duplicate(row, column, value) or given.is_row_duplicate(row, value))):
                        # Value is available.
                        helper.values[row][column].append(value)
                    elif(given.values[row][column] != 0):
                        # Value is a given.
                        helper.values[row][column].append(given.values[row][column])
                        break

        # Seed a new population.
        random.seed()
        
        for p in range(0, Np):
            g = Grid()
            for i in range(0, Ng): # New row in candidate.
                row = numpy.zeros(Ng)
                
                # Fill in the givens.
                for j in range(0, Ng): # New column j value in row i.
                
                    # If value is already given, don't change it.
                    if(given.values[i][j] != 0):
                        row[j] = given.values[i][j]
                    # Fill in the gaps using the helper board.
                    elif(given.values[i][j] == 0):
                        row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j])-1)]

                # If we don't have a valid board, then try again. There must be no duplicates in the row.
                while(len(list(set(row))) != Ng):
                    for j in range(0, Ng):
                        if(given.values[i][j] == 0):
                            row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j])-1)]

                g.values[i] = row

            self.candidates.append(g)
            print g.values
            
        return
    
    def seed(self):
        
        return

class Grid(object):
    def __init__(self):
        self.values = numpy.zeros((Ng, Ng))
        return

class GivenGrid(Grid):
    def __init__(self, values):
        self.values = values
        return
        
    def is_row_duplicate(self, row, value):
        """ Check whether there is a duplicate of a FIXED value in a row. """
        for column in range(0, Ng):
            if(self.values[row][column] == value):
               return True
        return False

    def is_column_duplicate(self, column, value):
        """ Check whether there is a duplicate of a FIXED value in a column. """
        for row in range(0, Ng):
            if(self.values[row][column] == value):
               return True
        return False

    def is_block_duplicate(self, row, column, value):
        """ Check whether there is a duplicate of a FIXED value in a 3 x 3 block. """
        i = 3*(int(row/3))
        j = 3*(int(column/3))

        if((self.values[i][j] == value)
           or (self.values[i][j+1] == value)
           or (self.values[i][j+2] == value)
           or (self.values[i+1][j] == value)
           or (self.values[i+1][j+1] == value)
           or (self.values[i+1][j+2] == value)
           or (self.values[i+2][j] == value)
           or (self.values[i+2][j+1] == value)
           or (self.values[i+2][j+2] == value)):
            return True
        else:
            return False
        
        
class Sudoku(object):
    def __init__(self):
        self.given = None
        return
    
    def load(self, path):
        # Load a configuration to solve.
        with open(path, "r") as f:
            values = numpy.loadtxt(f).reshape((Ng, Ng)).astype(int)
            self.given = GivenGrid(values)
        return
    
    def solve(self):
        self.population = Population(10, self.given)
    
s = Sudoku()
s.load("puzzle.txt")
s.solve()
