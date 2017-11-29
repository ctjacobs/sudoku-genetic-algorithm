""" Solves a Sudoku puzzle using a genetic algorithm. This is based on a piece of coursework produced as part of the CS3M6 Evolutionary Computation module at the University of Reading.

@author: Christian Thomas Jacobs
"""

import numpy
import random
random.seed()

Nd = 9

def sort_fitness(x, y):
    if(x.fitness < y.fitness):
        return 1
    elif(x.fitness == y.fitness):
        return 0
    else:
        return -1
            
class Population(object):
    def __init__(self):
        return

    def seed(self, Np, given):
        self.candidates = []  # The candidate solutions (also known as the chromosomes) in the population.
        
        # Determine the legal values that each square can take.
        helper = Candidate()
        helper.values = [[[] for j in range(0, Nd)] for i in range(0, Nd)]
        for row in range(0, Nd):
            for column in range(0, Nd):
                for value in range(1, 10):
                    if((given.values[row][column] == 0) and not (given.is_column_duplicate(column, value) or given.is_block_duplicate(row, column, value) or given.is_row_duplicate(row, value))):
                        # Value is available.
                        helper.values[row][column].append(value)
                    elif(given.values[row][column] != 0):
                        # Value is a given.
                        helper.values[row][column].append(given.values[row][column])
                        break

        # Seed a new population.       
        for p in range(0, Np):
            g = Candidate()
            for i in range(0, Nd): # New row in candidate.
                row = numpy.zeros(Nd)
                
                # Fill in the givens.
                for j in range(0, Nd): # New column j value in row i.
                
                    # If value is already given, don't change it.
                    if(given.values[i][j] != 0):
                        row[j] = given.values[i][j]
                    # Fill in the gaps using the helper board.
                    elif(given.values[i][j] == 0):
                        row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j])-1)]

                # If we don't have a valid board, then try again. There must be no duplicates in the row.
                while(len(list(set(row))) != Nd):
                    for j in range(0, Nd):
                        if(given.values[i][j] == 0):
                            row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j])-1)]

                g.values[i] = row

            self.candidates.append(g)
            print(g.values)
            print(g.fitness)
        
        print("Seeding complete.")
        
        return
        
    def sort(self):
        """ Sort the population based on fitness. """
        self.candidates.sort(sort_fitness)
        return
    

class Candidate(object):
    def __init__(self):
        self.values = numpy.zeros((Nd, Nd))
        return

    @property
    def fitness(self):
        column_count = numpy.zeros(Nd)
        block_count = numpy.zeros(Nd)
        column_sum = 0
        block_sum = 0

        for i in range(0, Nd):  # For each column...
            for j in range(0, Nd):  # For each number within the current column...
                column_count[self.values[i][j]-1] += 1  # ...Update list with occurrence of a particular number.

            column_sum += (1.0/len(set(column_count)))/Nd
            column_count = numpy.zeros(Nd)

        # For each block...
        for i in range(0, Nd, 3):
            for j in range(0, Nd, 3):
                block_count[self.values[i][j]-1] += 1
                block_count[self.values[i][j+1]-1] += 1
                block_count[self.values[i][j+2]-1] += 1
                
                block_count[self.values[i+1][j]-1] += 1
                block_count[self.values[i+1][j+1]-1] += 1
                block_count[self.values[i+1][j+2]-1] += 1
                
                block_count[self.values[i+2][j]-1] += 1
                block_count[self.values[i+2][j+1]-1] += 1
                block_count[self.values[i+2][j+2]-1] += 1

                block_sum += (1.0/len(set(block_count)))/Nd
                block_count = numpy.zeros(Nd)

        # Calculate overall fitness.
        if (int(column_sum) == 1 and int(block_sum) == 1):
            fitness = 1.0
        else:
            fitness = column_sum * block_sum
            
        return fitness
        
    def mutate(self, mutation_rate, given):
        """ Mutate a candidate. """

        r = random.uniform(0, 1.1)
        while(r > 1): # Outside [0, 1] boundary - choose another
            r = random.uniform(0, 1.1)
    
        success = False
        if (r < mutation_rate):  # Mutate.
            while(not success):
                row1 = random.randint(0, 8)
                row2 = random.randint(0, 8)
                row2 = row1
                
                from_column = random.randint(0, 8)
                to_column = random.randint(0, 8)
                while(from_column == to_column):
                    from_column = random.randint(0, 8)
                    to_column = random.randint(0, 8)   

                # Check if the two places are free...
                if(given.values[row1][from_column] == 0 and given.values[row1][to_column] == 0):
                    # ...and that we are not causing a duplicate in the rows' columns.
                    if(not given.is_column_duplicate(to_column, self.values[row1][from_column])
                       and not given.is_column_duplicate(from_column, self.values[row2][to_column])
                       and not given.is_block_duplicate(row2, to_column, self.values[row1][from_column])
                       and not given.is_block_duplicate(row1, from_column, self.values[row2][to_column])):
                    
                        # Swap values.
                        temp = self.values[row2][to_column]
                        self.values[row2][to_column] = self.values[row1][from_column]
                        self.values[row1][from_column] = temp
                        success = True
    
        return success

class GivenGrid(Candidate):
    def __init__(self, values):
        self.values = values
        return
        
    def is_row_duplicate(self, row, value):
        """ Check whether there is a duplicate of a FIXED value in a row. """
        for column in range(0, Nd):
            if(self.values[row][column] == value):
               return True
        return False

    def is_column_duplicate(self, column, value):
        """ Check whether there is a duplicate of a FIXED value in a column. """
        for row in range(0, Nd):
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

class Tournament(object):

    def __init__(self):
        return
        
    def compete(self, candidates):
        """ Pick 2 random candidates from the population and get them to compete against each other. """
        c1 = candidates[random.randint(0, len(candidates)-1)]
        c2 = candidates[random.randint(0, len(candidates)-1)]
        f1 = c1.fitness
        f2 = c2.fitness

        # Find the fittest and the weakest.
        if(f1 > f2):
            fittest = c1
            weakest = c2
        else:
            fittest = c2
            weakest = c1

        selection_rate = 0.85
        r = random.uniform(0, 1.1)
        while(r > 1): # Outside [0, 1] boundary. Choose another.
            r = random.uniform(0, 1.1)
        if(r < selection_rate):
            return fittest
        else:
            return weakest
        
class Sudoku(object):
    def __init__(self):
        self.given = None
        return
    
    def load(self, path):
        # Load a configuration to solve.
        with open(path, "r") as f:
            values = numpy.loadtxt(f).reshape((Nd, Nd)).astype(int)
            self.given = GivenGrid(values)
        return
    
    def solve(self):
        Np = 10  # Population size.
        Ne = 2  # Number of elites.
        Ng = 2  # Number of generations.
        Nm = 0  # Number of mutations.
        
        # Mutation parameters.
        phi = 0
        sigma = 1
        mutation_rate = 0.06
    
        # Create an initial population.
        self.population = Population()
        self.population.seed(Np, self.given)
    
        # For up to 10000 generations...
        stale = 0
        for generation in range(0, Ng):
            
            # Check for a solution.
            for c in range(0, Np):
                if(self.population.candidates[c].fitness == 1):
                    print("Solution found at generation %d!" % generation)
                    print(self.population.candidates[c])
                    return self.population.candidates[c]

            next_population = []

            # Select elites.
            self.population.sort()
            elites = []
            for e in range(0, Ne):
                elite = Candidate()
                elite.values = numpy.copy(self.population.candidates[e].values)
                elites.append(elite)

            # If no solution present, create the next population.
            for count in range(Ne, Np, 2):
                # Select parents from population via a tournament.
                t = Tournament()
                parent1 = t.compete(self.population.candidates)
                parent2 = t.compete(self.population.candidates)
                
                ## Cross-over.
                #child1, child2 = CycleCrossover(parent1, parent2)
                import copy
                child1 = copy.deepcopy(parent1)
                child2 = copy.deepcopy(parent2)
                
                # Mutate child1.
                old_fitness = child1.fitness
                success = child1.mutate(mutation_rate, self.given)
                if(success):
                    Nm += 1
                    if(child1.fitness > old_fitness):  # Used to calculate the relative success rate of mutations
                        phi = phi + 1
                
                # Mutate child2.
                old_fitness = child2.fitness
                success = child2.mutate(mutation_rate, self.given)
                if(success):
                    Nm += 1
                    if(child2.fitness > old_fitness):  # Used to calculate the relative success rate of mutations
                        phi = phi + 1
                
                ## Add children to new population
                next_population.append(child1)
                next_population.append(child2)

            # Append elites onto the end of the population. These will not have been affected by crossover or mutation.
            for e in range(0, Ne):
                next_population.append(elites[e])
            
            
            # Select next generation.
            self.population.candidates = next_population
            
            # Calculate new adaptive mutation rate (based on Rechenberg's 1/5 success rule).
            if(Nm == 0):
                phi = 0  # Avoid divide by zero.
            else:
                phi = phi / Nm
            
            if(phi > 0.2):
                sigma = sigma/0.998
            elif(phi < 0.2):
                sigma = sigma*0.998

            mutation_rate = abs(numpy.random.normal(loc=0.0, scale=sigma, size=None))
        
            # Check for stale population.
            self.population.sort()
            if(self.population.candidates[0].fitness != self.population.candidates[1].fitness):
                stale = 0
            else:
                stale += 1

            # Kill off the population if 100 generations have passed without a change in their best fitness.
            if(stale >= 100):
                print "Earth Quake!"
                self.population.seed(Np, self.given)
                stale = 0
                sigma = 1
                phi = 0
                mutations = 0
                mutation_rate = 0.06
        
        
s = Sudoku()
s.load("puzzle.txt")
s.solve()
