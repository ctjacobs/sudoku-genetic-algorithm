# Overview

This project is capable of solving a Sudoku puzzle using a genetic algorithm. Puzzle configurations are read in from a plain text file containing a string of 9 x 9 digits separated by spaces, with an example provided in the file `puzzle_mild.txt`. Zero is used to represent an unknown digit, whereas numbers in [1, 9] are assumed to be known/given.

Run the code by executing `python sudoku.py` at the command line. Example output looks like this:

```
~/sudoku-genetic-algorithm $ python sudoku.py 
Seeding complete.
Generation 0
Best fitness: 0.286694
Generation 1
Best fitness: 0.303155
Generation 2
Best fitness: 0.303155
Generation 3
Best fitness: 0.303155
Generation 4
Best fitness: 0.303155
Generation 5
Best fitness: 0.390947
Generation 6
Best fitness: 0.473251
Generation 7
Best fitness: 0.473251
Generation 8
Best fitness: 0.473251
Generation 9
Best fitness: 0.473251
Generation 10
Best fitness: 0.473251
Generation 11
Best fitness: 0.489712
Generation 12
Best fitness: 0.489712
Generation 13
Best fitness: 0.489712
Generation 14
Best fitness: 0.489712
Generation 15
Best fitness: 0.489712
Generation 16
Best fitness: 0.536351
Generation 17
Best fitness: 0.536351
Generation 18
Best fitness: 0.536351
Generation 19
Best fitness: 0.536351
Generation 20
Best fitness: 0.536351
Generation 21
Best fitness: 0.536351
Generation 22
Best fitness: 0.536351
Generation 23
Best fitness: 0.547325
Generation 24
Best fitness: 0.547325
Generation 25
Best fitness: 0.547325
Generation 26
Best fitness: 0.547325
Generation 27
Best fitness: 0.604938
Generation 28
Best fitness: 0.604938
Generation 29
Best fitness: 0.604938
Generation 30
Best fitness: 0.604938
Generation 31
Best fitness: 0.604938
Generation 32
Best fitness: 0.851852
Generation 33
Best fitness: 0.851852
Generation 34
Best fitness: 0.851852
Generation 35
Best fitness: 0.851852
Generation 36
Best fitness: 0.851852
Generation 37
Solution found at generation 37!
[[ 8.  3.  9.  2.  7.  4.  6.  5.  1.]
 [ 5.  2.  4.  1.  3.  6.  7.  8.  9.]
 [ 7.  6.  1.  5.  8.  9.  4.  2.  3.]
 [ 1.  9.  7.  8.  5.  3.  2.  6.  4.]
 [ 6.  8.  3.  4.  9.  2.  5.  1.  7.]
 [ 2.  4.  5.  6.  1.  7.  9.  3.  8.]
 [ 3.  5.  2.  7.  4.  1.  8.  9.  6.]
 [ 9.  7.  8.  3.  6.  5.  1.  4.  2.]
 [ 4.  1.  6.  9.  2.  8.  3.  7.  5.]]
```
