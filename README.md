# Overview

This project is capable of solving a Sudoku puzzle using a genetic algorithm. Puzzle configurations are read in from a plain text file containing a string of 9 x 9 digits separated by spaces, with an example provided in the file `puzzle_mild.txt`. Zero is used to represent an unknown digit, whereas numbers in [1, 9] are assumed to be known/given.

Run the code by executing `python sudoku.py` at the command line. Example output looks like this:

```
~/sudoku-genetic-algorithm $ python sudoku.py 
Seeding complete.
Generation 0
Best fitness: 0.555556
Generation 1
Best fitness: 0.555556
Generation 2
Best fitness: 0.629630
Generation 3
Best fitness: 0.629630
Generation 4
Best fitness: 0.777778
Generation 5
Best fitness: 0.851852
Generation 6
Best fitness: 0.851852
Generation 7
Best fitness: 0.851852
Generation 8
Best fitness: 0.851852
Generation 9
Best fitness: 0.851852
Generation 10
Best fitness: 0.851852
Generation 11
Best fitness: 0.851852
Generation 12
Best fitness: 0.851852
Generation 13
Best fitness: 0.851852
Generation 14
Best fitness: 0.851852
Generation 15
Best fitness: 0.851852
Generation 16
Best fitness: 0.851852
Generation 17
Best fitness: 0.851852
Generation 18
Best fitness: 0.851852
Generation 19
Solution found at generation 19!
[[ 8.  3.  9.  2.  7.  4.  1.  5.  6.]
 [ 5.  2.  4.  1.  3.  6.  7.  8.  9.]
 [ 7.  6.  1.  5.  9.  8.  4.  2.  3.]
 [ 1.  9.  7.  8.  5.  3.  2.  6.  4.]
 [ 6.  5.  3.  4.  9.  2.  1.  8.  7.]
 [ 2.  4.  8.  6.  1.  7.  9.  3.  5.]
 [ 4.  5.  2.  7.  6.  1.  8.  9.  3.]
 [ 9.  6.  7.  3.  8.  5.  1.  4.  2.]
 [ 3.  1.  8.  9.  2.  4.  6.  7.  5.]]
```
