# Overview

This project is capable of solving a Sudoku puzzle using a genetic algorithm. Puzzle configurations are read in from a plain text file containing a string of 9 x 9 digits separated by spaces, with an example provided in the file `puzzle_mild.txt`. Zero is used to represent an unknown digit, whereas numbers in [1, 9] are assumed to be known/given.

Run the code by executing `python sudoku.py` at the command line. Example output looks like this:

```
~/sudoku-genetic-algorithm $ python sudoku.py 
Seeding complete.
Generation 0
Best fitness: 0.481481
Generation 1
Best fitness: 0.481481
Generation 2
Best fitness: 0.555556
Generation 3
Best fitness: 0.555556
Generation 4
Best fitness: 0.629630
Generation 5
Best fitness: 0.629630
Generation 6
Best fitness: 0.703704
Generation 7
Best fitness: 0.703704
Generation 8
Best fitness: 0.703704
Generation 9
Best fitness: 0.703704
Generation 10
Best fitness: 0.703704
Generation 11
Best fitness: 0.703704
Generation 12
Best fitness: 0.703704
Generation 13
Best fitness: 0.777778
Generation 14
Best fitness: 0.777778
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
 [ 1.  9.  3.  8.  5.  7.  2.  6.  4.]
 [ 6.  5.  8.  4.  3.  2.  1.  9.  7.]
 [ 2.  4.  7.  6.  1.  9.  5.  3.  8.]
 [ 3.  5.  2.  7.  4.  1.  8.  9.  6.]
 [ 9.  7.  8.  3.  6.  5.  1.  4.  2.]
 [ 4.  1.  6.  9.  2.  8.  3.  7.  5.]]
```
