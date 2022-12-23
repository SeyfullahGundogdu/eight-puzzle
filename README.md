## 8-puzzle solver in Rust!
8-puzzle is a game in which you have a matrix of 3x3, 8 pieces randomly placed with an additional empty tile.

In this app, the number 0 is considered an empty tile.
Our goal matrix has the following layout:  
| 0 1 2 |  
| 3 4 5 |  
| 6 7 8 |

There are number of methods for solving this problem. Namely,
* [BFS](https://en.wikipedia.org/wiki/Breadth-first_search)
* [DFS](https://en.wikipedia.org//wiki/Depth-first_search)
* [A*](https://en.wikipedia.org/wiki/A*_search_algorithm)  

We have 2 heuristic functions that we use for A* algorithm, one of them(h1) is simply counting the number of pieces which are wrongly placed, and h2 algorithm calculates the manhattan distance for each piece, and uses their sum for choosing the best candidate. The lower the sum, the closer we are to the solution.  
However, we _can't_ solve 8 puzzle problem for any arbitrary initial and goal state.

DFS algorithm is unimplemented (for now).
