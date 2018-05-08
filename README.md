# Monte-Carlo-Tree-Search

This module implements Monte Carlo Tree Search to play 2 player board games. As demonstration, we've implemented Connect4. The program is loosely based on the [AlphaGo paper](https://www.nature.com/articles/nature16961) (though heavily simplified). There are 2 versions of the algorithm:

1. Pure MCTS: This is the basic MCTS algorithm. It uses lookahead for every move, i.e., every time the computer must play a move, it expands the game tree and runs simulations in the subtree to identify the best child state (then moves to reach that state).

2. MCTS-NN: This is MCTS coupled with a neural network which serves as the value network. The neural net model maps states to a "value" in [0,1] (likelihood of winning from that state). The model is trained by making the program play against itself, for 2000 games.

**Usage**

1. Pure MCTS, without graphics:
        
        python mcts.py pure
      
2. MCTS-NN, without graphics:

        python mcts.py nn
        
Add the "graphics" option to display PyGame graphics: `python mcts.py pure graphics` or `python mcts.py nn graphics`.
