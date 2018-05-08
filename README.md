# Monte-Carlo-Tree-Search

This module implements Monte Carlo Tree Search to play 2 player board games. As demonstration, we've implemented Connect4. The program is loosely based on the [AlphaGo paper](https://www.nature.com/articles/nature16961) (though heavily simplified). There are 2 versions of the algorithm:

1. Pure MCTS: This is the basic MCTS algorithm. It uses lookahead for every move, i.e., every time the computer must play a move, it expands the game tree and runs simulations in the subtree to identify the best child state (then moves to reach that state).

2. MCTS-NN: This is MCTS coupled with a neural network which serves as the value network. The neural net model maps a state to a "value" in [0,1] (likelihood of winning from that state).

**Usage**

1. Pure MCTS, without graphics:
        
        python mcts.py pure
      
2. MCTS-NN, without graphics:

        python mcts.py nn
        
Add the "graphics" option to display PyGame graphics: `python mcts.py pure graphics` or `python mcts.py nn graphics`.

**Dataset Creation and Model Training**

The model is trained by making the program play against itself, for 2000 games. To create a new dataset (maybe by playing more games to train a better model):

1. Set the path to save the dataset as `dsfile` (edit line 249 in mcts.py)

2. Set the number games to play to create the dataset (set `n_games_for_ds`,line 248 in mcts.py)

3. Run: `python mcts.py dgen nn`

This will create a new dataset and save it in the path specified in step 1. The default dataset ('./model/conn4data.npy') was gneerated from 2000 simulated games.

With the new dataset, a new model may be trained for better performance. To do this, modify the model parameters as required in mcts.py (line 237). To train the new model,

1. Set the path to save the model as `model_path` (edit line 250 in mcts.py)

2. Run: `python mcts.py train nn`
