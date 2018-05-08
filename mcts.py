# Monte-Carlo Tree Search to play simple 2 player board games. Has both pure MCTS, and MCTS with value networks.
# Run this file to play the games with trained models.
# See README for usage.
#
# Authors: Prithvijit Chakrabarty (prithvichakra@gmail.com), Ankur Tomar (atomar@umass.edu)

import sys
import numpy as np
from matplotlib import pyplot as plt

from connect4 import Connect4

class MCTS():
    def __init__(self,board,mode='play',model_path=None):
        self.board = board
        self.scores = {}
        self.mode = mode
        self.model_path = model_path
        self.use_nn = False
        self.model = None
        if not (model_path is None):
            self.use_nn = True
            self.model = Q(arch)
            self.model.load(model_path)

    #Convert board state to a hashable key
    def board2key(self,state):
        return ''.join(str(state.flatten())[1:-1].replace(' ','').replace('.','').replace('\n',''))

    #Convert board state to multiple binary channels
    def board2vec(self,state):
        pieces = set(map(int,state.flatten()))
        n_pieces = len(pieces)
        unit = np.diag(np.ones(n_pieces))
        vec = np.zeros((n_pieces,state.shape[0],state.shape[1]))
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                vec[:,i,j] = unit[int(state[i,j])]
        return vec

    #Find the best move for the AI
    def best_move(self,save_score=True):
        print 'Thinking...'
        allowed_moves = self.board.possible_moves()
        pid = self.board.current_player
        cur_state = self.board.state.copy()
        #Explore low conf. children
        thresh = 500
        mscores = []
        for m in allowed_moves:
            #Cache current player and board state
            self.board.current_player = pid
            self.board.state = cur_state.copy()
            
            #Find child node
            self.board.play_move(m)
            child = self.board.state.copy()
            child_key = self.board2key(child)
            
            if self.use_nn == True:
                #Prediction with the trained neural net model
                s = self.model.predict(self.board2vec(self.board.state))[0]
                print s
                if False: #(s >= 0.3) & (s <= 0.5):
                    self.explore(child,pid,save_score=save_score)
                    nw,nt = self.scores[child_key]
                    c = 1
                    s = nw/float(nt)+c*np.sqrt(np.log(100)/nt)
            else:
                #Prediction with dynamic MCTS tree expansion for exploration
                if (not (child_key in self.scores)):
                    self.explore(child,pid,save_score=save_score)
                elif (self.scores[child_key][1] < thresh):
                    self.explore(child,pid,save_score=save_score)
                nw,nt = self.scores[child_key]
                c = 1
                s = nw/float(nt)+c*np.sqrt(np.log(100)/nt)
            
            #Restore board state and current player ID
            self.board.current_player = pid
            self.board.state = cur_state
            mscores.append(s)
        #If generating training data, add diversity: uniform exploration with probability 0.5
        #Otherwise, move to the state with the highest score.
        if len(mscores) > 0:
            if pid == 2:
                if self.mode == 'dgen':
                    if np.random.random() < 0.5:
                        next_move = allowed_moves[np.random.choice(range(len(allowed_moves)))]
                    else:
                        next_move = allowed_moves[np.argmax(mscores)]
                elif self.mode == 'play':
                    next_move = allowed_moves[np.argmax(mscores)]
            elif pid == 1:
                if self.mode == 'dgen':
                    if np.random.random() < 0.5:
                        next_move = allowed_moves[np.random.choice(range(len(allowed_moves)))]
                    else:
                        next_move = allowed_moves[np.argmin(mscores)]
                elif self.mode == 'play':
                    next_move = allowed_moves[np.argmin(mscores)]
        else:
            next_move = -1
        return next_move

    #Explore the subtree below the current state
    def explore(self,start_state,pid,save_score=True):
        n = 120
        cur_state = self.board.state.copy()
        for _ in range(n):
            #Cache current state and player ID
            self.board.current_player = pid
            self.board.state = cur_state.copy()
            state_seq,w = self.simulate(start_state,pid)
            #Compute state dict.: map state -> [number of wins,number of visits]
            for s in state_seq:
                if not (s in self.scores):
                    if save_score == True:
                        self.scores.update({s : [0.,0.]})
                else:
                    if w == 2:
                        self.scores[s][0] += 1.
                    self.scores[s][1] += 1.
            #Restore state and player ID
            self.board.current_player = pid
            self.board.state = start_state

    #Simulate a game playout with random moves
    def simulate(self,start_state,pid):
        state_seq = []
        w = 0
        #Cache current state and player ID
        self.board.state = start_state.copy()
        self.current_player = pid
        state_seq.append(self.board2key(self.board.state))
        while True:
            allowed = self.board.possible_moves()
            if len(allowed) == 0:
                #Draw: no further moves, no winner
                return (state_seq,0)
            #Pick a random move
            m = np.random.choice(allowed)
            self.board.play_move(m)
            w = c.winner()
            state_seq.append(self.board2key(self.board.state))
            if w != 0:
                break
        #Restore current state and player ID
        self.board.current_player = pid
        self.board.state = start_state
        return (state_seq,w)

    #Run the game: play with the user
    def play(self):
        width,height = (300,500)
        if graphics == True:
            pygame.init()
            screen = pygame.display.set_mode((width, height))
            pygame.display.update()
        while True:
            self.board.print_board()
            #User
            m1 = None
            print 'Your move!'
            print 'Enter column number (0-4):',
            while not isinstance(m1,int):
                inp = raw_input('').strip()
                try:
                    m1 = int(inp)
                except ValueError:
                    print 'Enter column number (0-4):',
                if (m1 < 0) | (m1 > self.board.state.shape[1]):
                    m1 = None
                    print 'Enter column number (0-4):',
            self.board.play_move(m1)
            self.board.print_board()
            if graphics == True:
                self.board.render(screen)
            #Ai
            m2 = self.best_move()
            self.board.play_move(m2)
            if graphics == True:
                self.board.render(screen)
            w = c.winner()
            if w != 0:
                print 'Game over'
                self.board.print_board()
                print 'Winner:',w
                if graphics == True:
                    k = raw_input('press enter to end...')
                break

    #Generate training data: play many games and save the scores dict. in dsfile
    def datagen(self,nsamples,dsfile):
        key2vec = {}
        ds = []
        for ns in range(nsamples):
            #Reset game
            self.board.reset()
            gameover = False
            while not gameover:
                self.board.print_board()
                #Ai player 1
                m1 = self.best_move(save_score=True)
                self.board.play_move(m1)
                self.board.print_board()
                key2vec.update({self.board2key(self.board.state) : self.board2vec(self.board.state)})
                #Ai player 2
                m2 = self.best_move()
                if m2 == -1:
                    print 'Game over.'
                    self.board.print_board()
                    print '\tWinner: 0'
                    gameover = True
                self.board.play_move(m2)
                w = c.winner()
                if w != 0:
                    print 'Game over.'
                    self.board.print_board()
                    print '\tWinner:',w
                    gameover = True
            print ns
        #Collect state values from the samples
        min_visit = 20
        for st in key2vec.keys():
            w,n = self.scores[st]
            k = key2vec[st]
            if n > min_visit:
                ds.append((k,w/n))
                print st,k.shape,w/n
        x,y = map(np.array,zip(*ds))
        print 'Dataset created!'
        print x.shape,y.shape
        print 'ds size:',len(ds)
        np.save(dsfile,np.array(ds))

#####***** PARAM CONTROLS HERE *****#####
arch = {'in_d':[3,5,5],
        'nfilt':25,
        'h1':200,
        'h2':200
       }
lr = 1e-4
eps = 1e-8
bz = 5
epch = 100
tc = 0.8
dsfile = './model/conn4data.npy'
model_path = './model'
#########################################

#Command line option parsing
use_nn = False
if len(sys.argv) >= 3:
    if (sys.argv[2] == 'nn'):
        use_nn = True
        from vmodel import Q
    else:
        model_path = None
else:
    model_path = None

graphics = False
if len(sys.argv) >= 4:
    if (sys.argv[3] == 'graphics'):
        graphics = True
        import pygame

c = Connect4(5,graphics=graphics)
if sys.argv[1] == 'dgen':
    m = MCTS(c,mode='dgen')
    n_games = 2000
    m.datagen(n_games,dsfile)

elif sys.argv[1] == 'play':
    m = MCTS(c,mode='play',model_path=model_path)
    m.play()

elif sys.argv[1] == 'train':
    mpath = './model'
    q = Q(arch)
    q.lr = lr
    q.eps = eps
    q.bz = bz
    q.tc = tc
    q.epch = epch
    q.train(dsfile,mpath)
