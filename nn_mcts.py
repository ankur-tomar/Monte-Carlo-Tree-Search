import sys
import numpy as np
from matplotlib import pyplot as plt

from connect4 import Connect4
from vmodel import Q

class MCTS():
    def __init__(self,board,mode='play',model_path=None):
        self.board = board
        self.scores = {}
        self.mode = mode
        self.model_path = model_path
        self.use_nn = False
        self.model = None
        if model_path != None:
            self.use_nn = True
            self.model = Q({'in_d':75,'h1':30,'h2':30})
            self.model.load(model_path)

    def board2key(self,state):
        return ''.join(str(state.flatten())[1:-1].replace(' ','').replace('.','').replace('\n',''))

    def board2vec(self,state):
        pieces = set(map(int,state.flatten()))
        n_pieces = 3
        unit = np.diag(np.ones(n_pieces))
        vec = np.zeros((n_pieces,state.shape[0],state.shape[1]))
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                vec[:,i,j] = unit[int(state[i,j])]
        return vec

    def best_move(self,save_score=True):
        print 'Thinking...'
        allowed_moves = self.board.possible_moves()
        #rand_move = np.random.choice(allowed_moves)
        pid = self.board.current_player
        cur_state = self.board.state.copy()
        #Explore low conf. children
        thresh = 500 #20
        mscores = []
        for m in allowed_moves:
            self.board.current_player = pid
            self.board.state = cur_state.copy()
            
            self.board.play_move(m)
            child = self.board.state.copy()
            child_key = self.board2key(child)
            
            if self.use_nn == True:
                s = self.model.predict(self.board2vec(self.board.state))
                print s
                if (s >= 0.3) & (s <= 0.6):
                    self.explore(child,pid,save_score=save_score)
                    #self.board.current_player = pid
                    #self.board.state = cur_state
                    nw,nt = self.scores[child_key]
                    c = 1
                    s = nw/float(nt)+c*np.sqrt(np.log(100)/nt)
            else:
                if (not (child_key in self.scores)):
                    self.explore(child,pid,save_score=save_score)
                elif (self.scores[child_key][1] < thresh):
                    self.explore(child,pid,save_score=save_score)
                #self.board.current_player = pid
                #self.board.state = cur_state
                nw,nt = self.scores[child_key]
                c = 1
                s = nw/float(nt)+c*np.sqrt(np.log(100)/nt)
            self.board.current_player = pid
            self.board.state = cur_state
            mscores.append(s)
        if len(mscores) > 0:
            if pid == 2:
                if self.mode == 'dgen':
                    if np.random.random() < 0.5:
                        next_move = allowed_moves[np.random.choice(range(len(allowed_moves)))]
                    else:
                        next_move = allowed_moves[np.argmax(mscores)]
                else:
                    next_move = allowed_moves[np.argmax(mscores)]
            elif pid == 1:
                if self.mode == 'dgen':
                    if np.random.random() < 0.5:
                        next_move = allowed_moves[np.random.choice(range(len(allowed_moves)))]
                    else:
                        next_move = allowed_moves[np.argmin(mscores)]
                else:
                    next_move = allowed_moves[np.argmin(mscores)]
        else:
            next_move = -1

        #print self.board.state

        return next_move

    def explore(self,start_state,pid,save_score=True):
        n = 100
        #pid = self.board.current_player
        cur_state = self.board.state.copy()
        for _ in range(n):
            self.board.current_player = pid
            self.board.state = cur_state.copy()
            
            state_seq,w = self.simulate(start_state,pid)
            #a = raw_input('SIM DONE')
            for s in state_seq:
                if not (s in self.scores):
                    if save_score == True:
                        self.scores.update({s : [0.,0.]})
                else:
                    #if self.mode == 'play':
                    #    wp = 2
                    #elif self.mode == 'dgen':
                    #    wp = pid
                    if w == 2:
                        self.scores[s][0] += 1.
                    self.scores[s][1] += 1.
                    #print '>>>',self.scores[s]
            
            self.board.current_player = pid
            self.board.state = start_state

    def simulate(self,start_state,pid):
        state_seq = []
        w = 0
        self.board.state = start_state.copy()
        self.current_player = pid

        state_seq.append(self.board2key(self.board.state))
        while True:
            #m = self.best_move()
            allowed = self.board.possible_moves()
            if len(allowed) == 0:
                #Draw
                return (state_seq,0)
            
            #Use NN here
            m = np.random.choice(allowed)

            self.board.play_move(m)
            w = c.winner()
            state_seq.append(self.board2key(self.board.state))
            if w != 0:
                #print 'Simulation done:'
                #self.board.print_board()
                #print self.board2key()
                #print 'Winner:',w
                break

        self.board.current_player = pid
        self.board.state = start_state

        return (state_seq,w)

    def play(self):
        
        width,height = (300,500)
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.update()
        
        while True:
            self.board.print_board()
            #User
            m1 = int(raw_input('Play your move:'))
            self.board.play_move(m1)
            self.board.print_board()

            self.board.render(screen)

            #Ai
            m2 = self.best_move()
            self.board.play_move(m2)
            
            self.board.render(screen)
            
            w = c.winner()
            if w != 0:
                print 'Game over'
                self.board.print_board()
                print 'Winner:',w
                break 

    def datagen(self,nsamples):
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
                #allowed = self.board.possible_moves()
                #m1 = np.random.choice(allowed)
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
        for st in key2vec.keys():
            w,n = self.scores[st]
            k = key2vec[st]
            ds.append((k,w/n))
            print st,k.shape,w/n
        x,y = map(np.array,zip(*ds))
        print x.shape,y.shape
        print 'ds size:',len(ds)
        np.save('./conn4data.npy',np.array(ds))

import pygame
c = Connect4(5)
if sys.argv[1] == 'dgen':
    m = MCTS(c,mode='dgen')
    m.datagen(500)
elif sys.argv[1] == 'play':
    m = MCTS(c,mode='play',model_path='./model')
    m.play()
