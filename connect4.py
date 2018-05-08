# Module implementing the game connect4.
#
# Authors: Prithvijit Chakrabarty (prithvichakra@gmail.com), Ankur Tomar (atomar@umass.edu)

import numpy as np

class Connect4():
    def __init__(self,N,graphics=False):
        self.N = N
        self.n_players = 2
        self.state = np.zeros((N,N))
        self.current_player = 1
        self.graphics = graphics
        if graphics == True:
            import pygame

    def reset(self):
        self.state = np.zeros((self.N,self.N))
        self.current_player = 1
    
    #Return possible moves from current state
    def possible_moves(self):
        player = self.current_player
        allowed = [c for c in range(self.N) if (0 in self.state[:,c])]
        return allowed

    #Play a move
    def play_move(self,col):
        p = self.current_player
        for i in range(self.N-1,-1,-1):
            if self.state[i,col] == 0:
                self.state[i,col] = p
                break
        if p == 1:
            self.current_player = 2
        elif p == 2:
            self.current_player = 1

    def print_board(self):
        print 'Board:'
        print self.state

    #Identify winners in current state
    def winner(self):
        N = self.N
        for i in range(N):
            for j in range(N):
                x = self.state[i,j]
                v1 = self.state[i:i+4,j]
                v2 = self.state[max(0,i-3):i+1,j]
                h1 = self.state[i,j:j+4]
                h2 = self.state[i,max(0,j-3):j+1]
                d1,d2,d3,d4 = [],[],[],[]
                if min(N-i,N-j) >= 4:
                    d1 = self.state[np.arange(i,i+4),np.arange(j,j+4)]
                if min(i+1,N-j) >= 4:
                    d2 = self.state[np.arange(i-3,i+1),np.arange(j+3,j-1,-1)]
                if min(N-i,j+1) >= 4:
                    d3 = self.state[np.arange(i+3,i-1,-1),np.arange(j-3,j+1)]
                if min(i+1,j+1) >= 4:
                    d4 = self.state[np.arange(i-3,i+1),np.arange(j-3,j+1)]
                d = map(np.array,[v1,v2,h1,h2,d1,d2,d3,d4])
                for k in d:
                    if len(k) == 4:
                        if ((k-k.min()).max() == 0) & (k.sum() > 0):
                            #print k
                            return x
        return 0
        
    #Render the state to graphics
    def render(self,screen):
        if self.graphics:
            import pygame
        width,height = (300,500)
        col_list = [(255,255,255),(255,0,0),(0,0,255),(0,0,0)]
        background = col_list[-1]
        screen.fill(background)
        sx = 100
        sy = 60
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                col = col_list[int(self.state[i,j])]
                pygame.draw.circle(screen,col,(j*sy+25,i*sx+25),20)
        pygame.display.update()
