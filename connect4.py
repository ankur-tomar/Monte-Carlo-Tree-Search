import numpy as np

class Connect4():
    def __init__(self,N):
        self.N = N
        self.state = np.zeros((N,N))
        self.current_player = 1
        self.n_players = 2
    
    def possible_moves(self):
        player = self.current_player
        allowed = [c for c in range(self.N) if (0 in self.state[:,c])]
        return allowed

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

"""c = Connect4(5)
c.print_board()
c.play_move(1,1)
c.play_move(0,2)
c.play_move(2,1)
c.play_move(1,2)
c.play_move(2,1)
c.play_move(2,2)
c.play_move(3,1)
c.play_move(3,1)
c.play_move(3,1)
c.play_move(3,2)
c.play_move(3,2)
c.play_move(3,2)
c.print_board()
print 'Winner',c.winner()
print c.possible_moves()
"""
