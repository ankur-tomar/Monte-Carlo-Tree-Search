import numpy as np

from connect4 import Connect4

class MCTS():
    def __init__(self,board):
        self.board = board
        self.scores = {}

    def board2key(self,state):
        return ''.join(str(state.flatten())[1:-1].replace(' ','').replace('.','').replace('\n',''))

    def best_move(self):
        print 'Thinking...'
        allowed_moves = self.board.possible_moves()
        #rand_move = np.random.choice(allowed_moves)
        pid = self.board.current_player
        cur_state = self.board.state.copy()
        #Explore low conf. children
        thresh = 20
        mscores = []
        for m in allowed_moves:
            self.board.current_player = pid
            self.board.state = cur_state.copy()
            
            #print self.board.state,']]]]]'

            self.board.play_move(m)
            #self.board.print_board()
            child = self.board.state.copy()
            child_key = self.board2key(child)
            #print '>>>',child
            #if child in self.scores:
            
            if (not (child_key in self.scores)):
                self.explore(child,pid)
            elif (self.scores[child_key][1] < thresh):
                self.explore(child,pid)
            
            self.board.current_player = pid
            self.board.state = cur_state
            
            #print cur_state,'++++++'

            #print self.scores
            nw,nt = self.scores[child_key]
            c = 1
            s = nw/float(nt)+c*np.sqrt(np.log(100)/nt)
            mscores.append(s)
        next_move = allowed_moves[np.argmax(mscores)]

        print self.board.state

        return next_move

    def explore(self,start_state,pid):
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
                    self.scores.update({s : [0.,0.]})
                else:
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
        while True:
            self.board.print_board()
            #User
            m1 = int(raw_input('Play your move:'))
            self.board.play_move(m1)
            self.board.print_board()
            #Ai
            m2 = self.best_move()
            self.board.play_move(m2)
            w = c.winner()
            if w != 0:
                print 'Game over'
                self.board.print_board()
                print 'Winner:',w
                break 

c = Connect4(5)
m = MCTS(c)
#m.simulate()
m.play()
