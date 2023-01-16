import numpy as np
class CustomMaze:
    def __init__(self):
        self.maze = []
    def generateMaze(self,N):
        G = [[1 for _ in range(N)] for _ in range(N)]
        
        for j in range(N):
            G[0][j] = 0
        
        for i in range(3,N,2):
            for j in range(i):
                if j == N: break
                G[j+1][i] = 0
            if i < N-1 and j < N-1: G[j+1][i+1] = 0
        
        init = (0,1) 
        rwd = (N-1,N-1)
        rwd2 = (0,0) 
        
        walls = []
        for i in range(N):
            for j in range(N):
                if G[i][j] == 1:
                    walls.append( (i,j) )
        
        d ={'size': (N, N),
        'walls': walls,
        'start_state': np.array([list(init)]),
        'goal_state.s_1': np.array([list(rwd)]),
        'goal_state.s_2': np.array([list(rwd2)]),
        'reward_magnitude.s': np.array([[1]]),
        'reward_std.s': np.array([[0.1]]),
        'reward_prob.s': np.array([[1]])}
        
        return d


