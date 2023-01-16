import matplotlib.pyplot as plt 

N = 10

# Everything is a wall
G = [[1 for _ in range(N)] for _ in range(N)] # Everything is a wall

# Accessible boxes
for j in range(N):
	G[0][j] = 0 # Accessible cases

# Accessible boxes
for i in range(3,N,2):
	for j in range(i):
		G[j+1][i] = 0
	if i < N-1 and j < N-1: G[j+1][i+1] = 0

# Definite init place, first reward, second reward after 25 eisodes
init = (0,1)
rwd = (N-1,N-1)
rwd2 = (0,0)

G[init[0]][init[1]] = "I"
G[rwd[0]][rwd[1]] = "R1"
G[rwd2[0]][rwd2[1]] = "R2"

# Show the grid
for i in range(N):
	for j in range(N):
		print(G[i][j],end=" ")
	print()