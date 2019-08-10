import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
from matplotlib.text import TextPath
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#
# Types of problems to handle
# https://www.rangevoting.org/AssetBC.html
# https://groups.google.com/forum/#!topic/electionscience/Rk4ZGf-s-s8

#centerists = False
centerists = True
#maximize = False
maximize = True

#Number of winners
W = 5

#the maximum possible score is 5
K = 5.0

mean_red = [-2.5*np.sqrt(3), 2.5]
cov_red = [[5, 0], [0, 5]]  # diagonal covariance
pos_red = np.random.multivariate_normal(mean_red, cov_red, 5000)
df_red = pd.DataFrame.from_records(pos_red, columns = ['x','y'])
df_red['colour'] = 'red'

mean_green = [0, -5]
cov_green = [[5, 0], [0, 5]]  # diagonal covariance
pos_green = np.random.multivariate_normal(mean_green, cov_green, 5000)
df_green = pd.DataFrame.from_records(pos_green, columns = ['x','y'])
df_green['colour'] = 'green'

mean_blue = [2.5*np.sqrt(3), 2.5]
cov_blue = [[5, 0], [0, 5]]  # diagonal covariance
pos_blue = np.random.multivariate_normal(mean_blue, cov_blue, 5000)
df_blue = pd.DataFrame.from_records(pos_blue, columns = ['x','y'])
df_blue['colour'] = 'blue'

candidates = [['A',0,0],
                ['Z',0,2.5],
                ['R1',-1*np.sqrt(3), 1],
                ['R2',-2.5*np.sqrt(3), 2.5],
                ['R3',-4*np.sqrt(3), 4],
                ['G1',0, -2],
                ['G2',0, -5],
                ['G3',0, -8],
                ['B1',1*np.sqrt(3), 1],
                ['B2',2.5*np.sqrt(3),2.5],
                ['B3',4*np.sqrt(3), 4]]

df_can = pd.DataFrame.from_records(candidates, columns = ['Name','x','y'] )

fig = plt.figure(figsize=(20,10))
fig.suptitle('Political Simulation')

#image
ax1 = fig.add_subplot(1, 2, 1)
img=mpimg.imread('Political Compass.jpg')
ax1.imshow(img)
ax1.axis('off')

# Scatter plot
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(df_red['x'],df_red['y'],".",label = 'Red', color='r')
ax2.plot(df_green['x'],df_green['y'],".",label = 'Green', color='g')
ax2.plot(df_blue['x'],df_blue['y'],".",label = 'Blue', color='b')

#Candidates
for c in candidates:
    ax2.plot(c[1], c[2],marker=TextPath((0,0), c[0]),markersize=20, color='k')

ax2.set_xlim(-10, 10)
ax2.set_ylim(-10, 10)
ax2.set_title('Political Compass')
ax2.set_xlabel('Planned Economy  <--  Economics  -->  Free Market')
ax2.set_ylabel('Liberal  <-- Government  --> Authoritarian')
lgd2 = ax2.legend(loc=1)
fig.savefig("Simulated_Spectrum", dpi=300)

if centerists:
    mean_center = [0,0]
    cov_center = [[5, 0], [0, 5]]  # diagonal covariance
    pos_center = np.random.multivariate_normal(mean_center, cov_center, 3500)
    df_center = pd.DataFrame.from_records(pos_center, columns = ['x','y'])
    df_center['colour'] = 'center'

    df_voters = pd.concat([df_red, df_green, df_blue, df_center],ignore_index=True)
else:
    df_voters = pd.concat([df_red, df_green, df_blue],ignore_index=True)


#Number of voters
V = df_voters.shape[0]

#Make 3d plot of df_voters
fig2 = plt.figure(figsize=(15,20))
fig2.suptitle('Voter Density')

#histogram
axa = fig2.add_subplot(211, projection='3d')
hist, xedges, yedges = np.histogram2d(df_voters['x'], df_voters['y'], bins=20, range=[[-10, 10], [-10, 10]])
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

axa.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
axa.set_xlabel('Economic')
axa.set_ylabel('Government')
axa.set_zlabel('Voter Count')
axa.view_init(60, 240)

#surface
X, Y = np.meshgrid(xedges[:-1] + 0.5, yedges[:-1] + 0.5, indexing="ij")

axb = fig2.add_subplot(212, projection='3d')
surf = axb.plot_surface(X, Y, hist, cmap="jet", linewidth=0, antialiased=False)
surf.set_edgecolors(surf.to_rgba(surf._A))
#surf.set_facecolors("white")
#cset = axb.contour(X, Y, hist, zdir='z', offset=-100, cmap=cm.coolwarm)
#cset = axb.contour(X, Y, hist, zdir='x', offset=-40, cmap=cm.coolwarm)
#cset = axb.contour(X, Y, hist, zdir='y', offset=40, cmap=cm.coolwarm)
axb.set_xlabel('Economic')
axb.set_ylabel('Government')
axb.set_zlabel('Voter Count')
axb.view_init(60, 240)
fig2.colorbar(ax = axb, mappable = surf, shrink=0.5, aspect=5)
fig2.savefig("3D_Population", dpi=300)

distance = pd.DataFrame()
S = pd.DataFrame()

for c in candidates:
    distance[c[0]] = df_voters[['x', 'y']].sub(np.array([c[1], c[2]])).pow(2).sum(1).pow(0.5)
    S[c[0]] = round(np.clip(K - distance[c[0]], 0, np.inf))

#rowwise max set to 5
if maximize:
    columns = distance.idxmin('columns')
    for index in S.index:
        S.loc[index,columns[index]] = 5

#define seleciton algorithm
def get_winners(S_in,Selection = 'default'):
    #print(Selection)
    score_remaining = np.ones(V)
    winner_list = []
    while len(winner_list) < W:

        #select winner
        if Selection == 'Monroe':
            sum_scores = pd.DataFrame.from_records(np.sort(S_in.values, axis=0), columns = S_in.columns).tail(round(V/W)).sum()
        else:
            sum_scores = S_in.sum()

        #print( sum_scores)

        #w = index with highest sum_scores
        w = sum_scores.idxmax()

        #print(w)

        winner_list.append(w)
        surplus_factor = max( sum_scores[w] *W/V , 1)

        #Score spent on each winner by each voter
        score_spent = S_in[w]/ surplus_factor

        #Total score left to be spent by each voter
        score_remaining = np.clip(score_remaining-score_spent,0,1)

        #Update Ballots
        #set scores to zero for winner so they don't win again
        #S_in[w]=0
        #Take score off of ballot (ie reweight)

        for c in S_in.columns:
            S_in[c] = pd.DataFrame([S_in[c], score_remaining]).min()

    return winner_list

default_winners = get_winners(S_in=S.divide(K).copy(),Selection = 'default')
monroe_winners = get_winners(S_in=S.divide(K).copy(),Selection = 'Monroe')


print('Utilitarian Winner set is:')
print(default_winners)

print('Monroe Winner set is:')
print(monroe_winners)

plt.show()