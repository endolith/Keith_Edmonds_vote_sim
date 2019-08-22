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

centerists = False
#centerists = True
#maximize = False
maximize = True

#Number of winners
W = 4.0

#the maximum possible score is 5
K = 5.0

voter_groups = []

#Three parties
# mean_red = [-2.5*np.sqrt(3), 2.5]
# cov_red = [[5, 0], [0, 5]]  # diagonal covariance
# pos_red = np.random.multivariate_normal(mean_red, cov_red, 5000)
# df_red = pd.DataFrame.from_records(pos_red, columns = ['x','y'])
# df_red['colour'] = 'red'
# voter_groups.append(df_red)
# 
# mean_green = [0, -5]
# cov_green = [[5, 0], [0, 5]]  # diagonal covariance
# pos_green = np.random.multivariate_normal(mean_green, cov_green, 5000)
# df_green = pd.DataFrame.from_records(pos_green, columns = ['x','y'])
# df_green['colour'] = 'green'
# voter_groups.append(df_green)
# 
# mean_blue = [2.5*np.sqrt(3), 2.5]
# cov_blue = [[5, 0], [0, 5]]  # diagonal covariance
# pos_blue = np.random.multivariate_normal(mean_blue, cov_blue, 5000)
# df_blue = pd.DataFrame.from_records(pos_blue, columns = ['x','y'])
# df_blue['colour'] = 'blue'
# voter_groups.append(df_blue)
# 
# candidates = [['A',0,0],
#                 ['Z',0,2.5],
#                 ['R1',-1*np.sqrt(3), 1],
#                 ['R2',-2.5*np.sqrt(3), 2.5],
#                 ['R3',-4*np.sqrt(3), 4],
#                 ['G1',0, -2],
#                 ['G2',0, -5],
#                 ['G3',0, -8],
#                 ['B1',1*np.sqrt(3), 1],
#                 ['B2',2.5*np.sqrt(3),2.5],
#                 ['B3',4*np.sqrt(3), 4]]

#4 parties
mean_red = [-1.5, 1.5]
cov_red = [[1, 0], [0, 1]]  # diagonal covariance
pos_red = np.random.multivariate_normal(mean_red, cov_red, 4000)
df_red = pd.DataFrame.from_records(pos_red, columns = ['x','y'])
df_red['colour'] = 'red'
voter_groups.append(df_red)

mean_green = [-1.5, -1.5]
cov_green = [[1, 0], [0, 1]]  # diagonal covariance
pos_green = np.random.multivariate_normal(mean_green, cov_green, 1500)
df_green = pd.DataFrame.from_records(pos_green, columns = ['x','y'])
df_green['colour'] = 'green'
voter_groups.append(df_green)

mean_blue = [1.5, 1.5]
cov_blue = [[1, 0], [0, 1]]  # diagonal covariance
pos_blue = np.random.multivariate_normal(mean_blue, cov_blue, 2000)
df_blue = pd.DataFrame.from_records(pos_blue, columns = ['x','y'])
df_blue['colour'] = 'blue'
voter_groups.append(df_blue)

mean_yellow = [1.5, -1.5]
cov_yellow = [[1, 0], [0, 1]]  # diagonal covariance
pos_yellow = np.random.multivariate_normal(mean_yellow, cov_yellow, 2500)
df_yellow = pd.DataFrame.from_records(pos_yellow, columns = ['x','y'])
df_yellow['colour'] = 'yellow'
voter_groups.append(df_yellow)

candidates = [
                ['A',0,0],
                ['Z1',0,0.5],
                ['Z2',0,1.5],
                ['Z3',0,2.5],
                ['R1',-0.5, 0.5],
                ['R2',-1.5, 1.5],
                ['R3',-2.5, 2.5],
                ['G1',-0.5, -0.5],
                ['G2',-1.5, -1.5],
                ['G3',-2.5, -2.5],
                ['B1',0.5, 0.5],
                ['B2',1.5, 1.5],
                ['B3',2.5, 2.5],
                ['Y1',0.5, -0.5],
                ['Y2',1.5, -1.5],
                ['Y3',2.5, -2.5]
                                        ]

df_can = pd.DataFrame.from_records(candidates, columns = ['Name','x','y'] )

fig = plt.figure(figsize=(20,10))
fig.suptitle('Political Simulation')

#image
try:
    ax1 = fig.add_subplot(1, 2, 1)
    img=mpimg.imread('Political Compass.jpg')
    ax1.imshow(img)
    ax1.axis('off')
except:
    print('image missing')
    
# Scatter plot
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(df_red['x'],df_red['y'],".",label = 'Red', color='r') 
ax2.plot(df_green['x'],df_green['y'],".",label = 'Green', color='g') 
ax2.plot(df_blue['x'],df_blue['y'],".",label = 'Blue', color='b')
ax2.plot(df_yellow['x'],df_yellow['y'],".",label = 'Yellow', color='y')

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
    voter_groups.append(df_center) 
    
df_voters = pd.concat(voter_groups,ignore_index=True)

#Number of voters
V = df_voters.shape[0]

#Make 3d plot of df_voters
fig2 = plt.figure(figsize=(20,10))
fig2.suptitle('Voter Density')

#histogram
axa = fig2.add_subplot(121, projection='3d')
hist, xedges, yedges = np.histogram2d(df_voters['x'], df_voters['y'], bins=40, range=[[-10, 10], [-10, 10]])
X, Y = np.meshgrid(xedges[:-1] + 0.125, yedges[:-1] + 0.125, indexing="ij")
xpos = X.ravel()
ypos = Y.ravel()
zpos = 0

# Construct arrays with the dimensions for the bars.
dx = dy = 0.25 * np.ones_like(zpos)
dz = hist.ravel()

axa.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
axa.set_xlabel('Economic')
axa.set_ylabel('Government')
axa.set_zlabel('Voter Count')
axa.view_init(35, 240)

#surface
axb = fig2.add_subplot(122, projection='3d')
surf = axb.plot_surface(X, Y, hist, cmap="gist_rainbow", linewidth=0, antialiased=False)
surf.set_edgecolors(surf.to_rgba(surf._A))
#surf.set_facecolors("white")
#cset = axb.contour(X, Y, hist, zdir='z', offset=-100, cmap=cm.coolwarm)
#cset = axb.contour(X, Y, hist, zdir='x', offset=-40, cmap=cm.coolwarm)
#cset = axb.contour(X, Y, hist, zdir='y', offset=40, cmap=cm.coolwarm)
axb.set_xlabel('Economic')
axb.set_ylabel('Government')
axb.set_zlabel('Voter Count')
axb.view_init(35, 240)
fig2.colorbar(ax = axb, mappable = surf, shrink=0.5, aspect=5)
fig2.savefig("3D_Population", dpi=300)

#Get distances then scores                
distance = pd.DataFrame()
S = pd.DataFrame()
        
for c in candidates: 
    distance[c[0]] = df_voters[['x', 'y']].sub(np.array([c[1], c[2]])).pow(2).sum(1).pow(0.5)    
    S[c[0]] = round(np.clip(K - 2.0*distance[c[0]], 0.0, K))

#rowwise max set to 5
if maximize:
    columns = distance.idxmin('columns')
    for index in S.index:
        S.loc[index,columns[index]] = 5

#define seleciton algorithm
def get_winners(S_in,Selection = 'Utilitarian',Reweight = 'Unitary', KP_Transform = False):
    
    if KP_Transform:
        S_wrk = pd.DataFrame([], columns = S_in.columns )
        for i in [0,1,2,3,4]:
            S_wrk = S_wrk.append(pd.DataFrame.from_records(np.where(S_in>i,1,0), columns = S_in.columns ))
    else:    
        S_wrk = S_in.divide(K)
        
    S_orig = S_wrk.copy()
    score_remaining = np.ones(V)
#     print('Selection type = ' + Selection)
#     print('Reweight type = ' + Reweight)
# 
#     print('Hare Quota', V/W)
#     print('Total Utility Remaining ',score_remaining.sum()) 
    print('~~~~~~~~~~~~~~~~~~')
    winner_list = []
    while len(winner_list) < W:
        #round
        R = len(winner_list)

        #select winner
        #w = index with highest selection metric
        if Selection == 'Monroe':
            w = pd.DataFrame.from_records(np.sort(S_wrk.values, axis=0), columns = S_wrk.columns).tail(round(V/W)).sum().idxmax() 
        elif Selection == 'Exponential':
            total_sum =  S_orig[winner_list].sum(axis=1)
            w = S_orig.pow(W-R).div(total_sum + 1.0, axis = 0).sum().idxmax()       
        elif Selection == 'Utilitarian':          
            w = S_wrk.sum().idxmax() 
        else:
            print('error: no selection criteria')
        
        #print( sum_scores)   
                    
        # print('Winner',w)
            
        winner_list.append(w)
        
        if Reweight == 'Unitary':
            surplus_factor = max( S_wrk.sum()[w] *W/V , 1.0)
        
            #Score spent on each winner by each voter
            score_spent = S_wrk[w]/ surplus_factor
            # print('Score Spent ',score_spent.sum()) 
            
            #Total score left to be spent by each voter
            score_remaining = np.clip(score_remaining-score_spent,0.0,1.0) 
            # print('Score Remaining ',score_remaining.sum())
            # print('------------------')
            
            #Update Ballots
            #set scores to zero for winner so they don't win again
            #S_wrk[w]=0
            #Take score off of ballot (ie reweight)
        
            for c in S_wrk.columns:
                S_wrk[c] = pd.DataFrame([S_wrk[c], score_remaining]).min()
        elif Reweight == 'Divisor':        
            total_sum =  S_orig[winner_list].sum(axis=1)
            S_wrk = S_orig.div(total_sum + 1.0, axis = 0)     
        else:
            print('no reweight')                       


    if KP_Transform:    
        print(Selection + ' ' + Reweight + ' with KP transform Winner set is:')
    else:
        print(Selection + ' ' + Reweight + ' Winner set is:')
    print(winner_list)
    S_metrics = S_in.divide(K)
    print('Total Utility = ',S_metrics[winner_list].sum(axis=1).sum())
    print('Total ln(Utility) = ',np.log1p(S_metrics[winner_list].sum(axis=1)).sum())
    print('Total UnSatisfied Utility = ', sum([1-i for i in S_metrics[winner_list].sum(axis=1) if i < 1]))
    print('Fully Satisfied voters = ', sum([(i>=1) for i in S_metrics[winner_list].sum(axis=1)]))
    print('Wasted voters = ', sum([(i==0) for i in S_metrics[winner_list].sum(axis=1)]))
    print(' ') 
       
    return winner_list

utilitarian_unitary_winners = get_winners(S_in=S.copy(),Selection = 'Utilitarian',Reweight = 'Unitary')
monroe_unitary_winners = get_winners(S_in=S.copy(),Selection = 'Monroe',Reweight = 'Unitary')
utilitarian_harmonic_winners = get_winners(S_in=S.copy(),Selection = 'Utilitarian',Reweight = 'Divisor')
utilitarian_harmonic_winners_KP = get_winners(S_in=S.copy(),Selection = 'Utilitarian',Reweight = 'Divisor', KP_Transform = True)
monroe_harmonic_winners = get_winners(S_in=S.copy(),Selection = 'Monroe',Reweight = 'Divisor') 
exponential_winners = get_winners(S_in=S.copy(),Selection = 'Exponential' ,Reweight = 'none')  

# print('SUMMARY')  
# print('Utilitarian Unitary Winner set is:')
# print(utilitarian_unitary_winners)
# print('Monroe Unitary Winner set is:')
# print(monroe_unitary_winners)
# print('Utilitarian Harmonic Winner set is:')       
# print(utilitarian_harmonic_winners)
# print('Monroe Harmonic Winner set is:')
# print(monroe_harmonic_winners)

plt.show()


