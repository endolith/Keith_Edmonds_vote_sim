import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
from time import time
import datetime
import sys
from scipy.spatial.distance import cdist

#
# Types of problems to handle
# https://www.rangevoting.org/AssetBC.html
# https://groups.google.com/forum/#!topic/electionscience/Rk4ZGf-s-s8

#Log the output
start_time_hr = datetime.datetime.now()
log_file = 'votelog_' + str(start_time_hr) + '.txt'
log_file = log_file.replace(':','-')
term = sys.stdout
sys.stdout = f = open(log_file,'w')

# Set the basic parameters
W = 5.0   # Number of winners
K = 5     # The maximum possible score is 5
V = 10000 # Number of voters

# function to turn scores into a winner set for various systems
def get_winners(S_in,Selection = 'Utilitarian',Reweight = 'Unitary', KP_Transform = False, W=W, K=K):

    V = S_in.shape[0]

    #create the working set of scores
    if KP_Transform:
        # The KP transform changes each voter into a set of K approval voters
        groups = []
        for threshold in range(K):
            groups.append(np.where(S_in.values > threshold, 1, 0))
        S_wrk = pd.DataFrame(np.concatenate(groups), columns=S_in.columns)
    else:
        #Normalise so scores are in [0,1]
        S_wrk = pd.DataFrame(S_in.values/K, columns=S_in.columns)

    #make copy of working scores
    S_orig = S_wrk.copy()
    score_remaining = np.ones(V)

    #Populate winners in a loop
    winner_list = []
    while len(winner_list) < W:
        #round
        R = len(winner_list)

        #select winner
        #w = index with highest selection metric
        if Selection == 'Monroe':
            w = pd.DataFrame.from_records(np.sort(S_wrk.values, axis=0), columns = S_wrk.columns).tail(round(V/W)).sum().idxmax()
        elif Selection == 'Utilitarian':
            w = S_wrk.sum().idxmax()
        elif Selection == 'Exponential':
            total_sum =  S_orig[winner_list].sum(axis=1)
            w = S_orig.pow(W-R).div(total_sum + 1.0, axis = 0).sum().idxmax()
        winner_list.append(w)

        #Reweight the working scores
        if Reweight == 'Unitary':
            surplus_factor = max( S_wrk.sum()[w] *W/V , 1.0)

            #Score spent on each winner by each voter
            score_spent = S_wrk[w]/ surplus_factor
            # print('Score Spent ',score_spent.sum())

            #Total score left to be spent by each voter
            score_remaining = np.clip(score_remaining-score_spent,0.0,1.0)

            #Update Ballots
            #set scores to zero for winner so they don't win again
            #in most simulations this would not be needed since you would want to check for for
            #S_wrk[w]=0
            #Take score off of ballot (ie reweight)

            for c in S_wrk.columns:
                S_wrk[c] = pd.DataFrame([S_wrk[c], score_remaining]).min()
        elif Reweight == 'Divisor':
            total_sum =  S_orig[winner_list].sum(axis=1)
            S_wrk = S_orig.div(total_sum + 1.0, axis = 0)

    return winner_list

#Method to get all output quality metrics for a winner set
def get_metrics(S_in,winner_list,K=5):
    S_metrics = S_in.divide(K)
    total_utility = S_metrics[winner_list].sum(axis=1).sum()
    total_ln_utility = np.log1p(S_metrics[winner_list].sum(axis=1)).sum()
    total_unsatisfied_utility = sum([1-i for i in S_metrics[winner_list].sum(axis=1) if i < 1])
    fully_satisfied_voters = sum([(i>=1) for i in S_metrics[winner_list].sum(axis=1)])
    wasted_voters = sum([(i==0) for i in S_metrics[winner_list].sum(axis=1)])

    return   total_utility, total_ln_utility, total_unsatisfied_utility, fully_satisfied_voters, wasted_voters

#Define Data frames to store the results of each iteration
methods = ['utilitarian_unitary','monroe_unitary','utilitarian_divisor','utilitarian_divisor_kp','monroe_divisor','exponential']

df_total_utility = pd.DataFrame(columns=methods)
df_total_ln_utility = pd.DataFrame(columns=methods)
df_total_unsatisfied_utility = pd.DataFrame(columns=methods)
df_fully_satisfied_voters = pd.DataFrame(columns=methods)
df_wasted_voters = pd.DataFrame(columns=methods)
df_parties = pd.DataFrame(columns = ['x', 'y', 'size', 'party_name'])

#Iterate over each simulation
start_time = time()
for iteration in range(1000):
    if (iteration % 10 == 0) or iteration < 10:
        print('iteration: ' + str(iteration))
        print(datetime.datetime.now())
        print('processing_minutes = ' + str((time() - start_time)/60), flush=True)

    num_parties = np.random.randint(2, 7)
    party_std = np.random.uniform(low=0.5, high=2.0, size=num_parties)
    party_voters = [int(i) for i in np.random.dirichlet(np.ones(num_parties))*V]

    #Make some fake vote data
    #each voter is simulated as a point in a 2D space
    location, party_ID = make_blobs(n_samples=party_voters, #voters
                    n_features=2, #dimensions
                    cluster_std=party_std,
                    center_box=(-8.0, 8.0),
                    shuffle=True,
                    random_state=iteration)  # For reproducibility

    party_list = ['red','green','blue','black','yellow','cyan']

    df_voters = pd.DataFrame(location,columns=['x','y'])
    df_voters['party_ID'] = party_ID
    df_voters['party_name'] = [party_list[i] for i in party_ID]

    # Put Candidates on each grid point in a 2D space
    # This is the best way to do it since candidates in the real world
    # will adjust to the chosen system and we only care about the optimal case
    xx, yy = np.meshgrid(range(-9, 10), range(-9, 10))
    df_can = pd.DataFrame({'x':xx.ravel(),'y':yy.ravel()})
    df_can['Name'] = '(' + df_can['x'].astype(str) +','+ df_can['y'].astype(str) + ')'

    # Calculate Euclidean between all voters and all candidates
    # and then convert to scores
    dists = cdist(df_voters[['x', 'y']], df_can[['x', 'y']])
    distance = pd.DataFrame(dists, columns=df_can.values[:, 2])
    scores = np.around(np.clip(K - 2.0*dists, 0.0, K))

    # Row-wise, set max to 5
    scores[np.arange(len(scores)), np.argmin(dists, 1)] = 5
    S = pd.DataFrame(scores, columns=df_can.values[:, 2])

    #store metrics for each method
    total_utility = {}
    total_ln_utility = {}
    total_unsatisfied_utility = {}
    fully_satisfied_voters = {}
    wasted_voters = {}

    #Run methods and get metrics
    winner_list = get_winners(S_in=S.copy(),Selection = 'Utilitarian',Reweight = 'Unitary')
    total_utility['utilitarian_unitary'], total_ln_utility['utilitarian_unitary'], total_unsatisfied_utility['utilitarian_unitary'], fully_satisfied_voters['utilitarian_unitary'], wasted_voters['utilitarian_unitary'] = get_metrics(S_in=S.copy(),winner_list = winner_list,K=5)

    winner_list = get_winners(S_in=S.copy(),Selection = 'Monroe',Reweight = 'Unitary')
    total_utility['monroe_unitary'], total_ln_utility['monroe_unitary'], total_unsatisfied_utility['monroe_unitary'], fully_satisfied_voters['monroe_unitary'], wasted_voters['monroe_unitary'] = get_metrics(S_in=S.copy(),winner_list = winner_list,K=5)

    winner_list = get_winners(S_in=S.copy(),Selection = 'Utilitarian',Reweight = 'Divisor')
    total_utility['utilitarian_divisor'], total_ln_utility['utilitarian_divisor'], total_unsatisfied_utility['utilitarian_divisor'], fully_satisfied_voters['utilitarian_divisor'], wasted_voters['utilitarian_divisor'] = get_metrics(S_in=S.copy(),winner_list = winner_list,K=5)

    winner_list = get_winners(S_in=S.copy(),Selection = 'Utilitarian',Reweight = 'Divisor', KP_Transform = True)
    total_utility['utilitarian_divisor_kp'], total_ln_utility['utilitarian_divisor_kp'], total_unsatisfied_utility['utilitarian_divisor_kp'], fully_satisfied_voters['utilitarian_divisor_kp'], wasted_voters['utilitarian_divisor_kp'] = get_metrics(S_in=S.copy(),winner_list = winner_list,K=5)

    winner_list = get_winners(S_in=S.copy(),Selection = 'Monroe',Reweight = 'Divisor')
    total_utility['monroe_divisor'], total_ln_utility['monroe_divisor'], total_unsatisfied_utility['monroe_divisor'], fully_satisfied_voters['monroe_divisor'], wasted_voters['monroe_divisor'] = get_metrics(S_in=S.copy(),winner_list = winner_list,K=5)

    winner_list = get_winners(S_in=S.copy(),Selection = 'Exponential' ,Reweight = 'none')
    total_utility['exponential'], total_ln_utility['exponential'], total_unsatisfied_utility['exponential'], fully_satisfied_voters['exponential'], wasted_voters['exponential'] = get_metrics(S_in=S.copy(),winner_list = winner_list,K=5)

    #Add metrics to dataframes
    df_total_utility = df_total_utility.append(total_utility, ignore_index=True)
    df_total_ln_utility = df_total_ln_utility.append(total_ln_utility, ignore_index=True)
    df_total_unsatisfied_utility = df_total_unsatisfied_utility.append(total_unsatisfied_utility, ignore_index=True)
    df_fully_satisfied_voters = df_fully_satisfied_voters.append(fully_satisfied_voters, ignore_index=True)
    df_wasted_voters = df_wasted_voters.append(wasted_voters, ignore_index=True)

    #Keep track of simulation points
    df_temp = df_voters.groupby('party_name')['x','y'].mean()
    df_temp['size'] = df_voters.groupby('party_name')['party_ID'].count()
    df_parties = df_parties.append(df_temp.reset_index(), ignore_index=True,sort=False)

#plots metrics
colors = ['b','r','k','#FFFF00','g','#808080','#56B4E9','#FF7F00']
fig = plt.figure(figsize=(15,20))
fig.suptitle('Quality of Election Method')

ax1 = fig.add_subplot(3, 2, 1)
for i, col in enumerate(df_total_utility.columns):
    count, bins, ignored = ax1.hist(list(df_total_utility[col]), 30, color=colors[i] ,histtype = 'step', label=col)
    textstr = '$\mu=%.0f$\n$\sigma=%.0f$'%(df_total_utility[col].mean(), df_total_utility[col].std())
    props = dict(boxstyle='round', facecolor='white', ec=colors[i], alpha=0.8)
    ax1.text(0.98, 0.95-0.1*i, textstr, transform=ax1.transAxes, bbox=props, verticalalignment='top',horizontalalignment='right')
lgd1 = ax1.legend(loc=2)
ax1.set_xlabel('Total Utility')
ax1.set_ylabel('Records in bin')

ax2 = fig.add_subplot(3, 2, 2)
for i, col in enumerate(df_total_ln_utility.columns):
    count, bins, ignored = ax2.hist(list(df_total_ln_utility[col]), 30, color=colors[i] ,histtype = 'step', label=col)
    textstr = '$\mu=%.0f$\n$\sigma=%.0f$'%(df_total_ln_utility[col].mean(), df_total_ln_utility[col].std())
    props = dict(boxstyle='round', facecolor='white', ec=colors[i], alpha=0.8)
    ax2.text(0.98, 0.95-0.1*i, textstr, transform=ax2.transAxes, bbox=props, verticalalignment='top',horizontalalignment='right')
lgd2 = ax2.legend(loc=2)
ax2.set_xlabel('Total Log Utility')
ax2.set_ylabel('Records in bin')

ax3 = fig.add_subplot(3, 2, 3)
for i, col in enumerate(df_total_unsatisfied_utility.columns):
    count, bins, ignored = ax3.hist(list(df_total_unsatisfied_utility[col]), 30, color=colors[i] ,histtype = 'step', label=col)
    textstr = '$\mu=%.0f$\n$\sigma=%.0f$'%(df_total_unsatisfied_utility[col].mean(), df_total_unsatisfied_utility[col].std())
    props = dict(boxstyle='round', facecolor='white', ec=colors[i], alpha=0.8)
    ax3.text(0.98, 0.95-0.1*i, textstr, transform=ax3.transAxes, bbox=props, verticalalignment='top',horizontalalignment='right')
lgd3 = ax3.legend(loc=2)
ax3.set_xlabel('Total Unsatisfied Utility')
ax3.set_ylabel('Records in bin')

ax4 = fig.add_subplot(3, 2, 4)
for i, col in enumerate(df_fully_satisfied_voters.columns):
    count, bins, ignored = ax4.hist(list(df_fully_satisfied_voters[col]), 30, color=colors[i] ,histtype = 'step', label=col)
    textstr = '$\mu=%.0f$\n$\sigma=%.0f$'%(df_fully_satisfied_voters[col].mean(), df_fully_satisfied_voters[col].std())
    props = dict(boxstyle='round', facecolor='white', ec=colors[i], alpha=0.8)
    ax4.text(0.98, 0.95-0.1*i, textstr, transform=ax4.transAxes, bbox=props, verticalalignment='top',horizontalalignment='right')
lgd4 = ax4.legend(loc=2)
ax4.set_xlabel('Fully Satisfied Voters')
ax4.set_ylabel('Records in bin')

ax5 = fig.add_subplot(3, 2, 5)
for i, col in enumerate(df_wasted_voters.columns):
    count, bins, ignored = ax5.hist(list(df_wasted_voters[col]), 30, color=colors[i] ,histtype = 'step', label=col)
    textstr = '$\mu=%.0f$\n$\sigma=%.0f$'%(df_wasted_voters[col].mean(), df_wasted_voters[col].std())
    props = dict(boxstyle='round', facecolor='white', ec=colors[i], alpha=0.8)
    ax5.text(0.98, 0.95-0.1*i, textstr, transform=ax5.transAxes, bbox=props, verticalalignment='top',horizontalalignment='right')
lgd5 = ax5.legend(loc=2)
ax5.set_xlabel('Wasted Voters')
ax5.set_ylabel('Records in bin')

ax6 = fig.add_subplot(3, 2, 6)
ax6.scatter(df_parties['x'],df_parties['y'], marker='.', s=(df_parties['size']/10).astype('int'), c=df_parties['party_name'])
ax6.set_xlim(-10, 10)
ax6.set_ylim(-10, 10)
ax6.set_xticks(range(-10, 11,2))
ax6.set_yticks(range(-10, 11,2))
ax6.set_title('Party Position')
ax6.set_xlabel('Planned Economy  <--  Economics  -->  Free Market')
ax6.set_ylabel('Liberal  <-- Government  --> Authoritarian')

fig.savefig("Results.png",dpi = 300)

print('done')

sys.stdout = term
f.close()