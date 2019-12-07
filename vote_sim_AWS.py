import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
from time import time
import datetime
import sys
from scipy.spatial.distance import cdist
import matplotlib.image as mpimg
import utils
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

#Define Data frames to store the results of each iteration
Methods = {}

Methods['utilitarian_unitary'] = {'Selection' : 'Utilitarian', 'Reweight' : 'Unitary', 'KP_Transform' : False} #Sequentially Spent Score
Methods['hare_voters_unitary'] = {'Selection' : 'Hare_Voters', 'Reweight' : 'Unitary', 'KP_Transform' : False}
Methods['hare_ballots_unitary'] = {'Selection' : 'Hare_Ballots', 'Reweight' : 'Unitary', 'KP_Transform' : False}

Methods['utilitarian_jefferson'] = {'Selection' : 'Utilitarian', 'Reweight' : 'Jefferson', 'KP_Transform' : False} #Reweighted Range Voting
Methods['hare_voters_jefferson'] = {'Selection' : 'Hare_Voters', 'Reweight' : 'Jefferson', 'KP_Transform' : False}
Methods['hare_ballots_jefferson'] = {'Selection' : 'Hare_Ballots', 'Reweight' : 'Jefferson', 'KP_Transform' : False}

Methods['utilitarian_allocate'] = {'Selection' : 'Utilitarian', 'Reweight' : 'Allocate', 'KP_Transform' : False} #Allocated Score
Methods['hare_voters_allocate'] = {'Selection' : 'Hare_Voters', 'Reweight' : 'Allocate', 'KP_Transform' : False}
Methods['hare_ballots_allocate'] = {'Selection' : 'Hare_Ballots', 'Reweight' : 'Allocate', 'KP_Transform' : False} #Sequential Monroe
#
# Methods['utilitarian_unitary_kp'] = {'Selection' : 'Utilitarian', 'Reweight' : 'Unitary', 'KP_Transform' : True}
# Methods['hare_voters_unitary_kp'] = {'Selection' : 'Hare_Voters', 'Reweight' : 'Unitary', 'KP_Transform' : True}
# Methods['hare_ballots_unitary_kp'] = {'Selection' : 'Hare_Ballots', 'Reweight' : 'Unitary', 'KP_Transform' : True}
#
# Methods['utilitarian_jefferson_kp'] = {'Selection' : 'Utilitarian', 'Reweight' : 'Jefferson', 'KP_Transform' : True}
# Methods['hare_voters_jefferson_kp'] = {'Selection' : 'Hare_Voters', 'Reweight' : 'Jefferson', 'KP_Transform' : True}
# Methods['hare_ballots_jefferson_kp'] = {'Selection' : 'Hare_Ballots', 'Reweight' : 'Jefferson', 'KP_Transform' : True}
#
# Methods['utilitarian_allocate_kp'] = {'Selection' : 'Utilitarian', 'Reweight' : 'Allocate', 'KP_Transform' : True}
# Methods['hare_voters_allocate_kp'] = {'Selection' : 'Hare_Voters', 'Reweight' : 'Allocate', 'KP_Transform' : True}
# Methods['hare_ballots_allocate_kp'] = {'Selection' : 'Hare_Ballots', 'Reweight' : 'Allocate', 'KP_Transform' : True}


method_list = sorted(list(Methods.keys()))

df_total_utility = pd.DataFrame(columns=method_list)
df_total_ln_utility = pd.DataFrame(columns=method_list)
df_total_favored_winner_utility = pd.DataFrame(columns=method_list)
df_total_unsatisfied_utility = pd.DataFrame(columns=method_list)
df_fully_satisfied_voters = pd.DataFrame(columns=method_list)
df_totally_unsatisfied_voters = pd.DataFrame(columns=method_list)
df_utility_deviation = pd.DataFrame(columns=method_list)
df_score_deviation = pd.DataFrame(columns=method_list)
df_favored_winner_deviation = pd.DataFrame(columns=method_list)
df_number_of_duplicates = pd.DataFrame(columns=method_list)
df_average_winner_polarization = pd.DataFrame(columns=method_list)
df_most_polarized_winner = pd.DataFrame(columns=method_list)
df_least_polarized_winner = pd.DataFrame(columns=method_list)

df_parties = pd.DataFrame(columns = ['x', 'y', 'size', 'party_name'])

#Iterate over each simulation
start_time = time()
for iteration in range(25000):
    if (iteration % 10 == 0) or iteration < 10:
        print('iteration: ' + str(iteration))
        print(datetime.datetime.now())
        print('processing_minutes = ' + str((time() - start_time)/60), flush=True)

    np.random.seed(iteration )

    # Number of party distributions
    num_parties = np.random.randint(2, 7)

    # Spread of each distribution
    party_std = np.random.uniform(low=0.5, high=2.0, size=num_parties)

    # Number of voters in each party
    party_voters = [int(i) for i in np.random.dirichlet(np.ones(num_parties))*V]

    # Make some fake vote data
    # Each voter is simulated as a point in a 2D space
    location, party_ID = make_blobs(n_samples=party_voters,  # voters
                    n_features=2,  # dimensions
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

    metrics = {}

    #Run methods and get metrics
    for method,value in Methods.items():
        winner_list = utils.get_winners(S_in=S.copy(),Selection = value['Selection'],Reweight = value['Reweight'], KP_Transform = value['KP_Transform'], K=K, W=W)
        metrics = utils.get_metrics(S_in=S.copy(), metrics =metrics, winner_list = winner_list, method = method, K=K)

    #Add metrics to dataframes
    df_total_utility = df_total_utility.append(metrics['total_utility'], ignore_index=True)
    df_total_ln_utility = df_total_ln_utility.append(metrics['total_ln_utility'], ignore_index=True)
    df_total_favored_winner_utility = df_total_favored_winner_utility.append(metrics['total_favored_winner_utility'], ignore_index=True)
    df_total_unsatisfied_utility = df_total_unsatisfied_utility.append(metrics['total_unsatisfied_utility'], ignore_index=True)
    df_fully_satisfied_voters = df_fully_satisfied_voters.append(metrics['fully_satisfied_voters'], ignore_index=True)
    df_totally_unsatisfied_voters = df_totally_unsatisfied_voters.append(metrics['totally_unsatisfied_voters'], ignore_index=True)
    df_utility_deviation = df_utility_deviation.append(metrics['utility_deviation'], ignore_index=True)
    df_score_deviation = df_score_deviation.append(metrics['score_deviation'], ignore_index=True)
    df_favored_winner_deviation = df_favored_winner_deviation.append(metrics['favored_winner_deviation'], ignore_index=True)
    df_number_of_duplicates = df_number_of_duplicates.append(metrics['number_of_duplicates'], ignore_index=True)
    df_average_winner_polarization = df_average_winner_polarization.append(metrics['average_winner_polarization'], ignore_index=True)
    df_most_polarized_winner = df_most_polarized_winner.append(metrics['most_polarized_winner'], ignore_index=True)
    df_least_polarized_winner = df_least_polarized_winner.append(metrics['least_polarized_winner'], ignore_index=True)

    #Keep track of simulation points
    df_temp = df_voters.groupby('party_name')['x','y'].mean()
    df_temp['size'] = df_voters.groupby('party_name')['party_ID'].count()
    df_parties = df_parties.append(df_temp.reset_index(), ignore_index=True,sort=False)

#Write dataframes of results
df_total_utility.to_csv('total_utility.csv',index=False)
df_total_ln_utility.to_csv('total_ln_utility.csv',index=False)
df_total_favored_winner_utility.to_csv('total_favored_winner_utility.csv',index=False)
df_total_unsatisfied_utility.to_csv('total_unsatisfied_utility.csv',index=False)
df_fully_satisfied_voters.to_csv('fully_satisfied_voters.csv',index=False)
df_totally_unsatisfied_voters.to_csv('totally_unsatisfied_voters.csv',index=False)
df_utility_deviation.to_csv('utility_deviation.csv',index=False)
df_score_deviation.to_csv('score_deviation.csv',index=False)
df_favored_winner_deviation.to_csv('favored_winner_deviation.csv',index=False)
df_number_of_duplicates.to_csv('number_of_duplicates.csv',index=False)
df_average_winner_polarization.to_csv('average_winner_polarization.csv',index=False)
df_most_polarized_winner.to_csv('most_polarized_winner.csv',index=False)
df_least_polarized_winner.to_csv('least_polarized_winner.csv',index=False)
df_parties.to_csv('parties.csv',index=False)


#make plots

fig = plt.figure(figsize=(15,20))
fig.suptitle('Utility Metrics')

ax1 = fig.add_subplot(3, 2, 1)
ax1 = utils.plot_metric(df = df_total_utility, Methods = Methods,axis=ax1,is_int = True)
lgd1 = ax1.legend(loc=2)
ax1.set_xlabel('Total Utility')
ax1.set_ylabel('Records in bin')

ax2 = fig.add_subplot(3, 2, 2)
ax2 = utils.plot_metric(df = df_total_ln_utility, Methods = Methods,axis=ax2,is_int = True)
lgd2 = ax2.legend(loc=2)
ax2.set_xlabel('Total Log Utility')
ax2.set_ylabel('Records in bin')

ax3 = fig.add_subplot(3, 2, 3)
ax3 = utils.plot_metric(df = df_total_favored_winner_utility, Methods = Methods,axis=ax3,is_int = True)
lgd3 = ax3.legend(loc=2)
ax3.set_xlabel('Total Favored Winner Utility')
ax3.set_ylabel('Records in bin')

ax4 = fig.add_subplot(3, 2, 4)
ax4 = utils.plot_metric(df = df_total_unsatisfied_utility, Methods = Methods,axis=ax4,is_int = True)
lgd4 = ax4.legend(loc=2)
ax4.set_xlabel('Total Unsatisfied Utility')
ax4.set_ylabel('Records in bin')

ax5 = fig.add_subplot(3, 2, 5)
ax5 = utils.plot_metric(df = df_fully_satisfied_voters, Methods = Methods,axis=ax5,is_int = True)
lgd5 = ax5.legend(loc=2)
ax5.set_xlabel('Fully Satisfied Voters')
ax5.set_ylabel('Records in bin')

ax6 = fig.add_subplot(3, 2, 6)
ax6 = utils.plot_metric(df = df_totally_unsatisfied_voters, Methods = Methods,axis=ax6,is_int = True)
lgd6 = ax6.legend(loc=2)
ax6.set_xlabel('Totally Unsatisfied Voters')
ax6.set_ylabel('Records in bin')

fig.savefig("Utility_Results.png",dpi = 300)

figB = plt.figure(figsize=(15,20))
figB.suptitle('Equity Metrics')

axB1 = figB.add_subplot(3, 2, 1)
axB1 = utils.plot_metric(df = df_utility_deviation, Methods = Methods,axis=axB1,is_int = False)
lgd2 = axB1.legend(loc=2)
axB1.set_xlabel('Utility Deviation')
axB1.set_ylabel('Records in bin')

axB2 = figB.add_subplot(3, 2, 2)
axB2 = utils.plot_metric(df = df_score_deviation, Methods = Methods,axis=axB2,is_int = False)
lgd2 = axB2.legend(loc=2)
axB2.set_xlabel('Score Deviation')
axB2.set_ylabel('Records in bin')

axB3 = figB.add_subplot(3, 2, 3)
axB3 = utils.plot_metric(df = df_favored_winner_deviation, Methods = Methods,axis=axB3,is_int = False)
lgd3 = axB3.legend(loc=2)
axB3.set_xlabel('Favored Winner Deviation')
axB3.set_ylabel('Records in bin')

axB4 = figB.add_subplot(3, 2, 4)
axB4 = utils.plot_metric(df = df_number_of_duplicates, Methods = Methods,axis=axB4,is_int = False)
lgd4 = axB4.legend(loc=2)
axB4.set_xlabel('Number of Duplicate Winners')
axB4.set_ylabel('Records in bin')

axB5 = figB.add_subplot(3, 2, 5)
axB5 = utils.plot_metric(df = df_average_winner_polarization, Methods = Methods,axis=axB5,is_int = False)
lgd5 = axB5.legend(loc=2)
axB5.set_xlabel('Average Winner Polarization')
axB5.set_ylabel('Records in bin')

# This metric is not very useful since most systems have the same most polarized winner
# axB5 = figB.add_subplot(3, 2, 5)
# axB5 = utils.plot_metric(df = df_most_polarized_winner, Methods = Methods,axis=axB5,is_int = False)
# lgd5 = axB5.legend(loc=2)
# axB5.set_xlabel('Most Polarized Winner')
# axB5.set_ylabel('Records in bin')

axB6 = figB.add_subplot(3, 2, 6)
axB6 = utils.plot_metric(df = df_least_polarized_winner, Methods = Methods,axis=axB6,is_int = False)
lgd6 = axB6.legend(loc=2)
axB6.set_xlabel('Least Polarized Winner')
axB6.set_ylabel('Records in bin')

figB.savefig("Equity_Results.png",dpi = 300)

figC = plt.figure(figsize=(20,10))
figC.suptitle('Ideological Space')

#image
try:
    axC1 = figC.add_subplot(1, 2, 1)
    img=mpimg.imread('Political Compass.jpg')
    axC1.imshow(img)
    axC1.axis('off')
except:
    print('image missing')

axC2 = figC.add_subplot(1, 2, 2)
axC2.scatter(df_parties['x'],df_parties['y'], marker='.', s=(df_parties['size']).astype('int'), c=df_parties['party_name'])
axC2.set_xlim(-10, 10)
axC2.set_ylim(-10, 10)
axC2.set_xticks(range(-10, 11,2))
axC2.set_yticks(range(-10, 11,2))
axC2.set_title('Party Position')
axC2.set_xlabel('Planned Economy  <--  Economics  -->  Free Market')
axC2.set_ylabel('Liberal  <-- Government  --> Authoritarian')

figC.savefig("Simulated_Spectrum.png",dpi = 300)

print('done')

sys.stdout = term
f.close()