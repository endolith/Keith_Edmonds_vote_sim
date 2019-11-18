import pandas as pd
import numpy as np

# function to turn scores into a winner set for various systems
def get_winners(S_in,Selection = 'Utilitarian',Reweight = 'Unitary', KP_Transform = False, W=5.0, K=5):

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
        #R = len(winner_list)

        #select winner
        #w = index with highest selection metric
        if Selection == 'Monroe':
            w = pd.DataFrame(np.sort(S_wrk.values, axis=0), columns=S_wrk.columns).tail(round(V/W)).sum().idxmax()
        elif Selection == 'Utilitarian':
            w = S_wrk.sum().idxmax()

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
            mins = np.minimum(S_wrk.values,
                              score_remaining.values[:, np.newaxis])
            S_wrk = pd.DataFrame(mins, columns = S_wrk.columns)
        elif Reweight == 'Divisor':
            total_sum =  S_orig[winner_list].sum(axis=1)
            S_wrk = S_orig.div(total_sum + 1.0, axis = 0)

    return winner_list

#Method to get all output quality metrics for a winner set

def get_metrics(S_in,metrics,winner_list,method,K=5):
    S_metrics = S_in.divide(K)
    metrics['total_utility'][method] = S_metrics[winner_list].sum(axis=1).sum()
    metrics['total_ln_utility'][method] = np.log1p(S_metrics[winner_list].sum(axis=1)).sum()
    metrics['total_favored_winner_utility'][method] = S_metrics[winner_list].max(axis=1).sum()
    metrics['total_unsatisfied_utility'][method] = sum([1-i for i in S_metrics[winner_list].sum(axis=1) if i < 1])
    metrics['fully_satisfied_voters'][method] = sum([(i>=1) for i in S_metrics[winner_list].sum(axis=1)])
    metrics['wasted_voters'][method] = sum([(i==0) for i in S_metrics[winner_list].sum(axis=1)])
    metrics['utility_deviation'][method] = S_metrics[winner_list].sum(axis=1).std()
    metrics['score_deviation'][method] = S_metrics[winner_list].values.flatten().std()
    metrics['favored_winner_deviation'][method] = S_metrics[winner_list].max(axis=1).std()
    metrics['average_winner_polarization'][method] = S_metrics[winner_list].std(axis=0).mean()
    metrics['most_polarized_winner'][method] = S_metrics[winner_list].std(axis=0).max()
    metrics['least_polarized_winner'][method] = S_metrics[winner_list].std(axis=0).min()
    return   metrics