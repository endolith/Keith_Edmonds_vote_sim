import pandas as pd
import numpy as np


def get_winners(S_in, Selection='Utilitarian', Reweight='Unitary', KP_Transform=False, W=5, K=5):
    """
    Turn scores into a winner set for various systems

    Parameters
    ----------
    S_in : pandas.DataFrame
        Table of scores given to each candidate by each voter
    Selection : {'Utilitarian', 'STAR', 'Hare_Ballots'}, optional
        Default is 'Utilitarian'
    Reweight : {'Unitary', 'Jefferson', 'Webster', 'Allocate'}, optional
        Default is 'Unitary'
    W : int, optional
        Maximum number of winners to return. Default is 5.
    K : int, optional
        Maximum possible score. Default is 5.
    """
    # To ensure float division in Python 2?
    W = float(W)

    # Create the working set of scores
    if KP_Transform:
        # The KP transform changes each voter into a set of K approval voters
        groups = []
        for threshold in range(K):
            groups.append(np.where(S_in.values > threshold, 1, 0))
        S_wrk = pd.DataFrame(np.concatenate(groups), columns=S_in.columns)
    else:
        # Normalise so scores are in [0, 1]
        S_wrk = pd.DataFrame(S_in.values/K, columns=S_in.columns)

    V = S_wrk.shape[0]

    # Make copy of working scores
    S_orig = S_wrk.copy()

    # These only matter for specific systems and are initialized here
    ballot_weight = pd.Series(np.ones(V))

    # Populate winners in a loop
    winner_list = []
    while len(winner_list) < W:
        #round
        #R = len(winner_list)

        #select winner
        #w = index with highest selection metric
        if Selection == 'STAR':
            #find top two
            df_tops = S_wrk[S_wrk.sum().nlargest(2, keep='all').index]
            #Run off winner
            w = df_tops.eq(df_tops.max(1), axis=0).sum().idxmax()    
        elif Selection == 'Hare_Ballots':
            # Find candidate with the highest vote sum in a hare quota of ballot weights

            # Sort each candidate by scores, from highest to lowest
            sort_idx = np.argsort(-S_orig.values, axis=0)

            # Collect ballot weights in same sorted order
            weights = ballot_weight.values[sort_idx]

            # Accumulate weights for each candidate
            sums = np.cumsum(weights, axis=0)

            # Accumulated weights under threshold
            thres = (sums < V/W)
            
            #Weight Scores
            weighted_scores = S_orig.mul(ballot_weight, axis = 0)
            
            # Sum scores for candidates under threshold
            c_score = np.sum(thres * np.take_along_axis(weighted_scores.values, sort_idx, axis=0),axis=0)
            w = S_orig.columns[np.argmax(c_score)]
        elif Selection == 'Utilitarian':
            w = S_wrk.sum().idxmax()

        winner_list.append(w)

        #Reweight the working scores
        if Reweight == 'Unitary':
            surplus_factor = max( S_wrk[w].sum() *W/V , 1.0)

            #Score spent on each winner by each voter
            score_spent = S_wrk[w]/ surplus_factor
            # print('Score Spent ',score_spent.sum())

            #Total score left to be spent by each voter
            ballot_weight = np.clip(ballot_weight-score_spent,0.0,1.0)

            #Update Ballots
            #set scores to zero for winner so they don't win again
            #in most simulations this would not be needed since you would want to check for for
            #S_wrk[w]=0
            #Take score off of ballot (ie reweight)
            mins = np.minimum(S_wrk.values,ballot_weight.values[:, np.newaxis])
            S_wrk = pd.DataFrame(mins, columns = S_wrk.columns)
            
        elif Reweight == 'Jefferson':
            total_sum =  S_orig[winner_list].sum(axis=1)
            #Ballot weight as defined by the Jefferson method
            ballot_weight = 1/(total_sum + 1)
            S_wrk = S_orig.mul(ballot_weight, axis = 0)
        
        elif Reweight == 'Webster':
            total_sum =  S_orig[winner_list].sum(axis=1)
            #Ballot weight as defined by the Webster method
            ballot_weight = 1/(2*total_sum + 1)
            S_wrk = S_orig.mul(ballot_weight, axis = 0)
            
        elif Reweight == 'Allocate':
            votes_to_allocate = round(V/W)
            cand_df = S_orig[[w]].copy()
            cand_df['ballot_weight'] = ballot_weight
            cand_df_sort = cand_df.sort_values(by=[w], ascending=False)
            
            #find the score where everybody abote is allocated
            split_point = cand_df_sort[cand_df_sort['ballot_weight'].cumsum() < V/W][w].iloc[-1]

            #if split point <0 then a full quota is not spent
            if split_point>0:
                #Amount of ballot for voters who voted on the split point
                voters_on_split = cand_df[cand_df[w] == split_point]['ballot_weight'].sum()
                
                #Amount of ballot for voters who voted more than the split point
                voters_allocated = cand_df[cand_df[w] > split_point]['ballot_weight'].sum()

                #amount to reweight the voters on the split by (ie surpluss handling)
                reweighted_value = 1 - (votes_to_allocate - voters_allocated)/voters_on_split

                #reweight voters on split
                cand_df.loc[cand_df[w] == split_point, 'ballot_weight'] = cand_df.loc[cand_df[w] == split_point, 'ballot_weight'] * reweighted_value

            #exhause ballots for those above split
            cand_df.loc[cand_df[w] >split_point, 'ballot_weight'] = 0

            #update
            ballot_weight = cand_df['ballot_weight']
            S_wrk = S_orig.mul(ballot_weight, axis = 0)

    return winner_list

#Method to get all output quality metrics for a winner set

def get_metrics(S_in,metrics,winner_list,method,K=5):

    #store metrics for each method
    if not metrics:
        average_utility = {}
        average_ln_utility = {}
        average_favored_winner_utility = {}
        average_unsatisfied_utility = {}
        fully_satisfied_voters = {}
        totally_unsatisfied_voters = {}
        harmonic_quality = {}
        unitary_quality = {}
        ebert_cost = {}
        most_blocking_loser_capture = {}
        largest_total_unsatisfied_group = {}
        average_utility_gain_from_extra_winner = {}
        utility_deviation = {}
        score_deviation = {}
        favored_winner_deviation = {}
        number_of_duplicates = {}
        average_winner_polarization = {}
        most_polarized_winner = {}
        least_polarized_winner = {}

        
        metrics = {
                    'average_utility' : average_utility,
                    'average_ln_utility' : average_ln_utility,
                    'average_favored_winner_utility' : average_favored_winner_utility,
                    'average_unsatisfied_utility' : average_unsatisfied_utility,
                    'fully_satisfied_voters' : fully_satisfied_voters,
                    'totally_unsatisfied_voters' : totally_unsatisfied_voters,
                    'harmonic_quality' : harmonic_quality,
                    'unitary_quality' : unitary_quality,
                    'ebert_cost' : ebert_cost,
                    'most_blocking_loser_capture' : most_blocking_loser_capture,
                    'largest_total_unsatisfied_group' : largest_total_unsatisfied_group,
                    'average_utility_gain_from_extra_winner' : average_utility_gain_from_extra_winner,
                    'utility_deviation' : utility_deviation,
                    'score_deviation' : score_deviation,
                    'favored_winner_deviation' : favored_winner_deviation,
                    'number_of_duplicates' : number_of_duplicates,
                    'average_winner_polarization' : average_winner_polarization,
                    'most_polarized_winner' : most_polarized_winner,
                    'least_polarized_winner' : least_polarized_winner
                    
                                        }

    S_norm = S_in.divide(K)
    S_winners = S_norm[winner_list]
    V = S_norm.shape[0]
    W = len(winner_list)
    #quota = W/V
    #Utility Metrics
    metrics['average_utility'][method] = S_winners.sum(axis=1).sum()  / V
    metrics['average_ln_utility'][method] = np.log1p(S_winners.sum(axis=1)).sum()  / V
    metrics['average_favored_winner_utility'][method] = S_winners.max(axis=1).sum()  / V
    metrics['average_unsatisfied_utility'][method] = sum([1-i for i in S_winners.sum(axis=1) if i < 1]) / V
    metrics['fully_satisfied_voters'][method] = sum([(i>=1) for i in S_winners.sum(axis=1)])  / V
    metrics['totally_unsatisfied_voters'][method] = sum([(i==0) for i in S_winners.sum(axis=1)])  / V

    #Represenation Metrics
    metrics['harmonic_quality'][method] = np.divide(S_winners.values , np.argsort(S_winners.values, axis=1) +1).sum()  / V
    metrics['unitary_quality'][method] = S_winners.divide((S_winners.sum() * W/V).clip(lower=1)).sum(axis = 1).clip(upper=1).sum() / V 
    metrics['ebert_cost'][method] = (S_winners.divide(S_winners.sum() * W/V).sum(axis = 1)**2).sum() / V 
    metrics['most_blocking_loser_capture'][method] = S_norm.gt((S_winners.sum(axis = 1)), axis=0).sum().max() / V 
    metrics['largest_total_unsatisfied_group'][method] = S_norm[S_winners.sum(axis = 1) == 0 ].astype(bool).sum(axis=0).max() / V
    metrics['average_utility_gain_from_extra_winner'][method] = S_norm.sub(S_winners.sum(axis = 1),axis = 0).clip(lower=0).sum(axis = 0).max() / V 

    #Equity Metrics
    metrics['utility_deviation'][method] = S_winners.sum(axis=1).std()
    metrics['score_deviation'][method] = S_winners.values.flatten().std()
    metrics['favored_winner_deviation'][method] = S_winners.max(axis=1).std()
    metrics['number_of_duplicates'][method] = len(winner_list) -len(set(winner_list))
    metrics['average_winner_polarization'][method] = S_winners.std(axis=0).mean()
    metrics['most_polarized_winner'][method] = S_winners.std(axis=0).max()
    metrics['least_polarized_winner'][method] = S_winners.std(axis=0).min()
    
    return   metrics


def plot_metric(df, Methods,axis,is_int = True):
    #plots metrics
    #colors = ['b','r','k','#FFFF00','g','#808080','#56B4E9','#FF7F00']
    colors = {'Jefferson' : '#FF7F00','Webster' : 'b', 'Allocate' : 'r','Unitary' : 'k',
              'Jefferson_KP' : '#FFFF00','Webster_KP' : 'm','Allocate_KP' : 'g','Unitary_KP' : '#808080'}
    styles = {'Utilitarian' : 'solid', 'STAR' : 'dashed', 'Hare_Ballots' : 'dotted'}
    bins = np.linspace(df.min().min(),df.max().max())
    for i, col in enumerate(df.columns):
        reweight = Methods[col]['Reweight']
        if Methods[col]['KP_Transform']: reweight = reweight + '_KP'
        selection = Methods[col]['Selection']
        count, bins, ignored = axis.hist(list(df[col]), bins = bins, color=colors[reweight],linestyle = styles[selection] ,histtype = 'step', label=col)
        if is_int:
            textstr = '$\mu=%.0f$\n$\sigma=%.0f$'%(df[col].mean(), df[col].std())
        else:
            textstr = '$\mu=%.3f$\n$\sigma=%.3f$'%(df[col].mean(), df[col].std())
        props = dict(boxstyle='round', facecolor='white', ec=colors[reweight],linestyle = styles[selection], alpha=0.8)
        axis.text(0.98, 0.95-0.1*i, textstr, transform=axis.transAxes, bbox=props, verticalalignment='top',horizontalalignment='right')
    return axis
