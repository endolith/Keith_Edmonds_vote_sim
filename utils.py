import pandas as pd
import numpy as np

# function to turn scores into a winner set for various systems
def get_winners(S_in,Selection = 'Utilitarian',Reweight = 'Unitary', KP_Transform = False, W=5.0, K=5):

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

    V = S_wrk.shape[0]

    #make copy of working scores
    S_orig = S_wrk.copy()
    
    #These only matter for specific systems and are initialized here
    ballot_weight = np.ones(V)

    #Populate winners in a loop
    winner_list = []
    while len(winner_list) < W:
        #round
        #R = len(winner_list)

        #select winner
        #w = index with highest selection metric
        if Selection == 'Hare_Voters':
            w = pd.DataFrame(np.sort(S_wrk.values, axis=0), columns=S_wrk.columns).tail(round(V/W)).sum().idxmax()
        elif Selection == 'Hare_Ballots':
            max_score = 0.0
            #Find candidate with the heighest vote sum in a hare quota of ballot weight
            for candidate in S_wrk.columns:
                cand_df = S_wrk[[candidate]].copy()
                cand_df['ballot_weight'] = ballot_weight
                cand_df = cand_df.sort_values(by=[candidate], ascending=False)
                c_score = cand_df[cand_df['ballot_weight'].cumsum() <= V/W][candidate].sum()
                if c_score > max_score:
                    max_score = c_score
                    w = candidate
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
            #Ballot weight as defined by the Jeffereson method
            ballot_weight = 1/(total_sum + 1)
            S_wrk = S_orig.mul(ballot_weight, axis = 0)
            
        elif Reweight == 'Allocate':   
            votes_to_allocate = round(V/W)
            cand_df = S_wrk[[w]].copy()
            cand_df['ballot_weight'] = ballot_weight
            cand_df_sort = cand_df.sort_values(by=[w], ascending=False)
            split_point = cand_df_sort[cand_df_sort['ballot_weight'].cumsum() <= V/W][w].iloc[-1]
            
            if split_point>0:
                voters_on_split = cand_df[cand_df[w] == split_point]['ballot_weight'].sum()
                voters_allocated = cand_df[cand_df[w] > split_point]['ballot_weight'].sum()
                
                reweighted_value = (votes_to_allocate - voters_allocated)/voters_on_split
                
                cand_df.loc[cand_df[w] == split_point, 'ballot_weight'] = cand_df.loc[cand_df[w] == split_point, 'ballot_weight'] * reweighted_value 

            cand_df.loc[cand_df[w] >split_point, 'ballot_weight'] = 0            
                                    
            ballot_weight = cand_df['ballot_weight'] 
            S_wrk = S_orig.mul(ballot_weight, axis = 0)

    return winner_list

#Method to get all output quality metrics for a winner set

def get_metrics(S_in,metrics,winner_list,method,K=5):
    
    #store metrics for each method
    if not metrics:
        total_utility = {}
        total_ln_utility = {}
        total_favored_winner_utility = {}
        total_unsatisfied_utility = {}
        fully_satisfied_voters = {}
        wasted_voters = {}
        utility_deviation = {}
        score_deviation = {}
        favored_winner_deviation = {}
        average_winner_polarization = {}
        most_polarized_winner = {}
        least_polarized_winner = {}
        
        metrics = {
                    'total_utility' : total_utility,
                    'total_ln_utility' : total_ln_utility,
                    'total_favored_winner_utility' : total_favored_winner_utility,
                    'total_unsatisfied_utility' : total_unsatisfied_utility,
                    'fully_satisfied_voters' : fully_satisfied_voters,
                    'wasted_voters' : wasted_voters,
                    'utility_deviation' : utility_deviation,
                    'score_deviation' : score_deviation,
                    'favored_winner_deviation' : favored_winner_deviation,
                    'average_winner_polarization' : average_winner_polarization,
                    'most_polarized_winner' : most_polarized_winner,
                    'least_polarized_winner' : least_polarized_winner,
                                        }

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
    
    
def plot_metric(df, Methods,axis,is_int = True):
    #plots metrics
    #colors = ['b','r','k','#FFFF00','g','#808080','#56B4E9','#FF7F00']
    colors = {'Jefferson' : 'b','Allocate' : 'r','Unitary' : 'k','Jefferson_KP' : 'g','Allocate_KP' : '#FF7F00','Unitary_KP' : '#FFFF00'}
    styles = {'Utilitarian' : 'solid', 'Hare_Voters' : 'dashed', 'Hare_Ballots' : 'dotted'}
    for i, col in enumerate(df.columns):
        reweight = Methods[col]['Reweight']
        if Methods[col]['KP_Transform']: reweight = reweight + '_KP'
        selection = Methods[col]['Selection']
        count, bins, ignored = axis.hist(list(df[col]), 50, color=colors[reweight],linestyle = styles[selection] ,histtype = 'step', label=col)
        if is_int:
            textstr = '$\mu=%.0f$\n$\sigma=%.0f$'%(df[col].mean(), df[col].std())
        else:
            textstr = '$\mu=%.3f$\n$\sigma=%.3f$'%(df[col].mean(), df[col].std())
        props = dict(boxstyle='round', facecolor='white', ec=colors[reweight],linestyle = styles[selection], alpha=0.8)
        axis.text(0.98, 0.95-0.1*i, textstr, transform=axis.transAxes, bbox=props, verticalalignment='top',horizontalalignment='right')
    return axis   
     