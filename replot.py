import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
import utils 

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
Methods['utilitarian_unitary_kp'] = {'Selection' : 'Utilitarian', 'Reweight' : 'Unitary', 'KP_Transform' : True}
Methods['hare_voters_unitary_kp'] = {'Selection' : 'Hare_Voters', 'Reweight' : 'Unitary', 'KP_Transform' : True}
Methods['hare_ballots_unitary_kp'] = {'Selection' : 'Hare_Ballots', 'Reweight' : 'Unitary', 'KP_Transform' : True}

Methods['utilitarian_jefferson_kp'] = {'Selection' : 'Utilitarian', 'Reweight' : 'Jefferson', 'KP_Transform' : True}
Methods['hare_voters_jefferson_kp'] = {'Selection' : 'Hare_Voters', 'Reweight' : 'Jefferson', 'KP_Transform' : True}
Methods['hare_ballots_jefferson_kp'] = {'Selection' : 'Hare_Ballots', 'Reweight' : 'Jefferson', 'KP_Transform' : True}

Methods['utilitarian_allocate_kp'] = {'Selection' : 'Utilitarian', 'Reweight' : 'Allocate', 'KP_Transform' : True}
Methods['hare_voters_allocate_kp'] = {'Selection' : 'Hare_Voters', 'Reweight' : 'Allocate', 'KP_Transform' : True}
Methods['hare_ballots_allocate_kp'] = {'Selection' : 'Hare_Ballots', 'Reweight' : 'Allocate', 'KP_Transform' : True}   


#path = '../November Results/'
#path = '../August Results/'
path = ''

#get dataframes stored as CSVs
try:
    df_total_utility = pd.read_csv(path + 'total_utility.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'total_utility.csv')  
try:
    df_total_ln_utility = pd.read_csv(path + 'total_ln_utility.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'total_ln_utility.csv')  
try:
    df_total_favored_winner_utility = pd.read_csv(path + 'total_favored_winner_utility.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'total_favored_winner_utility.csv')  
try:
    df_total_unsatisfied_utility = pd.read_csv(path + 'total_unsatisfied_utility.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'total_unsatisfied_utility.csv')  
try:
    df_fully_satisfied_voters = pd.read_csv(path + 'fully_satisfied_voters.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'fully_satisfied_voters.csv')  
try:
    df_totally_unsatisfied_voters = pd.read_csv(path + 'totally_unsatisfied_voters.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'totally_unsatisfied_voters.csv')  
try:
    df_utility_deviation = pd.read_csv(path + 'utility_deviation.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'utility_deviation.csv')  
try:
    df_score_deviation = pd.read_csv(path + 'score_deviation.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'score_deviation.csv')  
try:
    df_favored_winner_deviation = pd.read_csv(path + 'favored_winner_deviation.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'favored_winner_deviation.csv')  
try:
    df_number_of_duplicates = pd.read_csv(path + 'number_of_duplicates.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'number_of_duplicates.csv')  
try:
    df_average_winner_polarization = pd.read_csv(path + 'average_winner_polarization.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'average_winner_polarization.csv')  
try:
    df_most_polarized_winner = pd.read_csv(path + 'most_polarized_winner.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'most_polarized_winner.csv')  
try:
    df_least_polarized_winner = pd.read_csv(path + 'least_polarized_winner.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'least_polarized_winner.csv')  
try:
    df_parties = pd.read_csv(path + 'parties.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'parties.csv') 
    
#choose subset
method_all = list(Methods.keys())
method_subset = sorted([i for i in df_total_utility.columns if i in method_all])
#method_subset = ['hare_ballots_allocate', 'utilitarian_allocate','utilitarian_jefferson', 'utilitarian_unitary']

#make plots
fig = plt.figure(figsize=(15,20))
fig.suptitle('Utility Metrics')

try:
    ax1 = fig.add_subplot(3, 2, 1)
    ax1 = utils.plot_metric(df = df_total_utility[method_subset], Methods = Methods,axis=ax1,is_int = True)
    lgd1 = ax1.legend(loc=2)
    ax1.set_xlabel('Total Utility')
    ax1.set_ylabel('Records in bin')
except:
    pass
    
try:    
    ax2 = fig.add_subplot(3, 2, 2)
    ax2 = utils.plot_metric(df = df_total_ln_utility[method_subset], Methods = Methods,axis=ax2,is_int = True)
    lgd2 = ax2.legend(loc=2)
    ax2.set_xlabel('Total Log Utility')
    ax2.set_ylabel('Records in bin')
except:
    pass
    
try:      
    ax3 = fig.add_subplot(3, 2, 3)
    ax3 = utils.plot_metric(df = df_total_favored_winner_utility[method_subset], Methods = Methods,axis=ax3,is_int = True)
    lgd3 = ax3.legend(loc=2)
    ax3.set_xlabel('Total Favored Winner Utility')
    ax3.set_ylabel('Records in bin')
except:
    pass
    
try:      
    ax4 = fig.add_subplot(3, 2, 4)
    ax4 = utils.plot_metric(df = df_total_unsatisfied_utility[method_subset], Methods = Methods,axis=ax4,is_int = True)
    lgd4 = ax4.legend(loc=2)
    ax4.set_xlabel('Total Unsatisfied Utility')
    ax4.set_ylabel('Records in bin')
except:
    pass
    
try:      
    ax5 = fig.add_subplot(3, 2, 5)
    ax5 = utils.plot_metric(df = df_fully_satisfied_voters[method_subset], Methods = Methods,axis=ax5,is_int = True)
    lgd5 = ax5.legend(loc=2)
    ax5.set_xlabel('Fully Satisfied Voters')
    ax5.set_ylabel('Records in bin')
except:
    pass
    
try:      
    ax6 = fig.add_subplot(3, 2, 6)
    ax6 = utils.plot_metric(df = df_totally_unsatisfied_voters[method_subset], Methods = Methods,axis=ax6,is_int = True)
    lgd6 = ax6.legend(loc=2)
    ax6.set_xlabel('Totally Unsatisfied Voters')
    ax6.set_ylabel('Records in bin')
except:
    pass
    
fig.savefig(path + "Utility_Results.png",dpi = 300)

figB = plt.figure(figsize=(15,20))
figB.suptitle('Equity Metrics')

try:        
    axB1 = figB.add_subplot(3, 2, 1)
    axB1 = utils.plot_metric(df = df_utility_deviation[method_subset], Methods = Methods,axis=axB1,is_int = False)
    lgd2 = axB1.legend(loc=2)
    axB1.set_xlabel('Utility Deviation')
    axB1.set_ylabel('Records in bin')
except:
    pass
    
try:      
    axB2 = figB.add_subplot(3, 2, 2)
    axB2 = utils.plot_metric(df = df_score_deviation[method_subset], Methods = Methods,axis=axB2,is_int = False)
    lgd2 = axB2.legend(loc=2)
    axB2.set_xlabel('Score Deviation')
    axB2.set_ylabel('Records in bin')
except:
    pass
    
try:      
    axB3 = figB.add_subplot(3, 2, 3)
    axB3 = utils.plot_metric(df = df_favored_winner_deviation[method_subset], Methods = Methods,axis=axB3,is_int = False)
    lgd3 = axB3.legend(loc=2)
    axB3.set_xlabel('Favored Winner Deviation')
    axB3.set_ylabel('Records in bin')
except:
    pass
    
try:      
    axB4 = figB.add_subplot(3, 2, 4)
    axB4 = utils.plot_metric(df = df_number_of_duplicates[method_subset], Methods = Methods,axis=axB4,is_int = False)
    lgd4 = axB4.legend(loc=2)
    axB4.set_xlabel('Number of Duplicate Winners')
    axB4.set_ylabel('Records in bin')
except:
    pass
    
try:      
    axB5 = figB.add_subplot(3, 2, 5)
    axB5 = utils.plot_metric(df = df_average_winner_polarization[method_subset], Methods = Methods,axis=axB5,is_int = False)
    lgd5 = axB5.legend(loc=2)
    axB5.set_xlabel('Average Winner Polarization')
    axB5.set_ylabel('Records in bin')
except:
    pass
    
#try:      
    # This metric is not very useful since most systems have the same most polarized winner
    # axB5 = figB.add_subplot(3, 2, 5)
    # axB5 = utils.plot_metric(df = df_most_polarized_winner[method_subset], Methods = Methods,axis=axB5,is_int = False)
    # lgd5 = axB5.legend(loc=2)
    # axB5.set_xlabel('Most Polarized Winner')
    # axB5.set_ylabel('Records in bin')
# except:
#     pass
    
try:      
    axB6 = figB.add_subplot(3, 2, 6)
    axB6 = utils.plot_metric(df = df_least_polarized_winner[method_subset], Methods = Methods,axis=axB6,is_int = False)
    lgd6 = axB6.legend(loc=2)
    axB6.set_xlabel('Least Polarized Winner')
    axB6.set_ylabel('Records in bin')
except:
    pass
    
figB.savefig(path + "Equity_Results.png",dpi = 300)

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

try:
    axC2 = figC.add_subplot(1, 2, 2)
    axC2.scatter(df_parties['x'],df_parties['y'], marker='.', s=(df_parties['size']).astype('int'), c=df_parties['party_name'])
    axC2.set_xlim(-10, 10)
    axC2.set_ylim(-10, 10)
    axC2.set_xticks(range(-10, 11,2))
    axC2.set_yticks(range(-10, 11,2))
    axC2.set_title('Party Position')
    axC2.set_xlabel('Planned Economy  <--  Economics  -->  Free Market')
    axC2.set_ylabel('Liberal  <-- Government  --> Authoritarian')
except:
    pass
figC.savefig(path + "Simulated_Spectrum.png",dpi = 300)

print('done')