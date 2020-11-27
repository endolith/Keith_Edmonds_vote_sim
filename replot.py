import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
import utils 

Methods = {}

Methods['utilitarian_scale_score'] = {'Selection' : 'Utilitarian', 'Reweight' : 'Scale Score', 'KP_Transform' : False} #Sequentially Spent Score with Capping
# Methods['STAR_scale_score'] = {'Selection' : 'STAR', 'Reweight' : 'Scale Score', 'KP_Transform' : False}
# Methods['hare_ballots_scale_score'] = {'Selection' : 'Hare_Ballots', 'Reweight' : 'Scale Score', 'KP_Transform' : False}

Methods['utilitarian_cap_score'] = {'Selection' : 'Utilitarian', 'Reweight' : 'Cap Score', 'KP_Transform' : False} #Sequentially Spent Score With Scaling
# Methods['STAR_cap_score'] = {'Selection' : 'STAR', 'Reweight' : 'Cap Score', 'KP_Transform' : False}
# Methods['hare_ballots_cap_score'] = {'Selection' : 'Hare_Ballots', 'Reweight' : 'Cap Score', 'KP_Transform' : False}

# Methods['utilitarian_jefferson'] = {'Selection' : 'Utilitarian', 'Reweight' : 'Jefferson', 'KP_Transform' : False} #Reweighted Range Voting Jefferson
# Methods['STAR_jefferson'] = {'Selection' : 'STAR', 'Reweight' : 'Jefferson', 'KP_Transform' : False}
# Methods['hare_ballots_jefferson'] = {'Selection' : 'Hare_Ballots', 'Reweight' : 'Jefferson', 'KP_Transform' : False}

Methods['utilitarian_Webster'] = {'Selection' : 'Utilitarian', 'Reweight' : 'Webster', 'KP_Transform' : False} #Reweighted Range Voting Webster
# Methods['STAR_Webster'] = {'Selection' : 'STAR', 'Reweight' : 'Webster', 'KP_Transform' : False}
# Methods['hare_ballots_Webster'] = {'Selection' : 'Hare_Ballots', 'Reweight' : 'Webster', 'KP_Transform' : False}

Methods['utilitarian_allocate'] = {'Selection' : 'Utilitarian', 'Reweight' : 'Allocate', 'KP_Transform' : False} #Allocated Score sorted by original
# Methods['STAR_allocate'] = {'Selection' : 'STAR', 'Reweight' : 'Allocate', 'KP_Transform' : False}
Methods['hare_ballots_allocate'] = {'Selection' : 'Hare_Ballots', 'Reweight' : 'Allocate', 'KP_Transform' : False} #Sequential Monroe

Methods['utilitarian_allocate_current'] = {'Selection' : 'Utilitarian', 'Reweight' : 'Allocate Current', 'KP_Transform' : False} #Allocated Score sorted by current
# Methods['STAR_allocate_current'] = {'Selection' : 'STAR', 'Reweight' : 'Allocate Current', 'KP_Transform' : False}
#Methods['hare_ballots_allocate_current'] = {'Selection' : 'Hare_Ballots', 'Reweight' : 'Allocate Current', 'KP_Transform' : False} 

# Methods['utilitarian_scale_score_kp'] = {'Selection' : 'Utilitarian', 'Reweight' : 'Scale Score', 'KP_Transform' : True}
# Methods['STAR_scale_score_kp'] = {'Selection' : 'STAR', 'Reweight' : 'Scale Score', 'KP_Transform' : True}
# Methods['hare_ballots_scale_score_kp'] = {'Selection' : 'Hare_Ballots', 'Reweight' : 'Scale Score', 'KP_Transform' : True}

# Methods['utilitarian_cap_score_kp'] = {'Selection' : 'Utilitarian', 'Reweight' : 'Cap Score', 'KP_Transform' : True}
# Methods['STAR_cap_score_kp'] = {'Selection' : 'STAR', 'Reweight' : 'Cap Score', 'KP_Transform' : True}
# Methods['hare_ballots_cap_score_kp'] = {'Selection' : 'Hare_Ballots', 'Reweight' : 'Cap Score', 'KP_Transform' : True}

# Methods['utilitarian_jefferson_kp'] = {'Selection' : 'Utilitarian', 'Reweight' : 'Jefferson', 'KP_Transform' : True}
# Methods['STAR_jefferson_kp'] = {'Selection' : 'STAR', 'Reweight' : 'Jefferson', 'KP_Transform' : True}
# Methods['hare_ballots_jefferson_kp'] = {'Selection' : 'Hare_Ballots', 'Reweight' : 'Jefferson', 'KP_Transform' : True}

# Methods['utilitarian_Webster_kp'] = {'Selection' : 'Utilitarian', 'Reweight' : 'Webster', 'KP_Transform' : True}
# Methods['STAR_Webster_kp'] = {'Selection' : 'STAR', 'Reweight' : 'Webster', 'KP_Transform' : True}
# Methods['hare_ballots_Webster_kp'] = {'Selection' : 'Hare_Ballots', 'Reweight' : 'Webster', 'KP_Transform' : True}
#
#Methods['utilitarian_allocate_kp'] = {'Selection' : 'Utilitarian', 'Reweight' : 'Allocate', 'KP_Transform' : True}
# Methods['STAR_allocate_kp'] = {'Selection' : 'STAR', 'Reweight' : 'Allocate', 'KP_Transform' : True}
#Methods['hare_ballots_allocate_kp'] = {'Selection' : 'Hare_Ballots', 'Reweight' : 'Allocate', 'KP_Transform' : True}

#path = '../February Results/'
#path = '../August Results/'
path = '../January Results/'
#path = ''

#get dataframes stored as CSVs
try:
    df_average_utility = pd.read_csv(path + 'average_utility.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'average_utility.csv')  
try:
    df_average_ln_utility = pd.read_csv(path + 'average_ln_utility.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'average_ln_utility.csv')  
try:
    df_average_favored_winner_utility = pd.read_csv(path + 'average_favored_winner_utility.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'average_favored_winner_utility.csv')  
try:
    df_average_unsatisfied_utility = pd.read_csv(path + 'average_unsatisfied_utility.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'average_unsatisfied_utility.csv')  
try:
    df_fully_satisfied_voters = pd.read_csv(path + 'fully_satisfied_voters.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'fully_satisfied_voters.csv')  
try:
    df_totally_unsatisfied_voters = pd.read_csv(path + 'totally_unsatisfied_voters.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'totally_unsatisfied_voters.csv')  
try:
    df_harmonic_quality = pd.read_csv(path + 'harmonic_quality.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'harmonic_quality.csv')  
try:
    df_unitary_quality = pd.read_csv(path + 'unitary_quality.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'unitary_quality.csv')  
try:
    df_ebert_cost = pd.read_csv(path + 'ebert_cost.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'ebert_cost.csv')  
try:
    df_most_blocking_loser_capture = pd.read_csv(path + 'most_blocking_loser_capture.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'most_blocking_loser_capture.csv')  
try:
    df_largest_total_unsatisfied_group = pd.read_csv(path + 'largest_total_unsatisfied_group.csv')
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'largest_total_unsatisfied_group.csv')  
try:
    df_average_utility_gain_from_extra_winner = pd.read_csv(path + 'average_utility_gain_from_extra_winner.csv')   
except FileNotFoundError:
    print('WARNING: Could not find ' + path + 'average_utility_gain_from_extra_winner.csv')  
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
method_subset = sorted([i for i in df_average_utility.columns if i in method_all])
#method_subset = ['hare_ballots_allocate', 'utilitarian_allocate','utilitarian_jefferson', 'utilitarian_unitary']
method_subset = ['hare_ballots_allocate','utilitarian_unitary']

#make plots
fig = plt.figure(figsize=(15,20))
fig.suptitle('Utility Metrics')

try:
    ax1 = fig.add_subplot(3, 2, 1)
    ax1 = utils.plot_metric(df = df_average_utility[method_subset], Methods = Methods,axis=ax1,is_int = False)
    lgd1 = ax1.legend(loc=2)
    ax1.set_xlabel('Average Utility')
    ax1.set_ylabel('Records in bin')
except:
    pass
    
try:    
    ax2 = fig.add_subplot(3, 2, 2)
    ax2 = utils.plot_metric(df = df_average_ln_utility[method_subset], Methods = Methods,axis=ax2,is_int = False)
    lgd2 = ax2.legend(loc=2)
    ax2.set_xlabel('Average Log Utility')
    ax2.set_ylabel('Records in bin')
except:
    pass
    
try:      
    ax3 = fig.add_subplot(3, 2, 3)
    ax3 = utils.plot_metric(df = df_average_favored_winner_utility[method_subset], Methods = Methods,axis=ax3,is_int = False)
    lgd3 = ax3.legend(loc=2)
    ax3.set_xlabel('Average Favored Winner Utility')
    ax3.set_ylabel('Records in bin')
except:
    pass
    
try:      
    ax4 = fig.add_subplot(3, 2, 4)
    ax4 = utils.plot_metric(df = df_average_unsatisfied_utility[method_subset], Methods = Methods,axis=ax4,is_int = False)
    lgd4 = ax4.legend(loc=2)
    ax4.set_xlabel('Average Unsatisfied Utility')
    ax4.set_ylabel('Records in bin')
except:
    pass
    
try:      
    ax5 = fig.add_subplot(3, 2, 5)
    ax5 = utils.plot_metric(df = df_fully_satisfied_voters[method_subset], Methods = Methods,axis=ax5,is_int = False)
    lgd5 = ax5.legend(loc=2)
    ax5.set_xlabel('Fully Satisfied Voters')
    ax5.set_ylabel('Records in bin')
except:
    pass
    
try:      
    ax6 = fig.add_subplot(3, 2, 6)
    ax6 = utils.plot_metric(df = df_totally_unsatisfied_voters[method_subset], Methods = Methods,axis=ax6,is_int = False)
    lgd6 = ax6.legend(loc=2)
    ax6.set_xlabel('Totally Unsatisfied Voters')
    ax6.set_ylabel('Records in bin')
except:
    pass
    
fig.savefig(path + "Utility_Results.png",dpi = 300)


figA = plt.figure(figsize=(15,20))
figA.suptitle('Represenation Metrics')

try:  
    axA1 = figA.add_subplot(3, 2, 1)
    axA1 = utils.plot_metric(df = df_harmonic_quality[method_subset], Methods = Methods,axis=axA1,is_int = False)
    lgd2 = axA1.legend(loc=2)
    axA1.set_xlabel('Harmonic Quality')
    axA1.set_ylabel('Records in bin')
except:
    pass
    
try:     
    axA2 = figA.add_subplot(3, 2, 2)
    axA2 = utils.plot_metric(df = df_unitary_quality[method_subset], Methods = Methods,axis=axA2,is_int = False)
    lgd2 = axA2.legend(loc=2)
    axA2.set_xlabel('Unitary Quality')
    axA2.set_ylabel('Records in bin')
except:
    pass
    
try:     
    axA3 = figA.add_subplot(3, 2, 3)
    axA3 = utils.plot_metric(df = df_ebert_cost[method_subset], Methods = Methods,axis=axA3,is_int = False)
    lgd3 = axA3.legend(loc=2)
    axA3.set_xlabel('Ebert Cost')
    axA3.set_ylabel('Records in bin')
except:
    pass
    
try:     
    axA4 = figA.add_subplot(3, 2, 4)
    axA4 = utils.plot_metric(df = df_most_blocking_loser_capture[method_subset], Methods = Methods,axis=axA4,is_int = False)
    lgd4 = axA4.legend(loc=2)
    axA4.set_xlabel('Most Blocking Loser Capture')
    axA4.set_ylabel('Records in bin')
except:
    pass
    
try:     
    axA5 = figA.add_subplot(3, 2, 5)
    axA5 = utils.plot_metric(df = df_largest_total_unsatisfied_group[method_subset], Methods = Methods,axis=axA5,is_int = False)
    lgd5 = axA5.legend(loc=2)
    axA5.set_xlabel('Largest Totally Unsatisfied Group')
    axA5.set_ylabel('Records in bin')
except:
    pass
    
try:     
    axA6 = figA.add_subplot(3, 2, 6)
    axA6 = utils.plot_metric(df = df_average_utility_gain_from_extra_winner[method_subset], Methods = Methods,axis=axA6,is_int = False)
    lgd6 = axA6.legend(loc=2)
    axA6.set_xlabel('Average Utility Gain From Extra Winner')
    axA6.set_ylabel('Records in bin')
except:
    pass
    
figA.savefig(path + "Representation_Results.png",dpi = 300)


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