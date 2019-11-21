This code will simulate scoring and methods to sequantially evaluate scores to get a list of winnere

There are three selection methods: 'Utilitarian','Hare_Voters' and 'Hare_Ballots' . 
These choose winners from all candidates based on the voters scores.

There are three Reweight methods: 'Unitary', 'Jefferson' and 'Allocate'.
These alter the scores between rounds based on the winner to ensure an outcome which passes the standard criteria for Proportional Representation

All combinations are possible which means that there are 9 possible models.

There is also the option of applying the KP transform to any system which means there are 18 possbile systems. 

There are 6 metrics which are measures on Utility and 7 which are measures on variance/polarization/equity. I will include a python code for each based on a pandas dataframe of scores, S, with the winners as the columns and one row for each winner.

~Utility Metrics~

Total Utility
The total amount of score which each voter attributed to the winning candidates on their ballot. Higher values are better.
S.sum(axis=1).sum()

Total Log Utility
The sum of ln(score spent) for each user. This is motivated by the ln of utility being thought of as more fair in a number of philosophical works. Higher values are better.
np.log1p(S.sum(axis=1)).sum()

Total Favored Winner Utility
The total utility of each voters most highly scored winner. This may not be their true favorite if they strategically vote but all of these metrics assume honest voting. Higher values are better.
S.max(axis=1).sum()

Total Unsatisfied Utility
The sum of the total score for each user who did not get at least a total utility of MAX score. Lower values are better.
sum([1-i for i in S.sum(axis=1) if i < 1])
NOTE: The scores are normalized so MAX = 1

Fully Satisfied Voters
The number of voters who got candidates with a total score of MAX or more. In the single winner case getting somebody who you scored MAX would leave you satisfied. This translates to the multiwinner case if the one can assume that the mapping of score to Utility obeys Cauchy’s functional equation which essentially means that it is linear. Higher values are better.
sum([(i>=1) for i in S.sum(axis=1)])

Wasted Voters
The number of voters who did not score any winners. These are voters who had no influence on the election (other than the Hare Quota) so are wasted. Lower values are better.
sum([(i==0) for i in S.sum(axis=1)])

~Variance/Polarization/Equity Metrics~

Utility Deviation
The standard deviation of the total utility for each voter. This is motivated by the desire for each voter to have a similar total utility. This could be thought of as Equity. Lower values are better.
S[winner_list].sum(axis=1).std()

Score Deviation
The standard deviation of all the scores given to all winners. This is a measure of the polarization of the winner in aggregate. It is not known what a good value is for this but it can be useful for comparisons between systems.
S.values.flatten().std()

Favored Winner Deviation
The standard deviation of each users highest scored winner. It is somewhat of a check on what happens if the Cauchy’s functional equation is not really true. If the highest scored winner is a better estimate of the true happyness of the winner than the total score across winner. Lower values are better.
S.max(axis=1).std()

Average Winner Polarization
The standard deviation of each winner across all voters averaged across all winners. The polarization of a winner can be thought of as how similar the scores for them are across all voters.
S.std(axis=0).mean()

Number of Duplicates
The code currently allows for clones to be relected. Ideally this would not happen if there are enough candidates. This gives a mesure of the ability to find minority representors. Lower is better.
len(winner_list) -len(set(winner_list))

Most Polarized Winner 
The highest standard deviation of the winners across voters. The winner who has the highest standard deviation/polarization. This is not plotted since it is basically the same for all methods
S.std(axis=0).max()

Least Polarized Winner
The lowest standard deviation of the winners across voters. The winner who has the loweststandard deviation/polarization.
S.std(axis=0).min()

Types of problems to handle:
https://www.rangevoting.org/AssetBC.html
https://groups.google.com/forum/#!topic/electionscience/Rk4ZGf-s-s8

Discussion: https://forum.electionscience.org/t/utilitarian-sum-vs-monroe-selection/355/8


Possible Future additions:

Excess method:
https://as.nyu.edu/content/dam/nyu-as/faculty/documents/Excess%20Method%20(final).pdf
This is an Approval method so it would need the KP transform

Harmonic Voting:
https://www.rangevoting.org/QualityMulti.html
This like any other optimal system is likely too computationally expensive

