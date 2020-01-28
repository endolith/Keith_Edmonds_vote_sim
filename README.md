
This code will simulate scoring on candidates and apply electoral systems to evaluate winners winner sets produced by each method. The sequential systems are modular in the sense that each is composed of a selection and a reweigting method.

There are three selection methods: 'Utilitarian','STAR' and 'Hare_Ballots' . 
These choose winners from all candidates based on the voters scores.

There are four Reweight methods: 'Jefferson', 'Webster', 'Unitary' and 'Allocate'.
These alter the scores between rounds based on the winner to ensure an outcome which passes the standard criteria for Proportional Representation

All combinations are possible which means that there are 12 basic systems.

There is also the option of applying the KP transform to any system which means there are 24 possible systems. 

Since we know from the Gibbard-Satterthwaite theorem as well as the Balinski & Young's Impossibility Theorem there is no chance to make a perfect method. The task is then to mitigate the consequences of the flaws to get the best results in the end. To judge "best", many evaluation metrics are plotted for each system to compared under. 

# Simulation Methodology 

A 2D ideological space [-10,10] by [-10,10] is assumed. 10,000 voters are then randomly simulated in this space by them being members of parties. 2-7 parties are randomly selected and given a random position is the 2D space. The 10,000 voters are randomly assigned to each party and their distance from the party center is determined by a Gaussian distribution with a standard deviation between 0.5 and 2. Candidates are created at every grid point in the plane. They are not wanted to be random as we are trying to find the best system under optimal candidates. The score each voter gives to each candidate is determined from their euclidian distance, d, as score = 5 - 2.0*d with 5 being the maximum score. The score of the closest candidate is put to 5 to help make scores more realistic. We do not expect the distances or the method of deriving score to be particularly realistic. However, we do expect the distributions of score to span the space of reality. We do this simulation 25,000 times and compute several metrics for comparison. 

# System Details

All simulated systems are sequential score systems. They are all of the class where you select a winner and then apply a ballot or score reweighting mechanism. 

In any case this gives three coded selections methods:

    1. Utilitarian: Sum of score
    2. STAR: Top two Utilitarian then a pairwise runoff
    3. Hare Ballots: Sum of score in Hare quota of ballots

There are also three coded Reweight methods:

    1. Jeffereson: Reweight by 1/(1+SUM/MAX)
	2. Webster: Reweight by 1/(1+2*SUM/MAX)
    3. Unitary: Subtract SUM/MAX until exhausted
    4. Allocation: Exhaust whole ballots by allocating them to winners 

		*Note that both 3 and 4 require surplus handing and fractional surplus handling was done for both but in slightly differeing ways accordingto their theory.

Standard systems can be produced in this manner. For example Rewieghted Range voting is Utilitarian selection with Jefferson Reweighting. Calling the funtion get_winners() will return a list of winners.

For RRV this would be get_winners(S_in, Selection='Utilitarian', Reweight='Unitary', KP_Transform=False, W=5, K=5)
Where S_in is the score matrix, W is the number of winners and K is the max score. 

A KP-Transform can optionally be applied to any system.

# Evaluation Metrics
There are 6 metrics which are measures on Utility, 6 metrics which are measures on representation and 7 which are measures on variance/polarization/equity. Python code is included for clarity based on a normalized pandas dataframe of scores, S_norm, with the Candidates as the columns and one row for each Voter. There are W total winners and V total voters.

## Utility Metrics

### Average Utility

The average amount of score which each voter attributed to the winning candidates on their ballot. Higher values are better.

S_winners.sum(axis=1).sum()  / V

### Average Log Utility

The Average of ln(score spent) for each user. This is motivated by the ln of utility being thought of as more fair in a number of philosophical works. Higher values are better.

np.log1p(S_winners.sum(axis=1)).sum()  / V

### Average Favored Winner Utility

The Average utility of each voters most highly scored winner. This may not be their true favorite if they strategically vote but all of these metrics assume honest voting. Higher values are better with a max of 1.

S_winners.max(axis=1).sum()  / V

### Average Unsatisfied Utility

The Average of the total score for each user who did not get at least a total utility of MAX score. Lower values are better.

sum([1-i for i in S_winners.sum(axis=1) if i < 1]) / V

### Fully Satisfied Voters

The Fraction of voters who got candidates with a total score of MAX or more. In the single winner case getting somebody who you scored MAX would leave you satisfied. This translates to the multiwinner case if the one can assume that the mapping of score to Utility obeys Cauchy’s functional equation which essentially means that it is linear. Higher values are better with a max of 1.

sum([(i>=1) for i in S_winners.sum(axis=1)])  / V

### Totally Unsatisfied Voters

The Fraction of voters who did not score any winners. These are voters who had no influence on the election (other than the Hare Quota) so are wasted. Lower values are better.

sum([(i==0) for i in S_winners.sum(axis=1)])  / V

## Representation Metrics

### Harmonic Quality

A Theile based quality metric. Higher values are better.

https://rangevoting.org/QualityMulti.html

np.divide(S_winners.values , np.argsort(S_winners.values, axis=1) +1).sum()  / V

### Unitary Quality

A Monroe Based quality metric which maps score to utility linearly. Higher values are better.

https://electowiki.org/wiki/Vote_unitarity

S_winners.divide((S_winners.sum() * W/V).clip(lower=1)).sum(axis = 1).clip(upper=1).sum() / V 

### Ebert Cost 

A Phragmen based cost metric which minimizes the standard deviation of the loads. Lower values are better.

https://electowiki.org/wiki/Ebert%27s_Method

(S_winners.divide(S_winners.sum() * W/V).sum(axis = 1)**2).sum() / V 

### Most Blocking Loser Capture Fraction

This is basically the unelected candidate with the highest capture count over the whole winner set. It is a simple method for checking the stability for all S’ of size 1. Lower values are better. A value higher than 1/W implies a nonstable winner set.

https://electowiki.org/wiki/Stable_Winner_Set

S_norm.gt((S_winners.sum(axis = 1)), axis=0).sum().max() / V 

### Largest Totally Unsatisfied Group 

The Fraction of voters who did not get any winner but who are all voting for a nonwinner. This is basically the test for the simple Justified representation. Lower values are better. A value higher than 1/W implies a failure of Justified Representation if one assumes that everybody who is scored by a voter is approved.

https://electowiki.org/wiki/Justified_representation

S_norm[S_winners.sum(axis = 1) == 0 ].astype(bool).sum(axis=0).max() / V

### Average Utility Gain From Extra Winner

It may not be totally obvious but this is the same quantity as the prior if the score ballots are passed through the KP-Transform. Recall that Justified representation is not defined for score but approval. It also somewhat unfies the Most Blocking Loser Capture Fraction and Average Utility Gain From Extra Winner. Lower Values are better. A value higher than 1/W implies a problem.

S_norm.subtract(S_winners.sum() ).clip(lower=0).sum(axis = 0).max() / V 

## Variance/Polarization/Equity Metrics

### Utility Deviation

The standard deviation of the total utility for each voter. This is motivated by the desire for each voter to have a similar total utility. This could be thought of as Equity. Lower values are better.

S_winners.sum(axis=1).std()

### Score Deviation

The standard deviation of all the scores given to all winners. This is a measure of the polarization of the winner in aggregate. It is not known what a good value is for this but it can be useful for comparisons between systems.

S_winners.values.flatten().std()

### Favored Winner Deviation

The standard deviation of each users highest scored winner. It is somewhat of a check on what happens if the Cauchy’s functional equation is not really true. If the highest scored winner is a better estimate of the true happiness of the winner than the total score across winner. Lower values are better.

S_winners.max(axis=1).std()

### Number of Duplicates

The total nubmer of clones elected. The code currently allows for clones to be reelected. Ideally this would not happen if there are enough candidates. This gives a measure of the ability to find minority representors. Lower is better.

len(winner_list) - len(set(winner_list))

### Average Winner Polarization

The standard deviation of each winner across all voters averaged across all winners. The polarization of a winner can be thought of as how similar the scores for them are across all voters.

S_winners.std(axis=0).mean()

### Most Polarized Winner 

The highest standard deviation of the winners across voters. The winner who has the highest standard deviation/polarization. This is not plotted since it is basically the same for all methods

S_winners.std(axis=0).max()

### Least Polarized Winner

The lowest standard deviation of the winners across voters. The winner who has the lowest standard deviation/polarization.

S_winners.std(axis=0).min()

#Notes

## Public Discussions 

https://forum.electionscience.org/t/different-reweighting-for-rrv-and-the-concept-of-vote-unitarity 

https://forum.electionscience.org/t/utilitarian-sum-vs-monroe-selection

https://forum.electionscience.org/t/re-what-are-the-best-ways-to-do-allocated-cardinal-pr-in-public-elections

https://forum.electionscience.org/t/wolf-committee-results/

https://forum.electionscience.org/t/wholistic-metrics-on-winner-sets


## Types of problems to handle:
https://www.rangevoting.org/AssetBC.html

https://groups.google.com/forum/#!topic/electionscience/Rk4ZGf-s-s8

## Possible Future additions


### Sequential Phragmen

https://electowiki.org/wiki/Sequential_Phragmen

This is an Approval method so it would need the KP transform. There are also a few different ways to do a sequential phragmen model such as https://electowiki.org/wiki/Sequential_Ebert

### Excess method

https://as.nyu.edu/content/dam/nyu-as/faculty/documents/Excess%20Method%20(final).pdf

This is an Approval method so it would need the KP transform

### Harmonic Voting

https://www.rangevoting.org/QualityMulti.html

This like any other optimal system is likely too computationally expensive


### Single Transferable vote

This simulation does not really lend itself simply to STV. Score can be turned into rank so that we can use the same input. Also, all the comparison metrics only need the winner set so they will be comparable. The larger issue is that there are many candidates and I allow for clones so effectively infinite. 

