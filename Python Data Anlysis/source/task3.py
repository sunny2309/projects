import numpy as np
from numpy.random import shuffle
import math

#excercise 1 : generate a standard deck of playing cards
def generate_full_deck(): # this function get the full deck of cards
    full_deck = []
    for i in range(1, 14):
        for cnt in range(4):
            full_deck.append(i)  # build the deck array
    return full_deck
            
full_deck = generate_full_deck()  # call this function for testing
print("\nexcercise : 1")
print("=====================")
print("The Full Deck is : {0}".format(full_deck))
print("The total value of full deck is : {0}".format(sum(full_deck)))

#excercise 2 : perform shuffle operation
def do_shufffle():  # this funcion do shuffle by numpy
    rnd = np.random.RandomState(123) ## Introduced this step to produce same results each time.
    rnd.shuffle(full_deck) # do the shuffling with full deck of cards

def deal_half_deck(): # this function deals the cards to 2 players
    
    player1_hand = []
    player2_hand = []

    for i in range(len(full_deck)):  # for all cards
        card = full_deck[i]
        if i % 2 == 0:
            player1_hand.append(card)  # assign the even cards to player1
        else:
            player2_hand.append(card) # assign the odd cards to player1
    return player1_hand, player2_hand

do_shufffle()   # do shuffle
player1_hand,  player2_hand = deal_half_deck()   # deal the decks of cards to 2 players
print("\nexcercise : 2")
print("=====================")
print("The shuffled full Deck is : {0}".format(full_deck))
print("The hand of player 1 is : {0}".format(player1_hand))
print("The hand of player 2 is : {0}".format(player2_hand))

# excercise 3 : Simulate a game of Addition War between two players
"""
1. Shuffle deck
2. Deal half of deck to each player
3. They lay down their card one(2 at a time) by one ((2 at a time))
4. the winner of each round will have their cards and opposite's cards
5. In case of a tie in a particular round, cards for both players will be kept in a seperate deck until tie is broken by one player winning any further round. All cards kept in separate deck till tie is broken will be given to winning player.
6. In case last few rounds turn into tie then that cards will have equal weight for both player hence previousy played round will decide winner.(E.G. Last 2 or 1 round are tie then first 11 or 12 round winners will be winner.)
7. Game terminates after a maximum of 13 rounds (occurs when there are no ties) and the deck is
exhausted.
8. The player with the highest total card points (in their discard pile) wins the game.
"""
print("\nexcercise : 3")
print("=====================")
def play():  # this plays the cards game
    do_shufffle()    # do the shuffling
    player1_hand,  player2_hand = deal_half_deck()    # deal the cards to two players
    
    player1_pile,player2_pile,tie_pile, round_wins_track = [],[],[],[]
    for round in range(1, 26, 2):  # for each round
        player1_val = player1_hand[round - 1], player1_hand[round]  # calculate the player1's value
        player2_val = player2_hand[round - 1], player2_hand[round] # calculate the player2's value
        if sum(player1_val) > sum(player2_val):  # check if the player1 is won in this round
            if tie_pile: # We append tie cards first and then winning round cards.
                player1_pile.extend(tie_pile)
                tie_pile = [] ## Emptying tie deck as tie has resolved with winning of player 1
            player1_pile.extend(player1_val) # add their cards to pile
            player1_pile.extend(player2_val) # add ohter's cards to pile
            round_wins_track.append('P1 Wins')
        elif sum(player1_val) == sum(player2_val): ## Tie case. We maintain tie_pile until tie is solved by next rounds
            tie_pile.extend(player1_val)
            tie_pile.extend(player2_val)
            round_wins_track.append('Tie')
        else:                        # check if the player2 is won in this round
            if tie_pile: # We append tie cards first and then winning round cards.
                player2_pile.extend(tie_pile)
                tie_pile = [] ## Emptying tie deck as tie has resolved with winning of player 2
            player2_pile.extend(player1_val)  # add their cards to pile
            player2_pile.extend(player2_val)     # add ohter's cards to pile
            round_wins_track.append('P2 Wins')
    if tie_pile: ## This is like worst case scenario where after tie all rounds are over but still tie is there.
        for i in range(0, len(tie_pile),4):
            player1_pile.extend([tie_pile[i],tie_pile[i+1]])
            player2_pile.extend([tie_pile[i+2],tie_pile[i+3]])
    if sum(player1_pile) > sum(player2_pile) :  # we select the winner who has more values in their pile
        return 'player1', sum(player1_pile), player1_hand, player1_pile, 'player2', sum(player2_pile), player2_hand, player2_pile, round_wins_track
    elif sum(player1_pile) == sum(player2_pile) : #Added tie case here as well in worst-case where all vals match in both players.
        return 'Tie', sum(player1_pile), player1_hand, player1_pile, 'Tie', sum(player2_pile), player2_hand, player2_pile, round_wins_track
    else:
        return 'player2', sum(player2_pile), player2_hand, player2_pile, 'player1', sum(player1_pile), player1_hand, player1_pile, round_wins_track

winner , winning_score, init_winner_pile, winner_pile, loser, losing_score, init_loser_pile, loser_pile, round_wins_track = play()  # this test the playing of game

if winner == 'Tie': ## Tie throughout with different cominations or same in worst case.
    print('Round Winners Track : ')
    print(round_wins_track)
    print("\nTie : {0} , Player 1 Score : {1}, Player 2 Score : {2}".format(winner, winning_score,losing_score))
    
else:
    print('Round Winners Track : ')
    print(round_wins_track)
    print("\nWinner : {0} , Score : {1}".format(winner, winning_score))

# excercise 4 : some analysis of the gameplay
print("\nexcercise : 4")
print("=====================")
'''
Q 4.1 : Extract the final scores for the winning and losing players, and produce descriptive statistics for each scenario, specifically the minimum, mean, and maximum totals after each game. What do you observe about the winning and losing totals??

Ans 4.1 : From minimum, mean and maximum final score we, can conclude that scores stays in range 190-250 for winning side while losing side losing side end up with scores from 100-175.
'''
print('\n----------Answers for Step 4.1 : -----------\n')
n = 20

init_winner_piles,init_loser_piles = [],[]
init_winner_scores,init_loser_scores = [],[]
winner_final_piles,loser_final_piles = [],[]
winner_final_scores,loser_final_scores = [],[]
games_with_winner_having_lower_initial_total = 0
lowest_init_by_winner, highest_init_by_loser = [], []

for i in range(n):   # for 20 simulation of game
    winner,winning_score,init_winner_pile,winner_final_pile,loser,losing_score,init_loser_pile,loser_final_pile,round_wins_track = play()
    
    init_winner_piles.append(init_winner_pile)
    init_winner_scores.append(sum(init_winner_pile))
    winner_final_piles.append(winner_pile)  # build the winning piles
    winner_final_scores.append(winning_score)  # build the winning scores
    
    init_loser_piles.append(init_loser_pile)
    init_loser_scores.append(sum(init_loser_pile))
    loser_final_piles.append(loser_pile)   # build the losing piles
    loser_final_scores.append(losing_score)  # build the losing scores
    
    print('Game : %02d : Winner: %7s, Min(Loser Score)-%3d, Max(Winner Score)-%3d, Mean-%.2f'%\
            (i+1, winner, losing_score, winning_score, (losing_score+winning_score)/2))
    
    if sum(init_winner_pile) < sum(init_loser_pile):
        games_with_winner_having_lower_initial_total += 1
        lowest_init_by_winner.append(sum(init_winner_pile))
        highest_init_by_loser.append(sum(init_loser_pile))
        
'''
Step 4.2: What proportion of games resulted in a winner that had a lower initial total than their opponent?
What was the lowest initial total to win a game (or conversely, the highest initial total to lose a game)?

Ans 4.2 : We can observer 0.35 proportion of game resulted in winner that had a lower inital total than their opponent.
174 was lowest inital total to win a game.183 was highest initial total to lose a game.
We can observer that when inital totals are in range 170-190 then only game changes. If they are far separated then chances goes into favour of player with high total.
'''
print('\n------------- Answers for Step 4.2 : -------------------\n')
print('Proportion of games where winner had a lower initial total than their opponent : %.2f(%d/20)'%(games_with_winner_having_lower_initial_total/20, games_with_winner_having_lower_initial_total))
print('Lowest Initial to win a game : %d'%min(lowest_init_by_winner))
print('Highest Initial to lose a game : %d'%min(highest_init_by_loser))

'''
Step 4.3: Calculate the (linear) correlation between the initial and final totals for winners vs. losers? What do
you observe about the respective correlations, and what do these correlations tell you about the relationship between the initial and final totals?

Ans 4.3 : We can notice that there is strong correlation between inital and final totals for winner and loser.Initial values plays abig role in deciding winning side. When inital totals are in range 170-190 then only game changes. If they are far separated then chances goes into favour of player with high total.
'''

print('\n ------------ Answers for Step 4.3 : -------------------\n')

corr_winn = np.corrcoef(init_winner_scores, winner_final_scores)
print("Linear Correlation between initial and final totals for winner : {0}".format(corr_winn[0,1]))
corr_lose = np.corrcoef(init_loser_scores, loser_final_scores)
print("Linear Correlation between initial and final totals for Loser : {0}".format(corr_lose[0,1]))
