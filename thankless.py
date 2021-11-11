import numpy as np
import itertools
from collections import deque 
from collections import OrderedDict
import random 
import copy
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Activation
from tensorflow.keras.optimizers import Adam
import keras
import matplotlib.pyplot as plt

def sigmoid(x, x0, k):
  return 1 / (1 + np.exp(-k*(x-x0)))

def score_stats(scores):
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    winning_scores = np.max(scores, axis=1)
    mean_winning_score = np.mean(winning_scores)
    std_winning_score = np.std(winning_scores)

    print("Mean score: " + str(mean_score) + " Standard deviation: " + str(std_score))
    print("Mean winning score: " + str(mean_winning_score) + " Standard deviation: " + str(std_winning_score))

def winner_stats(scores):
    winners = np.argmax(scores, axis=1)
    pos, wins = np.unique(winners, return_counts=True)
    windict = dict(zip(pos, 100*wins/len(scores)))
    print("Win Rate for each player position: " + str(windict))

class NoThanksController(object):
    """Basic controller for playing multiple games of NoThanks"""
    def __init__(self):
        self.all_history = []
        self.winner_history = []
    
    def play_games(self, players, number_of_games, silent=True):
        score_history = []
        game = NoThanks(players, silent=silent)
        for i in range(number_of_games):
            game.reset()
            game.play()
            score_history.append(game.get_scores())
        return score_history

class GameState(object):
    """Generic model that an agent uses for representing the game state"""
    pass

class SimpleState(GameState):
    """Simple model that an agent uses for representing the game state"""
    # current player chips, chips on card,
    # value of face up card, player card reward if taken, 
    # opponent card reward if taken, min card distance between players cards, 
    # and min card distance across all players
    
    model_size = 8

    def __init__(self):
        self.reset()
        
    def reset(self):
        self.dict_state = OrderedDict()
        self.state = np.array(list(self.dict_state))
        self.last_card = None
        
    @staticmethod
    def min_distance(player, card):
        """Calculate smallest difference between face up card and owned cards"""
        # the smallest absolute difference between face up card and cards 
        # owned by this player, with the sign retained.
        # This is a way to measure if a card might be worth taking if it is
        # near other cards owned. 
        # if no cards, distance = np.nan.
        # special case, if card connects two other cards (in between) assign
        # distance of 0 because that is REALLY GOOD
        if (len(player.cards) > 0):
            distance = card-player.cards
            min_dist = min(distance, key=abs)
            if -1*min_dist in distance:
                # break ties negative, and check if this is the connection case
                min_dist = abs(min_dist)
                if min_dist == 1:
                    min_dist = 0
                else:
                    min_dist *= -1
        else:
            min_dist = np.inf
        
        return min_dist
    
    @staticmethod
    def card_reward(card, chips, dist):
        reward = chips - card
        if dist == 0:
            # connecting runs, reward is reducing score by game.card+1 (+chips)
            reward += 2*card+1 # add back game.card we subtracted
        elif dist == -1:
            # extra reward +1 because smaller end of a run
            reward += card+1 
        elif dist == 1:
            # add back game.card we subtracted
            reward += card
        return reward
        
    def update_state(self, player, game):
        """This is only called before choosing an action,
        only if that is possible. If the agent is forced to take a card, then 
        the state is not updated because that is irrelevant for making choices"""
        
        if self.last_card == game.card:
            # went around passing
            self.dict_state['player_chips'] -= 1
            self.dict_state['game_chips'] += game.number_of_players
            self.dict_state['p_reward'] += game.number_of_players
            self.dict_state['o_reward'] += game.number_of_players
            if self.inf_p_min_dist:
                self.dict_state['p_dist'] -= game.number_of_players
            if self.inf_o_min_dist:
                self.dict_state['o_dist'] -= game.number_of_players                
        else:
            # Only update distances if we haven't yet for this new card
            self.last_card = game.card
            # print("caller: ", player.position, self.last_card)
            # min_distance of this player
            p_min_dist = SimpleState.min_distance(player, game.card)
            
            # minimum min_distance of all other players        
            o_dists = [SimpleState.min_distance(p, game.card)
                       for p in game.players if p is not player]

            # this breaks ties negative
            o_min_dist = min(np.sort(o_dists), key=abs)
            
            # calculate rewards if cards were taken
            p_card_reward = self.card_reward(game.card, game.chips, p_min_dist)
            o_card_reward = self.card_reward(game.card, game.chips, o_min_dist)

            # if p_min_dist or o_min_dist are inf, no cards yet, so assign
            # this distance is the negative of the card_reward
            # this might incentivize taking cards earlier in the game with
            # many chips. These "distances" will be positive but decreasing.
            # When they hit 0, game force takes so no worrying about wrapping
            # to negative distances
            if np.isinf(p_min_dist):
                self.inf_p_min_dist = True
                p_min_dist = -1*p_card_reward
            else:
                self.inf_p_min_dist = False
            if np.isinf(o_min_dist):
                self.inf_o_min_dist = True
                o_min_dist = -1*(o_card_reward+1) # plus one chip for next player
            else:
                self.inf_o_min_dist = False
                # modify o_card_reward by the number of chips that will be added
                # if the card gets to that player
                o_positions = [p.position for p in game.players if p is not player]
                # find player position of this minimum
                o_min_dist_position = o_positions[o_dists.index(o_min_dist)]
                o_card_reward += (o_min_dist_position - player.position) % game.number_of_players
                
            self.dict_state['deck_size'] = game.deck_size
            self.dict_state['player_chips'] = player.chips
            self.dict_state['game_chips'] = game.chips
            self.dict_state['game_card'] = game.card
            self.dict_state['p_reward'] = p_card_reward
            self.dict_state['o_reward'] = o_card_reward
            self.dict_state['p_dist'] = p_min_dist
            self.dict_state['o_dist'] = o_min_dist
        
        # for model learning
        self.state = np.array(list(self.dict_state.values()))
        # print(player.position, self.state)
        return self.state
        
class FullState(GameState):
     """Full model that an agent uses for representing the game state"""
     def update_state(self, game):
        """Convert game state to player state based on relative positions"""
        # Shift ownership values of cards by players (values>=0) so that 
        # this player is 0 and the next player is 1 and so on (modulus)
        state = np.array(game.game_state)
        state[state>=0] = (state[state>=0] - self.position) % game.number_of_players
        # put player information at the beginning and number of chips on the card
        state = np.insert(state, [0,0], [self.chips, game.chips])
        self.player_state = state
        
                # Initialize game state for history. This is just an array
        # for the all possible cards. -2 for not exposed yet, -1 for currently
        # face up, and player position index if card is owned by that player
    #             self.update_card_owner(-1)
    #             # Record which player took this card
    #             self.update_card_owner(self.player.position)
    # def update_card_owner(self, new_owner):
    #             self.game_state = NoThanks.full_deck_size*[-2]
    #     self.game_state[self.card-NoThanks.lowest_card] = new_owner
     

class NoThanks(object):
    """Controller for playing a game of NoThanks"""
    starting_chips = 11
    lowest_card = 3
    highest_card = 35
    # starting_chips = 3
    # lowest_card = 3
    # highest_card = 11
    
    allowed_players = range(3,6)
    starting_deck = range(lowest_card, highest_card+1)
    full_deck_size = len(starting_deck) # the total number of possible cards
    
    # action, chips for player, chips on card
    cards_removed = 9 # number of cards never revealed
    #cards_removed = 2 # number of cards never revealed
    # select this number of cards out of starting_deck

    
    def __init__(self, players, silent=False):
        self.number_of_players = len(players)
        if self.number_of_players not in NoThanks.allowed_players:
            raise ValueError
        self.silent = silent
        self.players = players
        self.reset()

    def reset(self):
        """Reset game values with the same players"""
        self.done = False
        self.chips = 0
        self.deck_size = NoThanks.full_deck_size-NoThanks.cards_removed 
        self.shuffle_deck()

        for p in self.players:
            p.reset(p.position)
            
        self.player_cycle = itertools.cycle(self.players)
        self.next_player()

    def shuffle_deck(self):
        """Create game deck by shuffling and leaving out 9 cards"""
        self.deck = iter(np.random.choice(NoThanks.starting_deck,
                                          self.deck_size, 
                                          replace = False))
    def draw_card(self):
        self.card = next(self.deck)
        self.deck_size -= 1
        self.chips = 0
        
    def player_turn(self):
        return self.player.action(self)
                
    def game_over(self):
        self.done = True
        scores = self.get_scores()
        # Do any post game over work in players
        for p in self.players:
            p.game_over(self)
        self.winner = np.argmax(scores) #this breaks ties for us??
        
        if not self.silent:
            print("Player " + str(self.winner+1) + " wins")
            print("Scores : " + str(scores))
        
    def next_player(self):
        self.player = next(self.player_cycle)
        return self.player

    def play(self):
        """Play the game until the deck runs out"""
        if self.done:
            raise StopIteration()
        self.draw_card()
        while True:
            # If player took a card, still their turn
            if self.player_turn():
                try:
                    self.draw_card()
                except StopIteration:
                    self.game_over()
                    break
            # A chip was played, next player
            else:
                self.chips += 1
                self.next_player()

    def get_scores(self):
        return [s.score for s in self.players]

    def print_players(self, current=True):
        for p in self.players:
            p.cards.sort()
            str_cards = [str(c) for c in p.cards]
            if current and p is self.player:
                print("    My     Chips: " + str(self.player.chips) + 
                      " | Cards: " + ', '.join(str_cards))
            else:
                print("Player " + str(p.position+1) + "'s Chips: "
                  + str(p.chips) + " | Cards: " + ', '.join(str_cards)) 
                
    def print_game_state(self):
        print("=======================================")
        print("Card: " + str(self.card) + " | Chips: " + str(self.chips) + 
              " | # Cards Left: " + str(self.deck_size))
        print("=======================================")
        self.print_players()


class Player(object):
    """Basic functionality of a NoThanks player"""    

    def __init__(self, position):
        self.reset(position)

    def reset(self, position):
        self.position = position
        self.chips = NoThanks.starting_chips
        self.cards = []
        self.score = self.chips  
        self.reward = 0
        self.last_card = None # the last card that this player has seen
        self.forced_take = False
        
    def pass_turn(self):
        """1 = Pass turn by placing a chip on the card. Returns reward = -1"""
        self.chips -= 1
        self.score -= 1
        return -1 
    
    def take_card(self, game):
        """0 = Add card to player.cards, and add card chips to player chips"""
        old_score = self.score
        self.chips += game.chips
        self.cards.append(game.card)
        
        return self.calculate_score()-old_score
    
    # sublcasses must implement choose_action       
    def choose_action(self, game):
        raise NotImplementedError()
    
    def game_over(self, game):
        pass
        
    def action(self, game):
        """Game asks player to act. Returns 1 if card is taken, 0 if passed"""

        # Must take if no chips or if chips on card equals card value 
        # (why wouldn't you?)
        if self.chips<=0 or game.chips>=game.card:
            self.forced_take = True
            self.reward = self.take_card(game)
            a = 1
        else:
            self.forced_take = False
            a = self.choose_action(game)
            if a:
                self.reward = self.take_card(game)
            else:
                self.reward = self.pass_turn()

        # Update the last card that we have seen
        self.last_card = game.card        
        return a
        
    def calculate_score(self):
        """Calculate current score for this player"""
        # TODO optimize this?
        self.score = self.chips - sum([c for c in self.cards 
                                       if c-1 not in self.cards])
        return self.score
        
class Human(Player):
    """Human player via terminal input"""

    def choose_action(self, game):
        game.print_game_state()       
        return int(input("0 = Pass, 1 = Take Card: "))
    
    def game_over(self, game):
        print("===============FINAL STATE===================")
        game.print_players(current=False)

class Computer(Player):
    """Generic automated player"""

    @staticmethod
    def flip_coin(p):
        """Computer players flip a biased coin to choose action"""
        return np.random.binomial(1, p)

class PasserAgent(Computer):
    """Player that always chooses to pass unless forced"""
    def choose_action(self, game):
        return 0

class RandomAgent(Computer):
    """Random player that chooses action based on fixed probabilities"""
    p_take_card = 0.10
    def choose_action(self, game):
        return self.flip_coin(RandomAgent.p_take_card)

class HeuristicAgent(RandomAgent):
    """Agent that uses some heuristics to make decisions"""

    def __init__(self, position):
        self.state = SimpleState()
        super().__init__(position)
        
    def reset(self, position):
        super().reset(position)
        self.state.reset()
        
    @staticmethod
    def prob_state(x, state_key):
        """Probability of taking the card based on the state variable"""
        if state_key == 'deck_size':
            return sigmoid(x, (NoThanks.full_deck_size-NoThanks.cards_removed)/2, .5)
        elif state_key == 'player_chips':
            return 1-sigmoid(x, 6, .6)
        elif state_key == 'game_chips':
            # return sigmoid(x, 10, 0.6)
            return sigmoid(x, 8, 0.75) # need to pick up cards earlier
        elif state_key == 'game_card':
            return 1-sigmoid(x, 18, 0.25)
        elif state_key == 'p_reward' or state_key == 'o_reward':
            # its ok to take small negative rewards?
            # return sigmoid(x, -3, 1)
            return sigmoid(x, -5, 1) # need to pick up card earlier
        # player/opponent distance
        elif state_key == 'p_dist' or state_key == 'o_dist':
            return 1-2*np.abs(sigmoid(x, 0, 0.5)-0.5)
            # 1-sigmoid(np.abs(x), 2, 2)
        
        
    @staticmethod
    def heuristic_prob(state):
        """Return 0, p, or 1 based on consensus of coin flips for 
        various state variables"""
        
        parray = np.array([HeuristicAgent.prob_state(value, state_key) 
                           for state_key, value in state.items()])
        
        # Flip a biased coin for each probability associated with each
        # state variable
        state_flips = np.random.binomial(1, parray[~np.isnan(parray)])
        # gives 0, 0.5, or 1 based on consensus
        consensus = np.median(state_flips)
        
        # no consensus err on the side of not taking the card by dropping the 
        # probability
        if np.isclose(consensus, 0.5):
            consensus = HeuristicAgent.p_take_card
        return consensus
        
    def choose_action(self, game):        
        # each simple state variable corresponds to a weighted coin
        # with each coin being a function of the state variable
        self.state.update_state(self, game)

        p = self.heuristic_prob(self.state.dict_state)

        return self.flip_coin(p)

class QuickHeuristic(HeuristicAgent):
    
    @staticmethod
    def heuristic_prob(state):
        # positive rewards are very rare,
        # and usually the best move is to pass which makes choosing
        # positive rewards even harder.

        # if our p_dist is not neighboring, consider taking negative
        # total rewards becaues they give chips
        # but if our p_dist is neighboring, try to hold out for more points
        if abs(state['p_dist'])<=1:
            p = sigmoid(state['p_reward'], 5, 0.5)    
        else:
            if state['player_chips'] >= 15:
                p = sigmoid(state['p_reward'], -3, 1.25)
            else:
                p = sigmoid(state['p_reward'], -6, 0.7)

        return p

class Thankless(object):
    """Train a neural net on playing NoThanks"""

    gamma              = 0.95
    exploration_rate   = 1.0
    exploration_min    = 0.01
    exploration_decay  = 0.995
    mem_length         = 4000
    
    def __init__(self, input_dim = SimpleState.model_size):     
        self.input_shape = (1,input_dim)
        self.memory = deque(maxlen=Thankless.mem_length)
        self.set_eps(Thankless.exploration_rate)
        self.set_is_training()
        
        self.model = Sequential()
        self.model.add(Dense(36, activation='relu',
                             input_dim=input_dim))
        self.model.add(Dense(36, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='mse')
    
    def set_is_training(self, is_training = True):
        self.is_training = is_training
        
    def set_eps(self, eps):
        if eps < Thankless.exploration_min:
            self.eps = Thankless.exploration_min
        else:
            self.eps = eps
    
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        
    def replay(self, sample_batch_size=32):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state in sample_batch:
            state = state.reshape(self.input_shape)
            target = reward
            if next_state is not None:
                next_state = next_state.reshape(self.input_shape)
                target = reward + Thankless.gamma * \
                         self.model.predict(next_state)[0][0]
            self.model.fit(state, np.array([target]), epochs=1, verbose=0)
        self.set_eps(self.eps*Thankless.exploration_decay)
    
class LearningAgent(QuickHeuristic):
    
    def __init__(self, position, model):        
        self.state = SimpleState()
        self.model = model

        super().__init__(position)
        
    def reset(self, position):
        super().reset(position)
        if self.model.is_training:
            # Learn from the last game(s)
            self.model.replay()
        self.last_action = None
        self.passes = 0
        self.state.reset()
    
    def game_over(self, game):
        if self.model.is_training and self.last_action is not None and len(self.state.state)>0:
            reward = self.reward_shaping(game)
            self.model.remember(copy.deepcopy(self.state.state), 
                                self.last_action, reward, None)

    def reward_shaping(self, game):
        # all of this bypasses taking a card when forced
        # because we never choose an action
        
        if self.forced_take:
            # if the last action was forced to take a card, overwrite the last
            # action as passing (with 1 chip)
            self.last_action = 0
            # self.state.last_card is the last card we made a decision on
            # self.last_card is actually the card we forced took
            
            if self.state.last_card == self.last_card:
                # if they are the same we just the reward for taking the card
                return self.reward
            else:
                # if they are different, than the state.last_card was taken
                # by another player, so we lose those chips and also take the 
                # reward for taking the card
                return self.reward - self.passes
            self.passes = 0
            
        elif self.last_action:
            # if we take, the reward is just the normal reward
            # reset times passed on this card if we took a card
            self.passes = 0
            return self.reward
        else:
            self.passes += 1 # we previously passed
            # +(N-1) (number of players) if we passed but it comes back around
            #   this should incentivize passing when we don't think others will take
            if self.state.last_card == game.card:
                new_reward = (game.number_of_players-1)
            else:
                # -n chips we passed on a card if it is taken by someone else
                # and since we give those chips to another player, it should be even worse
                # let's just pretend we distributed our chips to all other players
                # 1/(N-1)
                #   this should disincentivize passing too much
                # new_reward = -(1+1/(game.number_of_players-1))*self.passes
                # new_reward = -2*self.passes
                new_reward = -1*self.passes
                self.passes = 0
            return new_reward
        
        
    def choose_action(self, game):
        """This is only called if we did not force take a card"""
        # Thus, self.state is only updated if we had a choice in action
        if self.model.is_training:
            old_state = copy.deepcopy(self.state.state)
            reward = self.reward_shaping(game)
            new_state = copy.deepcopy(self.state.update_state(self, game))
            
            # make sure we took an action and even had a state yet
            # (ex: could be forced to take 3 with 3 chips on it at the start)
            if self.last_action is not None and len(old_state)>0:
                self.model.remember(old_state, self.last_action, reward, new_state)
        else:
            self.state.update_state(self, game)

        # Early in training, use a quick heuristic agent
        if self.model.is_training and np.random.random() < self.model.eps:
            p = self.heuristic_prob(self.state.dict_state)
        else:
            p = self.model.model.predict(self.state.state.reshape(self.model.input_shape))[0][0]
        
        # A uncertainty on our p. As "random" play cools off
        # start cooling off the uncertainty in our edge probabilities
        # This might help avoid getting stuck at 0 or 1
        # Another way to implement exploration
        # never be more than 98% sure of taking a card
        # and 99% for passing a card
        uncertainty = 0.2*np.exp(-.05/self.model.eps)
        min_p = min([uncertainty, Thankless.exploration_min])            
        max_p = 1 - min([uncertainty, 2*Thankless.exploration_min])            
        p_new = np.clip(p, min_p, max_p)
        return self.flip_coin(p_new)
    
    def action(self, game):
        self.last_action = super().action(game)
        return self.last_action

def main():
    pass
    

if __name__ == "__main__":
    main()