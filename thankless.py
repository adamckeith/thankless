import numpy as np
import itertools
from collections import deque 
import random 
import copy
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Activation
from tensorflow.keras.optimizers import Adam
import keras

def sigmoid(x, x0, k):
  return 1 / (1 + np.exp(-k*(x-x0)))

class Thankless(object):
    """Train a neural net on playing NoThanks"""
    def __init__(self):
        pass
    
    def train(self):
        # ## Creating the model
        self.model = Sequential()
        self.model.add(Dense(16,input_dim=Player.state_size,activation='relu'))
        # self.model.add(Dense(8,activation='relu'))
        self.model.add(Dense(1, activation='softmax'))
        self.model.compile(optimizer='adam',loss='binary_crossentropy')

        self.c = NoThanksController()
        self.c.play_games(10000, 4)
        self.x = np.asarray(self.c.all_history)
        self.y = keras.utils.to_categorical(self.c.winner_history)
        self.model.fit(x=self.x, y=self.y, epochs=100, batch_size=50)

class NoThanksController(object):
    """Basic controller for playing multiple games of NoThanks"""
    def __init__(self):
        self.all_history = []
        self.winner_history = []
    #         number_of_computers = number_of_players-number_of_humans
        
    # number_of_players, number_of_humans
    #         # TODO revisit to shuffle players for human/computer mix
    #     self.players = [Computer(i) for i in range(number_of_computers)] + \
    #                    [Human(i+number_of_computers) for i in range(number_of_humans)]
    
    def play_games(self, players, number_of_games, silent=True):
        score_history = []
        game = NoThanks(players, silent=silent)
        for i in range(number_of_games):
            game.reset()
            game.play()
            score_history.append(game.get_scores())
        return score_history
        
    def gather_history_one_game(self, game):
        # just concatenate all histories for each player
        for p in game.players:
            if game.winner == p.position:
                win_or_lose = len(p.history)*[1]
            else:
                win_or_lose = len(p.history)*[0]
                
            if (p.history):
                #Its possible for players to have no history
                self.all_history.extend(p.history)
                self.winner_history.extend(win_or_lose)

class GameState(object):
    """Generic model that an agent uses for representing the game state"""
    pass

class SimpleState(GameState):
    """Simple model that an agent uses for representing the game state"""
    # number of players, current player chips, chips on card,
    # value of face up card, min card distance between players cards, and 
    # min card distance across all players
    
    shape = (1,6)
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.state = []
        self.shape = SimpleState.shape
        self.last_card = np.nan
        
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

            # print(player.position, player.cards, card, min_dist)
            # abs_dist = np.abs(distance)
            # mind_dist_ind = np.argmin(abs_dist)
            
            # min_dist = distance[mind_dist_ind]
            # min_abs_dist = abs_dist[mind_dist_ind]
            
            # # check if there are two distance == 1 and assign distance = 0
            # if min_abs_dist == 1:
            #     if np.count_nonzero(abs_dist == min_dist) == 2:
            #         min_dist = 0
        else:
            min_dist = card
        
        return min_dist
        
    def update_state(self, player, game):
        if self.last_card == game.card:
            # only thing that could have changed is player.chips and game.chips
            self.state[0] -= 1 
            self.state[1] += game.number_of_players
            self.state[2] += game.number_of_players
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
            o_min_dist = min(o_dists, key=abs)

            card_reward = game.chips - game.card
            if p_min_dist == 0:
                # connecting runs, reward is reducing score by game.card+1 (+chips)
                card_reward += 2*game.card+1 # add back game.card we subtracted
            elif p_min_dist == -1:
                # extra reward +1 because smaller end of a run
                card_reward += game.card+1 
            elif p_min_dist == 1:
                # add back game.card we subtracted
                card_reward += game.card 
                
            self.state=np.array([player.chips, game.chips, card_reward,
                                 game.card, p_min_dist, o_min_dist])
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
        # self.cards.sort()
        
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
            self.reward = self.take_card(game)
            a = 1
        else:
            a = self.choose_action(game)
            if a:
                self.reward = self.take_card(game)
            else:
                self.reward = self.pass_turn()
        # self.reward += 1 # try making everything positive
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
        print("=====================================")
        print("Card: " + str(game.card) + " | Chips: " + str(game.chips) + 
              " | # Cards Left: " + str(game.deck_size))
        print("=====================================")
        self.cards.sort()
        print("My Cards: " + str(self.cards) + " My Chips: " + str(self.chips))
        
        # wrap around all players back to ourself to print out their cards
        while game.next_player() is not self:
            game.player.cards.sort()
            print("Player " + str(game.player.position) + "'s cards: " + str(game.player.cards))    
        
        return int(input("0 = Pass, 1 = Take Card: "))
    
    def game_over(self, game):
        print("===============FINAL STATE===================")
        self.cards.sort()
        print("My Cards: " + str(self.cards) + " My Chips: " + str(self.chips))
        
        # wrap around all players back to ourself to print out their cards
        for p in game.players:
            if p is self:
                continue
            game.player.cards.sort()
            print("Player " + str(game.player.position) + "'s cards: " + str(game.player.cards))    

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
    def prob_state(x, state_index):
        """Probability of taking the card based on the state variable"""
        # [player.chips, game.chips, game.card, take_reward, p_dist, o_dist])
        
        if np.isnan(x):
            return x
        
        # player chips
        if state_index == 0:
            return 1-sigmoid(x, 6, .6)
        # card chips
        elif state_index == 1:
            # return sigmoid(x, 10, 0.6)
            return sigmoid(x, 8, 0.75) # need to pick up cards earlier
        # reward for taking card
        elif state_index == 2:
            # its ok to take small negative rewards?
            # return sigmoid(x, -3, 1)
            return sigmoid(x, -5, 1) # need to pick up card earlier
        # card value
        elif state_index == 3:
            return 1-sigmoid(x, 18, 0.25)
        # player/opponent distance
        elif state_index > 3:
            return 1-2*np.abs(sigmoid(x, 0, 0.5)-0.5)
            # 1-sigmoid(np.abs(x), 2, 2)
        
        
    @staticmethod
    def heuristic_prob(state):
        """Return 0, p, or 1 based on consensus of coin flips for 
        various state variables"""
        
        parray = np.array([HeuristicAgent.prob_state(value, i) 
                           for i, value in enumerate(state)])
        
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

        p = self.heuristic_prob(self.state.state)

        return self.flip_coin(p)

class HeuristicAgent2(HeuristicAgent):
    
    @staticmethod
    def heuristic_prob(state):
        """Return 0, p, or 1 based on consensus of coin flips for 
        various state variables"""
        
        parray = np.array([HeuristicAgent.prob_state(value, i) 
                           for i, value in enumerate(state)])

        # split into chip state consensus and card state consensus
        chip_ps = parray[0:3]
        chip_flips = np.random.binomial(1, chip_ps)
        chip_consensus = np.median(chip_flips)
        
        # this may contain nans
        card_ps = parray[3:]
        card_ps = card_ps[~np.isnan(card_ps)]
        card_flips = np.random.binomial(1, card_ps)
        card_consensus = np.median(card_flips)
        
        # This could be 0,0.25,0.5,0.75,1 which is ok?
        consensus = np.median([chip_consensus, card_consensus])

        return consensus
    
class LearningAgent(Computer):
    
    gamma              = 0.95
    exploration_rate   = 1.0
    exploration_min    = 0.01
    exploration_decay  = 0.995
    
    def __init__(self, position):        
        self.state = SimpleState()
        self.memory = deque(maxlen=1000)
        self.set_eps(LearningAgent.exploration_rate)
        self.set_is_training()
        
        super().__init__(position)
                
        self.model = Sequential()
        self.model.add(Dense(24, activation='relu',
                             input_dim=max(self.state.shape)))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='mse')
        
    def reset(self, position):
        super().reset(position)
        if self.is_training:
            # Learn from the last game(s)
            self.replay()
        self.last_action = None
        self.state.reset()
    
    def set_is_training(self, is_training = True):
        self.is_training = is_training
        
    def set_eps(self, eps):
        if eps < LearningAgent.exploration_min:
            self.eps = LearningAgent.exploration_min
        else:
            self.eps = eps
            
    def game_over(self, game):
        if self.is_training and self.last_action is not None:
            self.remember(copy.deepcopy(self.state.state), 
                          self.last_action, self.reward, None)

    def choose_action(self, game):
        
        if self.is_training:
            old_state = copy.deepcopy(self.state.state)
            new_state = copy.deepcopy(self.state.update_state(self, game))
            
            if self.last_action is not None:
                self.remember(old_state, self.last_action, self.reward, new_state)
        else:
            self.state.update_state(self, game)

        # Early in training, use a quick heuristic agent
        if self.is_training and np.random.random() < self.eps:
            if self.state.state[2] > -2:
                # positive rewards are very rare,
                # and usually the best move is to pass which makes choosing
                # positive rewards even harder, so help select them when learning
                p = 0.5
            else:
                # usually a good move to pass
                p = 0.2
        else:
            p = self.model.predict(self.state.state.reshape(self.state.shape))[0][0]
        
        
        # A uncertainty on our p. As "random" play cools off
        # start cooling off the uncertainty in our edge probabilities
        # This might help avoid getting stuck at 0 or 1
        # Another way to implement exploration
        if self.eps < 0.5:
            # never be more than 99% sure of a move
            min_p = min([0.1*np.exp(-.05/self.eps), 0.01])
        else:
            min_p = 0.1
            
        max_p = 1 - min_p
        p_new = np.clip(p, min_p, max_p)
        return self.flip_coin(p_new)
    
    def action(self, game):
        self.last_action = super().action(game)
        return self.last_action
    
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        
    def replay(self, sample_batch_size=32):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state in sample_batch:
            state = state.reshape(self.state.shape)
            target = reward
            if next_state is not None:
                next_state = next_state.reshape(self.state.shape)
                target = reward + LearningAgent.gamma * \
                         self.model.predict(next_state)[0][0]
            self.model.fit(state, np.array([target]), epochs=1, verbose=0)
        self.set_eps(self.eps*LearningAgent.exploration_decay)


    # def train(self):
    #     states = np.asarray(self.states)
    #     state_len = len(states)
    #     target_vectors = np.zeros((state_len, 2))
    #     for i in range(state_len):
    #         target_vectors[i][self.actions[i]] = self.rewards[i]
        
    #     self.model.fit(x=states, y=target_vectors, epochs = 5)

def main():
    pass
    

if __name__ == "__main__":
    main()