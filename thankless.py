import numpy as np
import itertools
import random 
import copy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
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
        self.model.add(Dense(8,activation='relu'))
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
    
    def play_games(self, players, number_of_games):
        score_history = []
        game = NoThanks(players, silent=True)
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
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.state = []
        self.last_card = np.nan
        
    def min_distance(player, card):
        """Calculate smallest difference between face up card and owned cards"""
        # the smallest absolute difference between face up card and cards 
        # owned by this player. Even though having a lower neighboring card 
        # is strictly better than a neighboring hire card, the difference is 1
        # and not worth worrying about
        # This is a way to measure if a card might be worth taking if it is
        # near other cards owned. 
        # if no cards, distance = np.nan.
        # special case, if card connects two other cards (in between) assign
        # distance of 0 because that is REALLY GOOD
        if (len(player.cards) > 0):
            diffs = np.abs(card-player.cards)
            distance = np.min(diffs)
            if distance == 1:
                # check if there are two distance == 1 and assign distance = 0
                if np.count_nonzero(diffs == distance) == 2:
                    distance = 0
        else:
            distance = np.nan
            
        return distance
        
    def update_state(self, player, game):
        if self.last_card == game.card:
            # only thing that could have changed is player.chips and game.chips
            self.state[0] -= 1 
            # card-chips cannot be negative because we hard coded
            # that a player will take a card if card==chips
            self.state[1] -= game.number_of_players
        else:
            # Only update distances if we haven't yet for this new card
            self.last_card = game.card
        
            # min_distance of this player
            p_dist = SimpleState.min_distance(player, game.card)
            
            # minimum min_distance of all other players        
            o_dist = np.min([SimpleState.min_distance(p, game.card)
                             for p in game.players if p is not player])

            self.state=np.array([player.chips, game.card-game.chips, 
                                 p_dist, o_dist])
        
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
    deck_size = full_deck_size-cards_removed 
    
    def __init__(self, players, silent=False):
        self.number_of_players = len(players)
        if self.number_of_players not in NoThanks.allowed_players:
            raise ValueError
        self.silent = silent
        self.players = players
        self.reset()

    def reset(self):
        """Reset game values with the same players"""
        self.chips = 0
        self.shuffle_deck()

        for p in self.players:
            p.reset(p.position)
            
        self.player_cycle = itertools.cycle(self.players)
        self.next_player()

    def shuffle_deck(self):
        """Create game deck by shuffling and leaving out 9 cards"""
        self.deck = iter(np.random.choice(NoThanks.starting_deck,
                                          NoThanks.deck_size, 
                                          replace = False))
    def draw_card(self):
        self.card = next(self.deck)
        self.chips = 0
        
    def player_turn(self):
        return self.player.action(self)
                
    def game_over(self):
        scores = self.get_scores()
        self.winner = np.argmax(scores) #this breaks ties for us??
        
        if not self.silent:
            print("Player " + str(self.winner+1) + " wins")
            print("Mean Score " + str(np.mean(scores)))
        
    def next_player(self):
        self.player = next(self.player_cycle)

    def play(self):
        """Play the game until the deck runs out"""
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
        self.score = 0       
        
    def pass_turn(self):
        """Pass turn by placing a chip on the card, returns 0"""
        self.chips -= 1;
        return 0
    
    def take_card(self, game):
        """Add card to player.cards, and add card chips to player chips"""
        self.cards.append(game.card)
        self.cards.sort()
        self.chips += game.chips
        return 1
    
    # sublcasses should implement choose_action        
    def choose_action(self, game):
        pass
        
    def action(self, game):
        """Game asks player to act. Returns 1 if card is taken, 0 if passed"""
        
        # Must take if no chips or if chips on card equals card value 
        # (why wouldn't you?)
        if self.chips<=0 or game.chips>=game.card:
            action = self.take_card(game)
        else:
            action = self.choose_action(game)
            if action:
                self.take_card(game)
            else:
                self.pass_turn()
        
        # Calculate score after each action, #TODO kinda slow
        self.calculate_score()
        return action
        
    def calculate_score(self):
        """Calculate current score for this player"""
        self.score = self.chips - sum([c for c in self.cards 
                                       if c-1 not in self.cards])
        
class Human(Player):
    """Human player via terminal input"""
    def choose_action(self):
        # TODO terminal input
        pass

class Computer(Player):
    """Generic automated player"""
    pass
        

class RandomAgent(Computer):
    """Random player that chooses action based on fixed probabilities"""
    
    def choose_action(self, game):
        return np.random.binomial(1, 0.10)

class HeuristicAgent(Computer):
    """Agent that uses some heuristics to make decisions"""

    def __init__(self, position):
        self.state = SimpleState()
        super().__init__(position)
        
    def reset(self, position):
        super().reset(position)
        self.state.reset()
    
    def prob_state(x, state_index):
        """Probability of taking the card based on the state variable"""
        # [player.chips, game.card-game.chips, p_dist, o_dist])
        
        # player chips
        if state_index == 0:
            return 1-sigmoid(x, 4, .8)
        # card-chips
        elif state_index == 1:
            return 1-sigmoid(x, 17, 0.35)
        # player/opponent distance
        elif state_index > 1:
            return 1-sigmoid(x, 2, 2)
        
    def p_take_card(state):
        # TODO maybe odist should only be relevant if we are worried about an
        # opponent taking a good card for us from us (compare distances)        
        parray = np.array([HeuristicAgent.prob_state(value, i) 
                           for i, value in enumerate(state)])
            
        return np.mean(parray[~np.isnan(parray)])
        
        
    def choose_action(self, game):
        # Calculate the probability of taking a card by averaging over
        # the probabilities of taking a card for each simple state variable
        
        # each simple state variable corresponds to a weighted coin
        # with each coin being a function of the state variable
        self.state.update_state(self, game)

        p_take_card = HeuristicAgent.p_take_card(self.state.state)

        action = np.random.binomial(1, p_take_card)
        return action 
    
class LearningAgent(Computer):
    # state_size = NoThanks.full_deck_size+3
            # First update the state. We will add the action chosen after
        # self.update_state(game)
    #         self.history = []
    # def update_history(self):
    #     self.history.append(self.player_state)
        # Add the action chosen to the previous state and update the history
        # self.player_state = np.insert(self.player_state, 0, action)
        # self.update_history()
    pass

def main():
    pass
    # game = NoThanks(4, 0)
    # game.play()
    # scores = game.get_scores()
    # print(scores)
    

if __name__ == "__main__":
    main()