import numpy as np
import itertools
import random 
import copy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
import keras

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
    
    def __init__(self):
        self.all_history = []
        self.winner_history = []
    
    def play_games(self, number_of_games, number_of_players):
        # Always only computer players when automating games
        for i in range(number_of_games):
            game = NoThanks(number_of_players, 0, silent=True)
            game.play()
            self.gather_history_one_game(game)
        
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

class StateModel(object):
    """Generic model that an agent uses for representing the game state"""

    pass

class SimpleState(StateModel):
     """Simple model that an agent uses for representing the game state"""
    # number of players, current player chips, chips on card,
    # value of face up card, min card distance between players cards, and 
    # min card distance across all players
    def __init__(self, player, game):
        self.update_state(self, player, game)
        
    def update_state(self, player, game):
        
        # distances of all other players
        diffs = [p.distance for p in game.players if p.position is not player.position]
        
        # only care about the minimum one? but keep sign?
        mindiff = diffs[np.argmin(np.abs(diffs))]
        
        self.state=np.array([game.number_of_players, player.chips, game.chips,
                             game.card, player.distance, mindiff])
        
class FullState(StateModel):
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
     

class NoThanks(object):
    """Controller for playing a game of NoThanks"""
    # starting_chips = 11
    # lowest_card = 3
    # highest_card = 35
    starting_chips = 3
    lowest_card = 3
    highest_card = 11
    
    allowed_players = range(3,6)
    starting_deck = range(lowest_card, highest_card+1)
    full_deck_size = len(starting_deck) # the total number of possible cards
    
    # action, chips for player, chips on card
    # cards_removed = 9 # number of cards never revealed
    cards_removed = 2 # number of cards never revealed
    # select this number of cards out of starting_deck
    deck_size = full_deck_size-cards_removed 
    
    def __init__(self, number_of_players, number_of_humans, silent=False):
        if number_of_players not in NoThanks.allowed_players:
            raise ValueError
        self.silent = silent
        # Initialize game state for history. This is just an array
        # for the all possible cards. -2 for not exposed yet, -1 for currently
        # face up, and player position index if card is owned by that player
        self.chips = 0
        self.game_state = NoThanks.full_deck_size*[-2]
        self.shuffle_deck()
        self.number_of_players = number_of_players
        number_of_computers = number_of_players-number_of_humans
        
        # TODO revisit to shuffle players for human/computer mix
        self.players = [Computer(self, i) for i in range(number_of_computers)] + \
                       [Human(self, i+number_of_computers) for i in range(number_of_humans)]
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
        self.update_card_owner(-1)
        for p in self.players:
            p.calculate_distance(self.card)

    def update_card_owner(self, new_owner):
        self.game_state[self.card-NoThanks.lowest_card] = new_owner
        
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
        self.draw_card()
        while True:
            # If player took a card, still their turn
            if self.player_turn():
                # Record which player took this card
                self.update_card_owner(self.player.position)
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
    """Contains player state and actions"""    
    state_size = NoThanks.full_deck_size+3

    def __init__(self, game, position):
        self.position = position
        self.chips = NoThanks.starting_chips
        self.distance = np.nan 
        self.cards = []
        self.history = []
        self.score = 0
        
    def calculate_distance(self, card):
         """Calculate smallest difference between face up card and owned cards"""
        # the smallest absolute difference between face up card and cards 
        # owned by this player. Keep the sign of the difference.
        # This is a way to measure if a card might be worth taking if it is
        # near other cards owned,
        # if no cards, distance = np.nan.
        if (self.cards.size > 0):
            # if no cards owned yet, distance will be np.nan
            diffs = card-cards
            mindiff_index = np.argmin(np.abs(diffs)
            self.distance = diffs[mindiff_index]

    
    def update_history(self):
        self.history.append(self.player_state)
        
    def pass_turn(self):
        self.chips -= 1;
        return False
    
    def take_card(self, game):
        """Returns True (signalling that a card was taken)"""
        self.cards.append(game.card)
        self.cards.sort()
        self.chips += game.chips
        return True
        
    def action(self, game):
        """Game asks player to act. Returns True if card is taken"""
        # First update the state. We will add the action chosen after
        self.update_state(game)
        
        # consider always taking card if chips on card == card value
        # this is VERY rational
        if self.chips<=0 or game.chips>=game.card:
            action = self.take_card(game)
        else:
            action = self.choose_action(game)
        
        # Add the action chosen to the previous state and update the history
        self.player_state = np.insert( self.player_state, 0, 1*action)
        self.update_history()
        
        # Calculate score after each action, #TODO kinda slow
        self.calculate_score()
        return action
        
    def calculate_score(self):
        """Calculate current score for this player"""
        self.score = self.chips - sum([c for c in self.cards 
                                       if c-1 not in self.cards])
        
class Human(Player):
    def choose_action(self):
        # TODO terminal input
        pass
    
class Computer(Player):
    
    def choose_action(self, game):
        action = random.choices([True, False], weights=[.25,.75], k=1)[0]
        # action = random.choice([True, False])
        if action:
            return self.take_card(game)
        else:
            return self.pass_turn()

def main():
    pass
    # game = NoThanks(4, 0)
    # game.play()
    # scores = game.get_scores()
    # print(scores)
    

if __name__ == "__main__":
    main()