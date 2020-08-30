# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:28:27 2020

@author: MN
"""
import pygame
from random import randint
from DQNAgent import DQNAgent
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

# Set options to activate or deactivate the game view, and its speed
display_option = True
speed = 0
pygame.font.init()

REAL_STATES = 6
NB_STATES = REAL_STATES + 1
UNIT = NB_STATES     # unit for flat array access
UNIT_SQ = UNIT**2    # unit to the square

BOARD_SIZE = 50

NB_RULES = NB_STATES**3

DISPLAY = "#F.*^v/"
BORDER = 0    # BORDER is a false state as far as rules are concerned
FIRE = 1
SLEEP = 2
GEN = 4
ACT_1 = 5
ACT_2 = 5
ACT_3 = 6
"""class Game:

    def __init__(self, game_width, game_height):
        pygame.display.set_caption('SnakeGen')
        self.game_width = game_width
        self.game_height = game_height
        self.gameDisplay = pygame.display.set_mode((game_width, game_height+60))
        self.bg = pygame.image.load("img/background.png")
        self.crash = False
        self.player = Player(self)
        self.food = Food()
        self.score = 0"""
class Automaton:
    """ inplements automaton rules """
    def __init__(self, aut = None):
        if aut == None:
            self.rules = np.zeros(NB_RULES, dtype=np.int16) # BORDER everywhere
            self.set_rule(SLEEP,SLEEP,SLEEP,SLEEP)
            self.set_rule(BORDER,SLEEP,SLEEP,SLEEP)
            self.set_rule(SLEEP,SLEEP,BORDER,SLEEP)
        else:
            self.rules = np.copy(aut.rules)
    def copy(self, aut):
            self.rules = np.copy(aut.rules)
    def get_rule(self,g,c,d):
        return self.rules[g * UNIT_SQ + c * UNIT + d]
    def set_rule(self,g,c,d,r):
        self.rules[g * UNIT_SQ + c * UNIT + d]=r
    def set_rule_pos(self,pos,r):
        self.rules[pos]=r
    def display(self, max_level, disp = sys.stdout):
        global DISPLAY
        for l in range(2, max_level+1):
            line = np.zeros(l+2)
            nline = np.zeros(l+2)
            line[1] = GEN
            line[2:l+1] = SLEEP
            self.print_line(disp,line)
            for i in range(1,2*l-1):
                for j in range(1,l+1):
                    nline[j]=self.get_rule(int(line[j-1]),int(line[j]),int(line[j+1]))
                self.print_line(disp, nline)
                if (i < 2*l-2 and FIRE in nline or
                    i == 2*l-2 and any(nline[1:l+1] != FIRE)):
                    disp.write("ERROR\n")
                    return
                line = np.copy(nline)
            print() # separate levels
    def print_line(self, disp, line):
        global DISPLAY
        for i in line:
            disp.write(DISPLAY[int(i)])
        disp.write("\n")

"""class Player(object):

    def __init__(self, game):
        x = 0.45 * game.game_width
        y = 0.5 * game.game_height
        self.x = x - x % 20
        self.y = y - y % 20
        self.position = []
        self.position.append([self.x, self.y])
        self.food = 1
        self.eaten = False
        self.image = pygame.image.load('img/snakeBody.png')
        self.x_change = 20
        self.y_change = 0

    def update_position(self, x, y):
        if self.position[-1][0] != x or self.position[-1][1] != y:
            if self.food > 1:
                for i in range(0, self.food - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = x
            self.position[-1][1] = y

    def do_move(self, move, x, y, game, food,agent):
        move_array = [self.x_change, self.y_change]

        if self.eaten:

            self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1
        if np.array_equal(move ,[1, 0, 0]):
            move_array = self.x_change, self.y_change
        elif np.array_equal(move,[0, 1, 0]) and self.y_change == 0:  # right - going horizontal
            move_array = [0, self.x_change]
        elif np.array_equal(move,[0, 1, 0]) and self.x_change == 0:  # right - going vertical
            move_array = [-self.y_change, 0]
        elif np.array_equal(move, [0, 0, 1]) and self.y_change == 0:  # left - going horizontal
            move_array = [0, -self.x_change]
        elif np.array_equal(move,[0, 0, 1]) and self.x_change == 0:  # left - going vertical
            move_array = [self.y_change, 0]
        self.x_change, self.y_change = move_array
        self.x = x + self.x_change
        self.y = y + self.y_change

        if self.x < 20 or self.x > game.game_width-40 or self.y < 20 or self.y > game.game_height-40 or [self.x, self.y] in self.position:
            game.crash = True
        eat(self, food, game)

        self.update_position(self.x, self.y)

    def display_player(self, x, y, food, game):
        self.position[-1][0] = x
        self.position[-1][1] = y

        if game.crash == False:
            for i in range(food):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                game.gameDisplay.blit(self.image, (x_temp, y_temp))

            update_screen()
        else:
            pygame.time.wait(300)"""
class Board:
    """ the time-space diagram:
        * nb lines = 2 * (max-1) + 1  # transforms + initial line
    """
    def __init__(self, board = None):
        if board == None:
            """ set all cells to BORDER, prepare for level 2 """
            self.board = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.int8)
            self.pn = 2     # level
            self.pi = 1
            self.pj = 1
            self.board[0][1] = GEN
            self.board[0][2] = SLEEP
            self.nrw = self.next_cell_rule()
            self.finished = False
            self.success = False
        else:
            self.board = np.copy(board.board)
            self.pn = board.pn
            self.pi = board.pi
            self.pj = board.pj
            self.finished = board.finished
            self.success = board.success
            self.nrw = board.nrw
    def copy(self, board):
            self.board = np.copy(board.board)
            self.pn = board.pn
            self.pi = board.pi
            self.pj = board.pj
            self.finished = board.finished
            self.success = board.success
            self.nrw = board.nrw
    def first(self, rules, max_level):
        self.pi = self.pn - 2
        self.pj = self.pn - 1
        self.nrw = self.next_cell_rule()
        self.play(rules, max_level)
    def clean_cells(self):
        self.board[self.pn-3][self.pn] = SLEEP
        self.board[self.pn-2][self.pn] = SLEEP
    def get(self):
        return self.board,self.pn,self.pi,self.pj
    def set(self,board,pn,pi,pj):
        self.board = board
        self.pn = pn
        self.pi = pi
        self.pj = pj
    def next_cell_rule(self):
        return (self.board[self.pi-1][self.pj-1] * UNIT_SQ +
                self.board[self.pi-1][self.pj] * UNIT +
                self.board[self.pi-1][self.pj + 1])
    def next_cell(self):
        if self.pi == self.pn-2: # special case just after next level
            self.pi+=1
            self.pj-=1
        else:
            self.pj+=1
            if self.pj > self.pn:
                self.pi+=1
                if self.pi < 2*self.pn-3:
                    self.pj = 2*self.pn-3-self.pi
                else:
                    self.pj = 1
        self.nrw = self.next_cell_rule()
    def play(self, rules, max_level):
        while True: # usually one can fill several cells
            if self.pi > 2*self.pn - 2: #  full last line
                if self.pn == max_level: # job finished
                    self.finished = True
                    self.success = True
                    return
                self.pn += 1
                self.clean_cells()
                self.first(rules, max_level)
            if self.finished: # either a success or not
                return
            #self.nrw = self.next_cell_rule() # try to fill next cell
            if rules[self.nrw] == BORDER: # rule undefined yet
                if self.pi == 2*self.pn - 2: # last line
                    rules[self.nrw] = FIRE # no other choice
                    continue
                else: # must be decided by caller
                    return
            else: # rule already defined: play it
                if (self.pi == 2 * self.pn - 2 and rules[self.nrw] != FIRE or
                    self. pi < 2 * self.pn - 2 and rules[self.nrw] == FIRE): # missfire
                    self.success = False
                    self.finished = True
                    self.nrw = 0
                    return
                else: # playable
                    self.board[self.pi][self.pj] = rules[self.nrw]
                    self.next_cell()
    def show_nrw(self, the_end=''):
        print(self.pn, self.pi, self.pj,
              self.board[self.pi-1][self.pj-1],
              self.board[self.pi-1][self.pj],
              self.board[self.pi-1][self.pj + 1],
              decode(self.nrw), end=the_end)

"""class Food(object):

    def __init__(self):
        self.x_food = 240
        self.y_food = 200
        self.image = pygame.image.load('img/food2.png')

    def food_coord(self, game, player):
        x_rand = randint(20, game.game_width - 40)
        self.x_food = x_rand - x_rand % 20
        y_rand = randint(20, game.game_height - 40)
        self.y_food = y_rand - y_rand % 20
        if [self.x_food, self.y_food] not in player.position:
            return self.x_food, self.y_food
        else:
            self.food_coord(game,player)

    def display_food(self, x, y, game):
        game.gameDisplay.blit(self.image, (x, y))
        update_screen()


def eat(player, food, game):
    if player.x == food.x_food and player.y == food.y_food:
        food.food_coord(game, player)
        player.eaten = True
        game.score = game.score + 1"""


"""def get_record(score, record):
        if score >= record:
            return score
        else:
            return record"""


"""def display_ui(game, score, record):
    myfont = pygame.font.SysFont('Segoe UI', 20)
    myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
    text_score = myfont.render('SCORE: ', True, (0, 0, 0))
    text_score_number = myfont.render(str(score), True, (0, 0, 0))
    text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
    text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
    game.gameDisplay.blit(text_score, (45, 440))
    game.gameDisplay.blit(text_score_number, (120, 440))
    game.gameDisplay.blit(text_highest, (190, 440))
    game.gameDisplay.blit(text_highest_number, (350, 440))
    game.gameDisplay.blit(game.bg, (10, 10))"""


"""def display(player, food, game, record):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game, game.score, record)
    player.display_player(player.position[-1][0], player.position[-1][1], player.food, game)
    food.display_food(food.x_food, food.y_food, game)


def update_screen():
    pygame.display.update()


def initialize_game(player, game, food, agent):
    state_init1 = agent.get_state(game, player, food)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = [1, 0, 0]
    player.do_move(action, player.x, player.y, game, food, agent)
    state_init2 = agent.get_state(game, player, food)
    reward1 = agent.set_reward(player, game.crash)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_new(agent.memory)


def plot_seaborn(array_counter, array_score):
    sns.set(color_codes=True)
    ax = sns.regplot(np.array([array_counter])[0], np.array([array_score])[0], color="b", x_jitter=.1, line_kws={'color':'green'})
    ax.set(xlabel='games', ylabel='score')
    plt.show()"""
#@jit
def decode(nrw):
    g = nrw // UNIT_SQ
    c = (nrw - g*UNIT_SQ) // UNIT
    d = nrw - g*UNIT_SQ - c*UNIT
    return (DISPLAY[g],DISPLAY[c],DISPLAY[d])

#@jit
def get_rule(rules, g,c,d):
    return rules[g * UNIT_SQ + c * UNIT + d]

#@jit
def exhaustive_search(aut, board, max_level):
    if board.finished:
        return
    for i in range(SLEEP,NB_STATES):
        """ work on a temporary """
        aut2 = Automaton(aut)
        board2 = Board(board)
        aut2.set_rule_pos(board.nrw, i)
        board2.play(aut2.rules, max_level)
        exhaustive_search(aut2, board2, max_level)
        if board2.success:
            aut.copy(aut2)
            board.copy(board2)
            return

def run():
    max_level = 5
    #pygame.init()
    agent = DQNAgent()
    counter_games = 0
    score_plot = []
    counter_plot =[]
    record = 0
    while counter_games < 150:
        # Initialize classes
        #game = Game(440, 440)
        aut = Automaton()
        board = Board()
        #player1 = game.player
        #food1 = game.food

        # Perform first move
        #initialize_game(player1, game, food1, agent)
        if display_option:
            #display(player1, food1, game, record)
            aut.display(max_level)
        while not board.finished:
            #agent.epsilon is set to give randomness to actions
            agent.epsilon = 80 - counter_games
            
            #get old state
            state_old = agent.get_state(board)
            
            #perform random actions based on agent.epsilon, or choose the action
            #if randint(0, 200) < agent.epsilon:
            #final_move = to_categorical(randint(SLEEP,NB_STATES), num_classes=8)
            final_move = randint(SLEEP,NB_STATES)
            aut2 = Automaton(aut)
            board2 = Board(board)
            aut2.set_rule_pos(board.nrw, final_move)
            board2.play(aut2.rules, max_level)
            """else:
                # predict action based on the old state
                prediction = agent.model.predict(state_old.reshape((1,11)))
                final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)"""
                
            #perform new move and get new state
            
            """exhaustive_search(aut,board,max_level)"""
            #player1.do_move(final_move, player1.x, player1.y, game, food1, agent)
            state_new = agent.get_state(board2)
            
            #set treward for the new state
            reward = agent.set_reward(board2, board2.finished)
            
            #train short memory base on the new action and state
            #agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
            
            # store the new data into a long term memory
            agent.remember(state_old, final_move, reward, state_new, board2.finished)
            #record = get_record(game.score, record)
            if display_option:
                aut2.display(max_level)
                #display(player1, food1, game, record)
                #pygame.time.wait(speed)
        
        agent.replay_new(agent.memory)
        counter_games += 1
        print('Game', counter_games)#, '      Score:', game.score"""
        #score_plot.append(game.score)
        counter_plot.append(counter_games)
    #agent.model.save_weights('weights.hdf5')
    #plot_seaborn(counter_plot, score_plot)


run()


