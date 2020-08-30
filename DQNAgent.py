# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 05:11:46 2020

@author: MN
"""
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd

FIRE = 1 # FIRE qui est mis de manière déterministe quand on arrive sur la dernière ligne diagramme espace/temps
max_level = 5
REAL_STATES = 6
NB_STATES = REAL_STATES + 1
UNIT = NB_STATES     # unit for flat array access
UNIT_SQ = UNIT**2    # unit to the square

BOARD_SIZE = 50

NB_RULES = NB_STATES**3

DISPLAY = "#F.*^v/"
BORDER = 0    # BORDER is a false state as far as rules are concerned
SLEEP = 2
GEN = 3
ACT_1 = 4
ACT_2 = 5
ACT_3 = 6
class DQNAgent(object):
    
    def __init__(self):
            self.reward = 0
            self.gamma = 0.9
            self.dataframe = pd.DataFrame()
            self.short_memory = np.array([])
            self.agent_target = 1
            self.agent_predict = 0
            self.learning_rate = 0.0005
            self.model = self.network()
            #self.model = self.network("weights.hdf5")
            self.epsilon = 0
            self.actual = []
            self.memory = []

    def get_state(self, board):
        
        old = [[0,1,0,0,1,0,0,1,0,0,1,0],[0,1,0,0,1,0,0,0,0,0,1,0],[0,0,0,0,1,0,0,1,0,0,1,0]] # liste des régle prédifinie
        
        liste = board.board[board.pi-1][board.pj-1 : board.pj+2]# les trois celluls précédents
        listb = []
        
        for i in liste: #convertire les cellule en binaire
            nw = self.binary(i)
            listb.append(nw)
            
        res = board.board[board.pi][board.pj] #la cellule ou on'est 
        nrg = self.binary(res)# transformer la régle en binaire
        listb.extend(nrg)
        
        

        state = [
            # pour la règle [BORDER,GEN,SLEEP]
            listb[0:3] == [0,0,0] , listb[3:6] == [1,1,0] , listb[6:9] == [0,1,0] , 
            # pour la règle [GEN,SLEEP,BORDER]
            listb[0:3] == [1,1,0] , listb[3:6] == [0,1,0] , listb[6:9] == [0,0,0] , 
            # pour la règle [GEN,SLEEP,SLEEP]
            listb[0:3] == [1,1,0] , listb[3:6] == [0,1,0] , listb[6:9] == [0,1,0] , 
            # pour la règle [ACT_1,SLEEP,BORDER]
            listb[0:3] == [0,0,1] , listb[3:6] == [0,1,0] , listb[6:9] == [0,0,0] , 
            # pour la règle [SLEEP,SLEEP,ACT_1]
            listb[0:3] == [0,1,0] , listb[3:6] == [0,1,0] , listb[6:9] == [0,0,1] ,
            # pour la règle [SLEEP,ACT_1,BORDER]
            listb[0:3] == [0,1,0] , listb[3:6] == [0,0,1] , listb[6:9] == [0,0,0] ,
            # pour la règle [BORDER,SLEEP,GEN]
            listb[0:3] == [0,0,0] , listb[3:6] == [0,1,0] , listb[6:9] == [1,1,0] ,
            # pour la règle [SLEEP,GEN,GEN]
            listb[0:3] == [0,1,0] , listb[3:6] == [1,1,0] , listb[6:9] == [1,1,0] ,
            # pour la règle [GEN,GEN,BORDER]
            listb[0:3] == [1,1,0] , listb[3:6] == [1,1,0] , listb[6:9] == [0,0,0] ,
            # pour la règle [SLEEP,GEN,SLEEP]
            listb[0:3] == [0,1,0] , listb[3:6] == [1,1,0] , listb[6:9] == [0,1,0] ,
            # pour la règle [SLEEP,ACT_2,GEN]
            listb[0:3] == [0,1,0] , listb[3:6] == [1,0,1] , listb[6:9] == [1,1,0] ,
            # pour la règle [GEN,GEN,GEN]
            listb[0:3] == [1,1,0] , listb[3:6] == [1,1,0] , listb[6:9] == [1,1,0] ,
            # pour la règle [SLEEP,GEN,ACT_1]
            listb[0:3] == [0,1,0] , listb[3:6] == [1,1,0] , listb[6:9] == [0,0,1] ,
            # pour la règle [GEN,ACT_1,BORDER]
            listb[0:3] == [1,1,0] , listb[3:6] == [0,0,1] , listb[6:9] == [0,0,0] ,
            # pour la règle [SLEEP,ACT_3,SLEEP]
            listb[0:3] == [0,1,0] , listb[3:6] == [0,1,1] , listb[6:9] == [0,1,0] , 
            # pour la règle [ACT_3,SLEEP,BORDER]
            listb[0:3] == [0,1,1] , listb[3:6] == [0,1,0] , listb[6:9] == [0,0,0] ,
            # pour la règle [SLEEP,SLEEP,ACT_3]
            listb[0:3] == [0,1,0] , listb[3:6] == [0,1,0] , listb[6:9] == [0,1,1] , 
            #la régle existe déja connue ou pas
            listb in old,
            #si on'est dans level 1
            board.pn == 2 and board.pi <= 2*board.pn - 2 and board.pj <= board.pn and board.finished == False,
            
            #si on'est dans level 2
            board.pn == 3 and board.pi <= 2 * board.pn - 2 and board.pj <= board.pn and board.finished == False,
            
             #on'est dans level 3
            board.pn == 4 and board.pi <= 2 * board.pn - 2 and board.pj <= board.pn and board.finished == False,
            
            #si on'est dans level 4
            board.pn == max_level and board.pi <= 2 * board.pn - 2 and board.pj <= board.pn and board.finished == False,
            
            
            # l'vancement est finie avec succés
            board.finished == True and board.success == True ,
            
            #ona fini sur une ligne
            board.finished == True and board.success == False ,
            # on'est dans le dérnier ligne
            board.pi == 2*board.pn - 2, 
            # on'est dan la ligne  de milieu dans un niveau pn
            board.pi == board.pn/2 ,
            #on'est avant le dernière ligne
            board.pi == 2*board.pn - 3,
            # sauter à un autre niveau on'est au BORDER
            board.pj > board.pn, 
            #au bout d'un ligne
            board.pj == board.pn, 
            # on peut encore avancer 
            board.pj <= board.pn,
            # passer à un autre niveau supérieur
            board.pj == board.pn and  board.success == False, 
            
            
            ]
        for i in range(9) : # savoire le ligne ou on'est
            state.append(board.pi == i)
            
        for j in range(6): # savoire le colonne ou on'est
            state.append(board.pj == j)
            
        #stocker les nouveaux régles
        if len(old) <= 240 and listb not in old:
            old.append(listb)
        

        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0

        return np.asarray(state)
    
    # une fonction binary qui convertir les cellules en binaire
    def binary(self,cell): 
        lst = []
        while cell !=0:
            remainder = cell % 2 #gives the exact remainder
            cell = cell // 2
            lst.append(remainder)
        if lst == []:
            lst = [0,0,0]
        else :
            if len(lst)<3:
                lst.append(0)
        return lst
    
    def set_reward(self,board,end):
        self.reward = 0
        if end:
            return
        if board.success:
            self.reward = 1
        return self.reward

    def network(self):
        model = Sequential()
        model.add(Dense(output_dim=120, activation='relu', input_dim=11))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=3, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        """if weights:
            model.load_weights(weights)
        return model"""

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 11)))[0])
        target_f = self.model.predict(state.reshape((1, 11)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 11)), target_f, epochs=1, verbose=0)
        
    