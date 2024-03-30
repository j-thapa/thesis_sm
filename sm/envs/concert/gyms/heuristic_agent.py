import numpy as np
import random

import time

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder



class HeuristicAgent():
    def __init__(
        self,
        shape,
        items_number,
        wall_encoding,
        agent_encoding,
        h_agent_encoding,
        object_encoding,
        random_factor,
        object_attached_encoding,
        goal_encoding,
        agent_attached_encoding,
        obstacle,
        h_agent_attached_encoding
        
    ) -> None:

      
        self.items = items_number #number of items 8 bit encoding
        self.walls_encoding = wall_encoding
        self.agent_encoding = agent_encoding
        self.h_agent_encoding = h_agent_encoding
        self.object_encoding = object_encoding
        self.object_attached_encoding = object_attached_encoding
        self.goal_encoding = goal_encoding
        self.shape = shape
        self.randomness = random_factor
        self.agent_attached_encoding = agent_attached_encoding
        self.h_agent_attached_encoding = h_agent_attached_encoding
     
        self.obstacle = obstacle
    
        # #records past 4 moves taken
        # self.selected_moves = []
        # #past move list obtained by A star
        # self.past_moves_list = []



    
    def path_list (self, observation, start, end, composite = False,goals_area = []):


        #reshape the flattened observation into grid
        matrix = np.transpose(np.array(observation).reshape([self.shape[0],self.shape[1]]))




        #the case where object is attached by both agent
        if composite == True:

            composite_loc = start
            start = composite_loc[0]

            #np.where(matrix==2)  loop them and make them obstacles aaccordingly

            #(array([2, 5, 8]), array([2, 2, 1])) coordinates are like this (x,x,x) and(y,y,y)

            

 


            #check whether the object attached is already on one of the goal coordinates
            if (start[0], start[1]) in goals_area:
                return("pick/drop") #drop the object



            goals = [x for x in goals_area if matrix[x[1],x[0]]== self.goal_encoding or 
            matrix[x[1],x[0]]== self.agent_attached_encoding or matrix[x[1],x[0]]==self.h_agent_attached_encoding ]
            #goals = [(goal[0][x], goal[1][x]) for x in range(len(goal[0]))]


            distance = [((e[0] - start[0].item())**2 + (e[1]- start[1].item())**2)**1/2 for e in goals]

            end_idx = distance.index(min(distance))
            end = goals[end_idx] #nearest goal

            print(start, end, "start end of composite object")

        

       

            
       # stationary items like wall, unattached object and attached agents are obstacles to avoid

        print(np.where(matrix==2),"coordinates are like this")
      
        matrix[matrix == 2] = -1
        matrix[matrix==19] = -1
        matrix[matrix==18] = -1
        matrix[matrix==1] = -1

        matrix[matrix>=0] = 1
        matrix[start[1], start[0]] = 1
        matrix[end[1], end[0]] = 1


        matrix = matrix.tolist()            
        grid = Grid(matrix=matrix)
        start = grid.node(start[0].item(), start[1].item())
        end = grid.node(end[0].item(), end[1].item())

        #no diagonal movement

        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        path, runs = finder.find_path(start, end, grid)

        moves_list = []
       
        #change movements into up,down,left and right
        for idx in range(len(path)-1):

            #Actions_extended = ["left", "right", "up", "down", "pick", "drop", "do_nothing"]
      
            if path[idx+1][0] > path[idx][0]:
                moves_list.append('right') #right
            elif path[idx+1][0] < path[idx][0]:
                moves_list.append('left') #left
            if path[idx+1][1] > path[idx][1]:
                moves_list.append('down') #down
            elif path[idx+1][1] < path[idx][1]:
                moves_list.append('up') #up

        grid.cleanup()

        print(moves_list)

    
    


    
        return moves_list
 
    
    def next_move(self, observation, composite = False, goals_area = [], start=(), end=()):

        #select next move based on observation for heuristic agent



        if composite == True:

           #get moves list
            moves_list = self.path_list(observation, composite = True, goals_area = goals_area, start = start, end = end)


            if len(moves_list)>0:

                return moves_list[0] #first move from the list
            else:
                return "do_nothing" #do_nothing

        
        else:

            #select a random move or move towards object for heuristic agent

            if np.random.choice([0,1], p=[1-self.randomness, self.randomness]) == 1:
                print("random move selected")
                select = np.random.choice(['up', 'down', 'left', 'right', 'do_nothing'])#select random move
  
                return select


            else:
                #get moves list
                moves_list =self.path_list(observation, composite = False, goals_area = goals_area,  start = start, end = end)
                
                if len(moves_list) == 1 and moves_list[0] != "do_nothing" : #it means the agent can reach the object and it is not attached
                    return ("pick/drop") #pick the object 
                else:
                    if len(moves_list)<1:
                        return "dp_nothing" #do nothing if moves list is empty

                    else:
        
                       
                        return moves_list[0] #return first move from the list



    
    
