import numpy as np
import random

import time

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

class RandomAgent():
    def __init__(
        self,
        shape,
        items_number,
        wall_encoding,
        agent_encoding,
        h_agent_encoding,
        r_agent_encoding,
        object_encoding,
        random_factor,
        object_attached_encoding = 5,
        goal_encoding = 2,
        agent_attached_encoding = 6,
        h_agent_attached_encoding = 8,
        goals_coord = [(1,1),(8,1),(1,8),(8,8)]
        
    ) -> None:

      
        self.items = items_number #number of items 8 bit encoding
        self.walls_encoding = wall_encoding
        self.agent_encoding = agent_encoding
        self.h_agent_encoding = h_agent_encoding
        self.r_agent_encoding = r_agent_encoding
        self.object_encoding = object_encoding
        self.object_attached_encoding = object_attached_encoding
        self.h_agent_attached_encoding = h_agent_attached_encoding
        self.goal_encoding = goal_encoding
        self.shape = shape
        
        self.agent_attached_encoding = agent_attached_encoding
        self.previous_goal_loc =[]



        #select random move based on random_factor
        self.randomness = random_factor
        
        #goals coordinates which will define the path for random obstacle, will hop from one goal to another depending on their weights
        self.goals_coord = goals_coord
        #give weights to goals of random dynamic obstacle; it will choose the goal according to these weights 0.1,0.2,0.3 accordingly on basis of their index
        self.goals_weights = [(self.goals_coord.index(i)+1)/10 for i in self.goals_coord]

        self.path_taken=[]

    
    def next_move(self, observation):



        #decides the next move of random agent based on observation

        temp_grid = []


   
        for x in range(0, len(observation),self.items):

            #encode observation into grid based on their items encoding, 1 means empty cell
    
            if observation [x:x+self.items] [self.walls_encoding] == 1 : 
                temp_grid.append(0)

            elif observation [x:x+self.items] [self.agent_encoding] == 1 :
                temp_grid.append(-1) #agent_wall_dynamic
            
            elif observation [x:x+self.items] [self.goal_encoding] == 1 :
                 temp_grid.append(-2) #goal

            elif observation [x:x+self.items] [self.r_agent_encoding] == 1 :
                 temp_grid.append(9) #random_agent
           
            elif observation [x:x+self.items] [self.object_encoding] == 1 :
                temp_grid.append(-4) #end_obj
            

            elif observation [x:x+self.items] [self.h_agent_encoding] == 1 :
                temp_grid.append(-1) #start_agent
           
            elif observation [x:x+self.items] [self.object_attached_encoding] == 1 :
                temp_grid.append(-1) #object_attached_encoding
            
            elif observation [x:x+self.items] [self.agent_attached_encoding] == 1 :
                temp_grid.append(-1) #agent_attached
                
            elif observation [x:x+self.items] [self.h_agent_attached_encoding] == 1 :
                temp_grid.append(-1) #agent_attached
            

 
            else:
                temp_grid.append(1)
        
        #reshape them elements of array in 10x10 grid
        matrix = np.array(temp_grid).reshape([self.shape[0],self.shape[1]])



        #select a random move from available moves according to value of randomness
        if np.random.choice([0,1], p=[1-self.randomness, self.randomness]) == 1:

            matrix = matrix.T
            
            #random agent index
            r_agent_idx = np.where(matrix == 9)
            r_x, r_y = r_agent_idx[0].item(),  r_agent_idx[1].item()


            dict_direction =  {2:matrix[r_x-1,r_y],
            3:matrix[r_x+1,r_y], 0:matrix[r_x,r_y-1],
            1:matrix[r_x,r_y+1]} #up,down,left,right and their corresponding coordinates
        


            #moves available if those coordinates are free
            available_moves = [x for x in dict_direction if dict_direction[x] > 0]





            #select a move based on sorrounding  free coordinates
            if len(available_moves)>=1:
                next_move = random.choice(available_moves)
            else:
                next_move = 6 #if no sorrounding coordinates are free


            return(next_move)

        else:
            #select the move for the next goal based on the goals coordinates not randomly

            #agent cooridnates 
       
            r_agent_idx = np.where(matrix == 9)
            r_x, r_y = r_agent_idx[0].item(),  r_agent_idx[1].item()

            #convert matrix to grid to feed path finding algo
            orig_matrix = matrix

            matrix =np.transpose(matrix).tolist()  
            
            grid = Grid(matrix=matrix)

            #random agent coordinate is starting node
            start = grid.node(r_x, r_y)

           
            
            
    

            #select the coordinate with minimum weight as dynamic object goal
            goal_idx = self.goals_weights.index(min(self.goals_weights))
            dynamic_goal = self.goals_coord[goal_idx]
            
            
            #increase goal weight by 1 if it is achieved by the dynamic obstacle
            if self.goals_coord[goal_idx] == (r_x, r_y):
                self.goals_weights[goal_idx] +=1
              
                #reset if goal is final goal and it is reached
                if  self.goals_coord[-1] == dynamic_goal:
                    self.goals_weights = [(self.goals_coord.index(i)+1)/10 for i in self.goals_coord]
                    self.path_taken=[]

            

            #check whether the goal coordinate is free or not
            
            
            #if goal coordinate is not free
            if orig_matrix[dynamic_goal[0], dynamic_goal[1] ] != 1:
                #get neighboring coordinates of the not free goal coordinate
                neigh_coord = [(dynamic_goal[0]+1, dynamic_goal[1]), (dynamic_goal[0]-1, dynamic_goal[1]), (dynamic_goal[0], dynamic_goal[1]+1), (dynamic_goal[0], dynamic_goal[1]-1)]
                #check if the dynamic agent has reached to neighboring coordinates of block goal
                if (r_x, r_y) in neigh_coord:
                    #then add weight on that goal ie it will be assumed as the goal is reached
                    self.goals_weights[goal_idx] += 1
                    #reset if the goal is final goal\
                    if  self.goals_coord[-1] == dynamic_goal:
                        self.goals_weights = [(self.goals_coord.index(i)+1)/10 for i in self.goals_coord]


            #check again for weights and get the lowest weight as new goal
            dynamic_goal = self.goals_coord[self.goals_weights.index(min(self.goals_weights))]


            
            #end node is dynamic goal(given goal coordinates) based on their weights
            end = grid.node(dynamic_goal[0],dynamic_goal[1])

            #algorithm for shortest path
            finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
            path, runs = finder.find_path(start, end, grid)

     
    

            moves_list = []
            

            #change movements into up,down,left and right
            for idx in range(len(path)-1):

                #Actions_extended = ["left", "right", "up", "down", "pick", "drop", "do_nothing"]
                if path[idx+1][0] > path[idx][0]:
                    moves_list.append(1) #right
                elif path[idx+1][0] < path[idx][0]:
                    moves_list.append(0) #left
                if path[idx+1][1] > path[idx][1]:
                    moves_list.append(3) #down
                elif path[idx+1][1] < path[idx][1]:
                    moves_list.append(2) #up
            
         

            grid.cleanup()
            
    
            
            if len(moves_list)>0:
                
                self.path_taken.append(path[1])
                
                return moves_list[0]
            else:
                return 6 #if no move just send do nothing







class HeuristicAgent():
    def __init__(
        self,
        shape,
        items_number,
        wall_encoding,
        agent_encoding,
        r_agent_encoding,
        h_agent_encoding,
        object_encoding,
        random_factor,
        object_attached_encoding,
        goal_encoding,
        agent_attached_encoding,
        dynamic,
        obstacle,
        three_grid_object
        
    ) -> None:

      
        self.items = items_number #number of items 8 bit encoding
        self.walls_encoding = wall_encoding
        self.agent_encoding = agent_encoding
        self.h_agent_encoding = h_agent_encoding
        self.r_agent_encoding = r_agent_encoding
        self.object_encoding = object_encoding
        self.object_attached_encoding = object_attached_encoding
        self.goal_encoding = goal_encoding
        self.shape = shape
        self.randomness = random_factor
        self.agent_attached_encoding = agent_attached_encoding
        self.dynamic = dynamic
        self.obstacle = obstacle
        self.three_grid_object = three_grid_object
        # #records past 4 moves taken
        # self.selected_moves = []
        # #past move list obtained by A star
        # self.past_moves_list = []



    
    def path_list(self, observation, composite = False,  goals_area = []):

        #get list of moves from heruistic agent or composite agent to goal or object based on observation

        temp_grid = []

        #convert observation items to grid ; 1 refers to free cells; can move from +ve cells
        for x in range(0, len(observation),self.items):

            #for walls
    
            if observation [x:x+self.items] [self.walls_encoding] == 1 : 
                temp_grid.append(0)

            elif observation [x:x+self.items] [self.agent_encoding] == 1 :
                temp_grid.append(-2) #agent_policy

            elif observation [x:x+self.items] [self.r_agent_encoding] == 1 :
                temp_grid.append(-3) #agent_dynamic
            
            elif observation [x:x+self.items] [self.goal_encoding] == 1 :
                 temp_grid.append(2) #goal
           
            elif observation [x:x+self.items] [self.object_encoding] == 1 :
                temp_grid.append(4) #object
            
            elif observation [x:x+self.items] [self.h_agent_encoding] == 1 :
                temp_grid.append(3) #start_agent
           
            elif observation [x:x+self.items] [self.object_attached_encoding] == 1 :
                temp_grid.append(5) #object_attached_encoding
            
            elif observation [x:x+self.items] [self.agent_attached_encoding] == 1 :
                temp_grid.append(6) #_agent_attached

            elif  observation [x:x+self.items] [8] == 1:
                temp_grid.append(7) #heuristic agent_attached

 
            else:
                temp_grid.append(1)
        
        #reshape them elements of array in 10x10 grid
        matrix = np.array(temp_grid).reshape([self.shape[0],self.shape[1]])





        #the case where object is attached by both agent
        if composite == True:




            if len(goals_area) > 1: #means goal area 2x2

                start = np.where(matrix == 5) #start object attached


                #check whether the object attached is already on one of the goal coordinates
                if (start[0], start[1]) in goals_area:
                    return([5]) #drop the object


                #replaces the 3*3 section of obs matrix where the obstacle is centre by 3*3 zeros matrix
                crop_matrix = matrix[1:-1, 1:-1]
                obs_idx = np.where(crop_matrix == 4 )
                matrix[obs_idx[0].item():obs_idx[0].item()+3, obs_idx[1].item():obs_idx[1].item()+3] = np.zeros([3,3])
         





                #goals = [(goal[0][0], goal[1][0]), (goal[0][1], goal[1][1]), (goal[0][2], goal[1][2]),(goal[0][3], goal[1][3]) ]
                goals = [x for x in goals_area if matrix[x[0],x[1]]==2 or matrix[x[0],x[1]]==6 or matrix[x[0],x[1]]==7 ]
                #goals = [(goal[0][x], goal[1][x]) for x in range(len(goal[0]))]
          

  
                distance = [((e[0] - start[0].item())**2 + (e[1]- start[1].item())**2)**1/2 for e in goals]

           

                end_idx = distance.index(min(distance))
                end = goals[end_idx] #nearest goal

              

              

                    
                

                # #replaces the 3*3 section of obs matrix where the obstacle is centre by 3*3 zeros matrix
                # crop_matrix = matrix[1:-1, 1:-1]
                # obs_idx = np.where(crop_matrix == 4 )
                # matrix[obs_idx[0].item():obs_idx[0].item()+3, obs_idx[1].item():obs_idx[1].item()+3] = np.ones([3,3]) * -1
                # #matrix = matrix[1:-1, 1:-1]
                # print(matrix)
            

            else:
    
                if self.obstacle == True and self.dynamic == False:
                    #replaces the 3*3 section of obs matrix where obstacle is centre by 3*3 zeros matrix
                    crop_matrix = matrix[1:-1, 1:-1]
                    obs_idx = np.where(crop_matrix == 0 )
                    matrix[obs_idx[0].item():obs_idx[0].item()+3, obs_idx[1].item():obs_idx[1].item()+3] = np.zeros([3,3])
                
                #no need to make 3 x 3 wall grid for dynamic obstacle
                # elif self.obstacle == True and self.dynamic == True:
                #     r_agent_idx = np.where(matrix == -3 )
                #     matrix[r_agent_idx[0].item()-1:r_agent_idx[0].item()+2, r_agent_idx[1].item()-1:r_agent_idx[1].item()+2] = np.zeros([3,3])

            
                
                #object attached not in matrix it means object attached is in the goal already
                if np.where(matrix == 5) == goals_area[0]:
                    
                    return([5]) #drop
                else:
                 


                    start = np.where(matrix == 5) #start object attached
                    end = goals_area[0] #end goal
                  

           
        else:

            #take attached agent as an obstacle if the heuristic is not running for composite object
            matrix[matrix == 6] = -1

      

            if (len(goals_area)) > 1 : #goal area configuration

                #if three grid contrib
                if self.three_grid_object:
                    start = np.where(matrix == 3)
                    end =  np.where(matrix == 4)

                    objects_list = [(list(end)[0][x],list(end)[1][x]) for x in range(len(list(end)[0]))] #objects list
                    end_coord = np.median(objects_list, axis = 0) 
                    end_idx = [x for x in range(len(objects_list)) if (objects_list[x][0]==end_coord[0] and objects_list[x][1]==end_coord[1])]
                    end = objects_list[end_idx[0]] #select middle object as end point
               
                    for obj in objects_list:
                       
                        if obj != end: #object is treated as obstacle if that object is not selected as goal
                            matrix[obj] = -4
                        
                    

                else:


                    start = np.where(matrix == 3)
                    #get rid of object from objects list if  thats already in the goal 

                    end =  np.where(matrix == 4)
                    



                    objects_list = [(list(end)[0][x],list(end)[1][x]) for x in range(len(list(end)[0]))] #objects list

                    
                    #take object as end option if its not in goal area already and if it is in goal area already its an obstacle
                    objects_ = []
                    for object_ in objects_list:
                        if object_ not in goals_area:
                            objects_.append(object_)
                        else:
                            matrix[object_] = -1

                    #objects_ = [x for x in objects_list if x not in goals_area]


                    distance = [((e[0] - start[0].item())**2 + (e[1]- start[1].item())**2)**1/2 for e in objects_ ]

                    
    


                    try:
                        end_idx = distance.index(min(distance))
                        end = objects_[end_idx]
                    except:
                        print(objects_, distance, objects_list)
                        print(start, end, "h_agent and objects ")
                        print(objects_, distance, objects_list)
                        print(goals_area)
                        end_idx = distance.index(min(distance))
                        end = objects_[end_idx]
                        

                
                    for obj in objects_:
                        if obj != end or obj not in objects_list: #object is treated as obstacle if that object is not selected as goal
                            matrix[obj] = -4
                

            else:
      
    

                #heuristic agent goes and attach to the object 
                if 3 not in matrix:
                    if -1 in matrix and 6 in matrix:

                        return([6]) #do_nothing i.e heuristic agent is the attached agent as there is already agent(-1)
                    
                    else:
                        matrix[matrix == 2] = 3 #it means the heuristic agent is in the goal so represented by goal

                if 4 not in matrix:
                    return [6] #object is already on the goal need to do nothing

                #take attached agent as an obstacle
                matrix[matrix == 6] = -1


                if 6 and -2 not in matrix: #i.e presence of no policy agent(policy agent is on the goal, can see only goal)
                    matrix[matrix == 2] = -1 #take goal as wall as the policy agent is in the goal

        #start is coordinates of agent node, end of object node
            
                start = np.where(matrix == 3)
                end  = np.where(matrix == 4)

      
       

    #matrix register as grid
    #    #get start and end nodes from initial matrix but feed tranposed matrix
            
       
        matrix[matrix>0] = 1
        matrix[matrix<=0] = -1
        matrix = np.transpose(matrix).tolist()            
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
                moves_list.append(1) #right
            elif path[idx+1][0] < path[idx][0]:
                moves_list.append(0) #left
            if path[idx+1][1] > path[idx][1]:
                moves_list.append(3) #down
            elif path[idx+1][1] < path[idx][1]:
                moves_list.append(2) #up

        grid.cleanup()

    
    


    
        return moves_list
 
    
    def next_move(self, observation, composite = False, goals_area = []):

        #select next move based on observation for heuristic agent



        if composite == True:

           #get moves list
            moves_list = self.path_list(observation, composite = True, goals_area = goals_area)


            if len(moves_list)>0:
            #     self.selected_moves.append(moves_list[0]) #append moves list
             
            #     if len(self.selected_moves) > 3: #when moves list is 4
             
            #         if set(self.selected_moves)=={0,1} or set(self.selected_moves)=={2,3}: #check loop condition one (left,right) or (up, down) moves only
                      

                
                    
            #             #check loop conidtion 2 
            #             if (self.selected_moves[0]==self.selected_moves[2]) and (self.selected_moves[1]==self.selected_moves[3]): #alternate repeating moves
                          
            #                 #looping ; empty selectd moves list
            #                 self.selected_moves = [] 

                            
            #                 return self.past_moves_list[1] # select  second move from past moves list to break the loop
                    
            #         self.selected_moves = []     #update moves list  
                 



            #     self.past_moves_list = moves_list

           
                return moves_list[0] #first move from the list
            else:
                return 6 #do_nothing

        
        else:

            #select a random move or move towards object for heuristic agent

            if np.random.choice([0,1], p=[1-self.randomness, self.randomness]) == 1:
                select = np.random.randint(low=0, high=5, size=1)[0] #select random move
                if select == 4 :
                    return 6 #do_nothing
                else:
                    return select


            else:
                #get moves list
                moves_list =self.path_list(observation, composite = False, goals_area = goals_area)
                
                if len(moves_list) == 1 and moves_list[0] != 6 : #it means the agent can reach the object and it is not attached
                    return (4) #pick the object 
                else:
                    if len(moves_list)<1:
                        return 6 #do nothing if moves list is empty

                    else:
                        # self.selected_moves.append(moves_list[0]) #append moves list
             
                        # if len(self.selected_moves) > 3: #when moves list is 4
             
                        #     if set(self.selected_moves)=={0,1} or set(self.selected_moves)=={2,3}: #check loop condition one (left,right) or (up, down) moves only
                      
                        #     #check loop conidtion 2 
                        #         if (self.selected_moves[0]==self.selected_moves[2]) and (self.selected_moves[1]==self.selected_moves[3]): #alternate repeating moves
                          
                        #         #looping ; empty selectd moves list
                        #             self.selected_moves = [] 

                            
                        #             return self.past_moves_list[1] # select  second move from past moves list to break the loop
                    
                        # self.selected_moves = []  #update moves list 
                        # self.past_moves_list = moves_list
                       
                        return moves_list[0] #return first move from the list



    
    
