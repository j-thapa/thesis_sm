import random

import numpy as np
from typing import List
import itertools
from itertools import product
import time
from sm.envs.concert.gridworld.items import ItemKind, ItemBase
from sm.envs.concert.gridworld import items, games, engine, defaults
from sm.envs.concert.gyms.examples.utils import handle_not_element



class WarehouseGame (games.GameBase_multiagent):
    """
    multi-agent game; environment just uses agent IDs, agent objects are created and used in the game;
    """
    def __init__(
        self,
        agent_ids: List[str] = ['agent_1'],
        shape=(10, 10),
        max_steps: int = 100,
        num_objects: int = 1,
        obstacle: bool = True,
        seed = 42
    ) -> None:
        super().__init__()
        self.shape = shape
        self.max_steps = max_steps
        self.agent_ids = agent_ids
        self.agent_dict = {} # mapping of agent ID to agent object
        self.num_objects = num_objects
        #self.num_goals = num_goals # not used
        self.objects: List[items.ObjectItem] = None
        self.obstacle= obstacle
        self.seed = seed

        #TODO pass parameter to know whether heuristic is being used or not in this environment
        self.heuristic = True

        self.moves_since_pick = 1000 # the number of successful moves since the successful pick action

       
        self.rng = np.random.default_rng(self.seed) # seeding the random number generator for improved reproducibility of training results

     
        # configuration of reward scheme parameters
        self.reward_game_success =  1.0
        self.reward_drop_success =  1.25
        self.reward_pick_success =  0.75
        self.reward_each_action = -0.01
        self.reward_illegal_action = -0.07



    def _create_game(self, goal_area:bool = False):
        """
        create game with random locations of items for any settings
        """
        
        self.engine = engine.GridEngine(self.shape, rng=self.rng)
        w = items.WallItem.create_border(self.shape)
        self.engine.add(w)
        



            
        # create goal area for the game

        self.goals, goals_coordinates = self.create_goal_area(size=2)
           
        #select objects location randomly not overlapping with goal area coordinates
        object_cells = list(itertools.product([x for x in range(1,self.shape[0]-1)], [x for x in range(1,self.shape[1]-1)]))
        object_cells = [object_cell for object_cell in object_cells if (object_cell[0], object_cell[1]) not in goals_coordinates]
 

                                
        self.objects = []
        for i in range(self.num_objects): # TODO if goal_area == False, self.num_objects is not used
            chosen_coord = tuple(self.rng.choice(object_cells))
            self.objects.append(items.ObjectItem(chosen_coord, num_carriers = len(self.agent_dict)))
            object_cells.remove(chosen_coord)


        self.engine._add(list(self.goals))
        self.engine._add(list(self.objects))


        for agent_id in self.agent_ids:

            #creates heuristic agen and policy agents
            agent = items.PickAgent((-1, -1), kind = agent_id)


            # if 'heuristic_agent' in agent_id:
            #     target_object = np.random.choice(self.non_targets)
            #     agent.target_object = target_object
            #     self.non_targets.remove(target_object)


                
            self.agent_dict[agent_id] = agent
            
        self.engine._add_randomly(list(self.agent_dict.values())) #add agents randomly 
       
         
            

  
        # TODO add obstacle as wall maybe for complex settings
            # if self.obstacle:           

            #         # get all locations within walls
            #     grid_cells = list(itertools.product([x for x in range(1,self.shape[0]-1)], [x for x in range(1,self.shape[1]-1)]))
                    

            #     #lambda func which gives list of neighboring coordinates
            #     neighbors = lambda x, y : [(x2, y2) for x2 in range(x-1, x+2)
            #                                     for y2 in range(y-1, y+2)
            #                                     if (-1 < x <= self.shape[0] and
            #                                         -1 < y <= self.shape[1] and
            #                                         (x != x2 or y != y2) and
            #                                         (0 <= x2 <= self.shape[0]) and
            #                                         (0 <= y2 <= self.shape[1]))]
                    
            #         # obstacle shoudn't be around goal location and object location and around their neighboring cells
            #         occupied_cells = [(object_x, object_y),(goal_x, goal_y)]
            #         occupied_cells.extend(neighbors(object_x,object_y))
            #         occupied_cells.extend(neighbors(goal_x,goal_y))

            #         free_obs_cells= [x for x in grid_cells if x not in occupied_cells]
                    
            #         #if obstacle is static just use wall item in free cells as obstacle
            #         self.obstacles = [items.WallItem((random.choice(free_obs_cells)))]
            #         self.engine._add(list(self.obstacles)) 
            #         self.engine._add(list(self.objects))
            #         self.engine._add(list(self.goals))
                    
            #         self.engine._add_randomly(list(self.agent_dict.values())) 
            # else:
            #     # no obstacle
            #     self.engine._add(list(self.objects))
            #     self.engine._add(list(self.goals))
            #     self.engine._add_randomly(list(self.agent_dict.values()))

       


    # TODO make game deterministic in case where all items and even shape of grid have to be passed in a dictionary 
    def _create_game_deterministic(self, items_dict):
        """
        initial locations for goals, objects agents and wall are specified
        """
        self.engine = engine.GridEngine(self.shape, rng=self.rng)
        w = items.WallItem.create_border(self.shape)
        self.engine.add(w)

        #assert len(self.agent_ids) == 2
        for agent_id in self.agent_ids:
            if agent_id != 'random_agent':
                if agent_id == 'heuristic_agent':
                    agent_obj = items.PickAgent((7, 1), kind = agent_id) # location of heuristic agent
       

                else:
                    agent_obj = items.PickAgent((8, 1), kind = agent_id) # location of policy agent
                self.agent_dict[agent_id] = agent_obj

        if self.num_objects == 2:
            self.objects = [items.ObjectItem((1, 8), num_carriers = len(self.agent_ids)), items.ObjectItem((8, 8), num_carriers = len(self.agent_ids))]
        else:
            self.objects = [items.ObjectItem((3, 6), num_carriers = len(self.agent_ids))]
        if goal_area:
            self.goals, goals_coordinates = self.create_goal_area(size=2, upper_left_corner=(3,3))
        else:
            if self.obstacle:
                #if dynamic obstacle
                if self.dynamic:

                    #all available grid cells in the grid
                    grid_cells = list(itertools.product([x for x in range(1,self.shape[0]-1)], [x for x in range(1,self.shape[1]-1)]))

                    #already occupied grid cells by agents, goal and object
                    occupied_cells = [(7, 1), (1, 1), (8, 1), (1,8), (4,5)]

                    #free cells
                    free_obs_cells = [x for x in grid_cells if x not in occupied_cells]

                    #if initial position is free
                    if self.initial_dynamic_pos in free_obs_cells:
                        #add dynamic obstacle in a fixed initial position while generating env
                        self.obstacles = [items.PickAgent(self.initial_dynamic_pos, kind="random_agent")]
                    else:

                        #adjacent grid cells of  initial position i.e left,right,up, down and all diagonal cells to the postion cell
                        points_near = [(self.initial_dynamic_pos[0]+1, self.initial_dynamic_pos[1]),(self.initial_dynamic_pos[0]-1, self.initial_dynamic_pos[1]),
                                        (self.initial_dynamic_pos[0]+1, self.initial_dynamic_pos[1]+1),(self.initial_dynamic_pos[0]-1, self.initial_dynamic_pos[1]-1),
                                        (self.initial_dynamic_pos[0]+1, self.initial_dynamic_pos[1]-1),(self.initial_dynamic_pos[0]-1, self.initial_dynamic_pos[1]+1),
                                        (self.initial_dynamic_pos[0], self.initial_dynamic_pos[1]+1),(self.initial_dynamic_pos[0], self.initial_dynamic_pos[1]-1)]

                        #if initial position is not free check adjacent free position and use that for initial position
                        free_adjacent_points = [x for x in points_near if x in free_obs_cells]
                        self.obstacles = [items.PickAgent(free_adjacent_points[0], kind="random_agent")]


                    
                   
                else:
                    self.obstacles = [items.WallItem((2,6))]


                
                self.engine._add(list(self.obstacles ))
            self.goals = [items.GoalItem((4, 5))]
        
        
        self.engine._add(list(self.objects ))
        #remove random agent from agent dict if it is there
        self.agent_dict.pop("random_agent", None)
        self.engine._add(list(self.agent_dict.values()))
        self.engine._add(list(self.goals ))
        
        
        if self.obstacle and self.dynamic:
            # add dynamic obstacle as random agent to agent dict
            self.agent_dict["random_agent"] = self.obstacles[0]


   


    def create_goal_area(self, size:int = 2, upper_left_corner = None) -> List[items.GoalItem]:
        """
        Creates a randomly positioned square goal area; the goal area does not cover locations on the edges of the gridworld;
        @param size: the size of the goal area is size x size
        @return: a list of GoalItem objects making up the goal area
        """
        area = [] # the return value
        # create upper left corner coordinates for goal area;
        # coordinates may not be on gridworld edges
        if not upper_left_corner == None:
            # deterministic goal area
            ulc_x = upper_left_corner[0]
            ulc_y = upper_left_corner[1]
        else:
            # random goal area
            ulc_x = np.random.randint(2, self.shape[0] - 3)  # if shape is 10 then 10-3 select any from 2 to 7 grid do not select 0,1 and 8,9
            ulc_y = np.random.randint(2, self.shape[1] - 3)
        # determine lower right coordinates for goal area
        lrc_x = ulc_x + size - 1
        lrc_y = ulc_y + size - 1
        # determine required (x,y) offsets for goal area, so no goal items are on gridworld edges
        offset_x = max(0, lrc_x - (self.shape[0] - 3))
        offset_y = max(0, lrc_y - (self.shape[1] - 3))
        ulc_x = ulc_x - offset_x
        ulc_y = ulc_y - offset_y

        area_coordinates = []
        # construct goal area
        for x in range(ulc_x, ulc_x + size):
            for y in range(ulc_y, ulc_y + size):
                area.append(items.GoalItem((x, y)))
                area_coordinates.append((x,y))

        return (area,area_coordinates)

    def get_key_from_value(self, d, val):
        for key, value in d.items():
            if value == val:
                return key
        return None

    def _step(self, step_data: games.StepData_multiagent) -> games.StepData_multiagent:
        #self.rng.shuffle(self.agent_ids) # avoid bias due to the sequence of agents in list agent_ids

        #TODO make this happen not working right now
        if 'agent_' in self.agent_ids:
            #heuristic agents always act first ih the environment
            self.agent_ids.remove('agent_')
            self.agent_ids.append('agent_')


        for agent in self.agent_ids:

            agent_obj = self.agent_dict[agent]

            if agent_obj.attached:
                if agent_obj.picked_object.attachable != True:
                    # add small reward as one step of attaching is achieved
                    # step_data.add_reward(0.005, agent)
                    # print("adding small reward for correct attachment and holding to it")
                    pass

  

            step_data[agent]["reward"] = float(0.)
            action = step_data[agent]['action']


            #for now reward only for policy agents
            # if "heuristic_agent" not in agent:

                # TODO skip if agent is attached to the object and the object is also attached



                    # small penalty for each action; this should incentivise to reach the goal as quick as possible;
                    # no penalty if object is attached and being driven by heuristic agent


            step_data.add_reward(self.reward_each_action, agent) 
            step_data[agent]["infos"]["game_success"] = float(0.0)
            step_data[agent]["infos"]["steps_exceeded"] = float(0.0)
            #step_data[agent]["infos"]["first_object_attached"] = float(0.0) #use in goal area to see when first object is attached by both agents

            if action in ["up", "down", "left", "right"]:
                step_data[agent]["infos"]["pick_success"] = float(0.0)
                step_data[agent]["infos"]["pick_failure"] = float(0.0)
                step_data[agent]["infos"]["drop_failure"] = float(0.0)

                #TODO one is driver and another is driven for composite object all the time whether to move or drop
                #i.e one give move other should do_nothing, otherwise action failure, one action should be donothing condition
                #heuristic fix heuristic driver in other case interchangable doesnt matter TODO
                #for heurisdtic heuristic shuld be bnot doing nothing
                

                if agent_obj.attached and agent_obj.picked_object.attachable != True:
                    carriers_ = [self.get_key_from_value(self.agent_dict, carrier) for carrier in agent_obj.picked_object.carriers]

                    #print([step_data[x]['action'] for x in carriers_],"wht are actions and their carriers", carriers_)

                    if self.heuristic:
                        normal_agent_action = step_data[[x for x in carriers_ if 'heuristic' not in x][0]]['action']
                       # print("normal agent action", normal_agent_action)
                        if normal_agent_action != 'do_nothing':
   
                            success = False
                        else:
                            success = agent_obj.move(action)


                    else:
                        actions_carriers = [self.action_dict[x] for x in carriers_]
                        if 'do_nothing' not in actions_carriers:
                            success = False
                        else:
                            success = agent_obj.move(action)
                else:
                    success = agent_obj.move(action)

                    
                if success:
                    step_data[agent]["infos"]["move_failure"] = float(0.0)
                else:
                    
                    step_data.add_reward(self.reward_illegal_action, agent) 
                    step_data[agent]["infos"]["move_failure"] = -float(1.0)

   
            elif action == "pick/drop":
      
                if agent_obj.attached != True:
                    #pick action should be done
                    
                    step_data[agent]["infos"]["move_failure"] = float(0.0)
                    step_data[agent]["infos"]["drop_failure"] = float(0.0)
                    success = agent_obj.pick()
                    if success: 

                    # for now if an object is picked sucessfully and it is attached then reward (game_success/n_objects)
                    
                        # if 'heuristic_agent' not in agent:
                        if agent_obj.picked_object.attachable != True:
                            # print(" ------ agents have succcessful attachment of the object -----")
                            carriers = [self.get_key_from_value(self.agent_dict, carrier) for carrier in agent_obj.picked_object.carriers]
                            
                            for carrier in carriers:
                                step_data.add_reward(self.reward_pick_success, carrier) #adding sucessful pick reward for bot agents

                            

                        step_data[agent]["infos"]["pick_success"] = float(1.0)
                        step_data[agent]["infos"]["pick_failure"] = float(0.0)
                        
                    else:
                        # if 'heuristic_agent' not in agent:
                        step_data.add_reward(self.reward_illegal_action, agent) # illegal pick
                        step_data[agent]["infos"]["pick_success"] = float(0.0)
                        step_data[agent]["infos"]["pick_failure"] = -float(1.0)

                else:
                    #drop action if agent is attached
                    step_data[agent]["infos"]["move_failure"] = float(0.0)
                    step_data[agent]["infos"]["pick_failure"] = float(0.0)
                    step_data[agent]["infos"]["pick_success"] = float(0.0)
                    object_picked = agent_obj.picked_object
                    object_attachable = agent_obj.picked_object.attachable #whether the object attached by the agent was attached or not
                    carriers_ = [self.get_key_from_value(self.agent_dict, carrier) for carrier in object_picked.carriers]
    
                    if agent_obj.attached and agent_obj.picked_object.attachable != True:
                        
                        #print([step_data[x]['action'] for x in carriers_],"wht are actions and their carriers",[ self.get_key_from_value(self.agent_dict, carrier) for carrier in agent_obj.picked_object.carriers])

                        if self.heuristic:
                            normal_agent_action = step_data[[x for x in carriers_ if 'heuristic' not in x][0]] ['action']
                           # print("normal agent action", normal_agent_action)
                            if normal_agent_action != 'do_nothing':
    
                                success = False
                            else:
                                success = agent_obj.drop()


                        else:
                            actions_carriers = [step_data[x]['action'] for x in carriers_]
                            if 'do_nothing' not in actions_carriers:
                                success = False
                            else:
                                success = agent_obj.drop()
                    else:
                        
                       
                        success = agent_obj.drop()




                    
                    if success:

                        #TODO giving unnecessary reward in dropping

                        step_data[agent]["infos"]["drop_failure"] = float(1.0)
                        goal_locs = [g.loc for g in self.goals]
                        if 'heuristic' in agent and object_picked.dropped and any(np.array_equal(object_picked.loc, j) for j in goal_locs) :
                            # print("----heuristic agent has driven and drop the composite object correctly----")

                            for carrier in carriers_:

                                step_data.add_reward(self.reward_drop_success, carrier) # i.e heuristic agent has driven and drop the composite object correctly, add reward for both agents


                    else:
  
                        step_data.add_reward(self.reward_illegal_action, agent) # illegal drop
                        step_data[agent]["infos"]["drop_failure"] = -float(1.0)


            elif action =="do_nothing":
                step_data[agent]["infos"]["move_failure"] = float(0.0)
                step_data[agent]["infos"]["pick_failure"] = float(0.0)
                step_data[agent]["infos"]["pick_success"] = float(0.0)
                step_data[agent]["infos"]["drop_failure"] = float(0.0)

        # Check game end

        

        if self._check_game_success():
            
            for agent in self.agent_ids:

                # step_data.add_reward(self.reward_game_success, agent) # game success reward for agent getting attached
              
                step_data[agent]["infos"]["game_success"] = float(1.0)

                step_data[agent]['terminated'] = True
            step_data.terminate_game({"goal_reached": True})
            
        elif step_data.steps_exceeded(self.max_steps):
            for agent in self.agent_ids:
                # if 'heuristic_agent' not in agent:
                #     # step_data.add_reward(-1.0, agent)
                step_data[agent]["infos"]["steps_exceeded"] = -float(1.0)
                step_data[agent]['terminated'] = True
            step_data.terminate_game({"goal_reached": False})


            
        return step_data


    def _check_game_success(self) -> bool:

        #all objects are in goal then game success
        goal_locs = [g.loc for g in self.goals]
        ob_locs = [obj.loc for obj in self.objects]

        dropped_objs = [i for i in self.objects if i.dropped and any(np.array_equal(i.loc, j) for j in goal_locs)]


        # if len(dropped_objs) == self.num_objects:
        #    print("++++++++++++++++++++++++++++++++++++++++++++++++++ ALL OBJECTS ON GOALS +++++++++++++++++++++++++++++++++++++++++++++++++")

        return len(dropped_objs) == self.num_objects


    def _check_pick_success(self) -> bool:

        #all objects are in goal then game success

        picked_objs = [i for i in self.objects if i.attachable != True]


        # if len(picked_objs) == self.num_objects:
        #    print("++++++++++++++++++++++++++++++++++++++++++++++++++ ALL OBJECTS picked +++++++++++++++++++++++++++++++++++++++++++++++++")

        return len(picked_objs) == self.num_objects

        
  

    def _check_goal_find_goal(self) -> bool:
        """
        copied from find goal game; the game is successful if a goal has been found, i.e. the agent moves on a goal
        """
        the_items = self.engine.items(locs=self.agent.loc)
        the_items = [i for i in the_items if i.kind == items.ItemKind.GOAL]
        return len(the_items) > 0

    def _check_goal_movewithobject(self) -> bool:
        """
        game is successful if the object has been picked and the agent has made a number of moves with the attached object;
        """
        if self.agent.attached and self.moves_since_pick == 2:
            print("++++++++ PICK SUCCESSFUL, MOVED TWICE ++++++++")
            return True
        else:
            return False

    def _check_goal_movewithobject_until_goaldistance(self, goal_distance) -> bool:
        """
        game success if the agent has moved with the object until a given distance to the goal;
        """
        distance_to_goal = self.move_distance(self.agent, self.goals[0])
        if self.agent.attached and distance_to_goal == goal_distance:
            print("++++++++++++++++ MOVED WITH OBJECT; DISTANCE TO GOAL: {} +++++++++++++".format(goal_distance))
            return True
        else:
            return False

    def render(self, image_observation:bool=True) -> np.ndarray:
        return defaults.render(self.engine, image_observation=image_observation)

    def is_adjacent_to_impassable(self, direction: str, item: ItemBase) -> (bool, ItemBase):
        """
        Returns True if an impassable item is adjacent to param "item" in the direction of param "direction"
        """
        if direction == "left":
            loc = np.array([item.loc + (-1, 0)])
        elif direction == "right":
            loc = np.array([item.loc + (1, 0)])
        elif direction == "up":
            loc = np.array([item.loc + (0, -1)])
        elif direction == "down":
            loc = np.array([item.loc + (0, 1)])

        items = self.engine.items(loc)
        contains_impassable = False
        impassable_item = None
        for item in items:
            if item.impassable:
                contains_impassable = True
                impassable_item = item
                break
        return contains_impassable, impassable_item

    def is_adjacent_to(self, direction: str, item: ItemBase, other_item: ItemBase):
        """
        Returns True if other_item is adjacent to item in the direction of param direction
        """
        if direction == "left":
            loc = np.array([item.loc + (-1, 0)])
        elif direction == "right":
            loc = np.array([item.loc + (1, 0)])
        elif direction == "up":
            loc = np.array([item.loc + (0, -1)])
        elif direction == "down":
            loc = np.array([item.loc + (0, 1)])

        items = self.engine.items(loc)
        contains_item = False
        for it in items:
            if it == item:
                contains_item = True
                break
        return contains_item

def play_warehousegame():
    import matplotlib.pyplot as plt

    print("Use w,a,s,d to move and q to toggle pick/drop")

    game = WarehouseGame_1(max_steps=100, num_objects=1, num_goals=1, num_obstacles=1)
    game.reset()

    fig, ax = plt.subplots()
    mpl_img = ax.imshow(game.render())

    def on_press(event):
        if event.key == "w":
            game.step({game.agent: "up"})
        elif event.key == "s":
            game.step({game.agent: "down"})
        elif event.key == "a":
            game.step({game.agent: "left"})
        elif event.key == "d":
            game.step({game.agent: "right"})
        elif event.key == "q":
            game.step({game.agent: "drop"})
        elif event.key == "e":
            game.step({game.agent: "pick"})
        if game.done:
            print(game.reward_history)
            game.reset()
        mpl_img.set_data(game.render())
        fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    plt.show()


if __name__ == "__main__":
    play_warehousegame()