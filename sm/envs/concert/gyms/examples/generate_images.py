from concert.gridworld.defaults import render_observation
import numpy as np
import  csv
import matplotlib
import matplotlib.pyplot as plt

def create_episode(csv_file, image_folder, shape=(10,10), eps = '1'):
    '''
    
    create every timestep of an episode from a csv file
    
    '''
    matplotlib.use('Agg')
    with open(csv_file, 'r') as file:

        reader = csv.reader(file)
        for row in reader:
            if len(row)>1:
                    
    #row[0]: Observation , row[1]:next_observation, row[2]:Agent , row[3]: Action , row[4]: Action_success
    #row[5]:object attached or not , row[6]: episode completed or not, row[7]: episode, row[8]: timestep
  

                    # -1 means action not success and 0 means object is not attached to both agents
                    if eps == row[-2] and 'agent_1' in row[2]:
                        print("Creating image for  episode ", row[-2]," timestep ",row[-1])

                        img = render_observation(row[0],shape)

                        fig, ax = plt.subplots()
                        mpl_img = ax.imshow(img)
                        mpl_img.set_data(img)
                        fig.canvas.draw()
                        plt.savefig(f"{image_folder}image_{row[-2]}_{int(row[-1])}_before.png")

                        img = render_observation(row[1],(10,10))

                        fig, ax = plt.subplots()
                        mpl_img = ax.imshow(img)
                        mpl_img.set_data(img)
                        fig.canvas.draw()
                        plt.savefig(f"{image_folder}image_{row[-2]}_{int(row[-1])}_next.png")




def create_images(csv_file, image_folder, shape=(10,10), failed_action = True, agent_only=True, failed_episode=False):
    """
#csv_file : provide input csv file path from which images has to be generated from observations
#image_folder : output folder path for generated images
#failed_action : boolean to create images for time step for a failed action
#agent_only: boolean which will create time step images for failed action of agents only
#failed_episode : boolean that will create images for all time steps of a failed episode

    """
    matplotlib.use('Agg')
    with open(csv_file, 'r') as file:

        reader = csv.reader(file)
        failed_eps=[]
        for row in reader:
            if len(row)>1:
    #row[0]: Observation , row[1]:next_observation, row[2]:Agent , row[3]: Action , row[4]: Action_success
    #row[5]:object attached or not , row[6]: episode completed or not, row[7]: episode, row[8]: timestep
  
                if failed_action:
                    if agent_only:
                        # -1 means action not success and 0 means object is not attached to both agents
                        if '-1' in row[4] and 'agent_1' in row[2] and '0' in row[5]:
                            print("Creating image for failed action in episode ", row[-2]," timestep ",row[-1])

                            img = render_observation(row[0],shape)

                            fig, ax = plt.subplots()
                            mpl_img = ax.imshow(img)
                            mpl_img.set_data(img)
                            fig.canvas.draw()
                            plt.savefig(f"{image_folder}image_{row[-2]}_{int(row[-1])}_before.png")

                            img = render_observation(row[1],(10,10))

                            fig, ax = plt.subplots()
                            mpl_img = ax.imshow(img)
                            mpl_img.set_data(img)
                            fig.canvas.draw()
                            plt.savefig(f"{image_folder}image_{row[-2]}_{int(row[-1])}_next.png")
                    else:
                        if '-1' in row[4]:
                            print("Creating image for failed action in episode ", row[-2]," timestep ",row[-1])

                            img = render_observation(row[0],shape)

                            fig, ax = plt.subplots()
                            mpl_img = ax.imshow(img)
                            mpl_img.set_data(img)
                            fig.canvas.draw()
                            plt.savefig(f"{image_folder}image_{row[-2]}_{int(row[-1])}_before.png")

                            img = render_observation(row[1],(10,10))

                            fig, ax = plt.subplots()
                            mpl_img = ax.imshow(img)
                            mpl_img.set_data(img)
                            fig.canvas.draw()
                            plt.savefig(f"{image_folder}image_{row[-2]}_{int(row[-1])}_next.png")
                
                
                if  failed_episode:
                    
                    if '-1' in row[-3]:
                        print(row[-2]," added to the failed eps")
                        failed_eps.append(row[-2])
            
        #set of failed episodes
        failed_eps = set(failed_eps)

        #create images for time steps in all failed eps
        if failed_episode:
            with open(csv_file, 'r') as file:

                reader = csv.reader(file)
                
                for row in reader:
                    if len(row)>1:

                        if row[-2] in failed_eps:
                            print("Creating image for failed actions in episode ", row[-2]," timestep ", row[-1])

                            img = render_observation(row[0],shape)

                            fig, ax = plt.subplots()
                            mpl_img = ax.imshow(img)
                            mpl_img.set_data(img)
                            fig.canvas.draw()
                            plt.savefig(f"{image_folder}image_{row[-2]}_{int(row[-1])}_before.png")

                            img = render_observation(row[1],(10,10))

                            fig, ax = plt.subplots()
                            mpl_img = ax.imshow(img)
                            mpl_img.set_data(img)
                            fig.canvas.draw()
                            plt.savefig(f"{image_folder}image_{row[-2]}_{int(row[-1])}_next.png")

            





#csv_file : provide input csv file path from which images has to be generated from observations
#image_folder : output folder path for generated images
#failed_action : boolean to create images for time step for a failed action
#agent_only: boolean which will create time step images for failed action of agents only
#failed_episode : boolean that will create images for all time steps of a failed episode

# create_images(csv_file="D:/Profactor/", image_folder = 'D:/Profactor/test_img/',
# failed_action= True, agent_only= True, failed_episode=False)

create_episode(csv_file="D:/Profactor/phase_shift_1", image_folder = 'D:/Profactor/test_images/',eps='8')
