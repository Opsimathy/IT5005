
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib as mpl
mpl.rcParams['animation.writer'] = 'ffmpeg'


def animation_policy_with_utilities(mdp, all_U, all_policy, pause_time=1):
    #This function overlay the utiltiies and policies on cells of GridMDP for each iteration in value iteration algorithm
    #mdp is the mdp object
    # all_U is the list of list; each element in the list is a utility vector
    # all_policy is the list of list; each element in the list is a policy vector


    fig, ax = plt.subplots(figsize=(5, 6))
    def update(frame):
        ax.clear()
        U = all_U[frame]
        policy = all_policy[frame]
        iteration = frame
        # Plot the grid
        for j in range(1, mdp.rows + 1):
            for i in range(1, mdp.cols + 1):
                if (i, j) == mdp.block:
                    ax.add_patch(patches.Rectangle((i - 1, mdp.rows - j), 1, 1, edgecolor='black', facecolor='black'))
                elif (i, j) in mdp.terminal_states:
                    reward = mdp.terminal_states[(i, j)]
                    color = 'red' if reward < 0 else 'green'
                    ax.add_patch(patches.Rectangle((i - 1, mdp.rows - j), 1, 1, edgecolor='black', facecolor=color))
                    ax.text(i - 0.5, mdp.rows - j + 0.5, f'{reward:.1f}', horizontalalignment='center', verticalalignment='center')
                else:
                    ax.add_patch(patches.Rectangle((i - 1, mdp.rows - j), 1, 1, edgecolor='black', facecolor='none'))
                    if (i, j) in U:
                        ax.text(i - 0.5, mdp.rows - j + 0.5, f'{U[(i, j)]:.4f}', horizontalalignment='center', verticalalignment='center')
                    # Overlay the policy
                    if (i, j) in policy:
                        action = policy[(i, j)]
                        if action == 'U':
                            dx, dy = (0, -0.3)
                        elif action == 'D':
                            dx, dy = (0, 0.3)
                        elif action == 'L':
                            dx, dy = (-0.3, 0)
                        elif action == 'R':
                            dx, dy = (0.3, 0)
                        ax.arrow(i - 0.5, mdp.rows - j + 0.5, dx, dy, head_width=0.2, head_length=0.1, fc='blue', ec='blue')

        # Set the limits and display the plot
        ax.set_xlim(0, mdp.cols)
        ax.set_ylim(0, mdp.rows)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Invert y axis to correct the top-bottom inversion
        plt.title(f'Grid MDP Values and Policy:  Iteration {iteration}')
        plt.xlabel('Columns')
        plt.ylabel('Rows')

    ani = FuncAnimation(fig, update, frames=len(all_U), interval=pause_time * 1000, repeat=False)

    # Display the animation inline in Jupyter Notebook

    display(HTML(ani.to_jshtml()))

    return ani 



def plot_policy_with_utilities(mdp, U, policy):
#This function overlays the policy and utilities of each state for the GridMDP
#mdp is the mdp vectpr
# U is the utility vector
# policy is the policy vector
    fig, ax = plt.subplots(figsize=(7, 8))
    # Plot the grid
    for j in range(1, mdp.rows + 1):
        for i in range(1, mdp.cols + 1):
            if (i, j) == mdp.block:
                ax.add_patch(patches.Rectangle((i - 1, mdp.rows - j), 1, 1, edgecolor='black', facecolor='black'))
            elif (i, j) in mdp.terminal_states:
                reward = mdp.terminal_states[(i, j)]
                color = 'red' if reward < 0 else 'green'
                ax.add_patch(patches.Rectangle((i - 1, mdp.rows - j), 1, 1, edgecolor='black', facecolor=color))
                ax.text(i - 0.5, mdp.rows - j + 0.5, f'{U[(i, j)]:.1f}', horizontalalignment='center', verticalalignment='center')
            else:
                ax.add_patch(patches.Rectangle((i - 1, mdp.rows - j), 1, 1, edgecolor='black', facecolor='none'))
                if (i, j) in U:
                    ax.text(i - 0.5, mdp.rows - j + 0.5, f'{U[(i, j)]:.2f}', horizontalalignment='center', verticalalignment='center')
                # Overlay the policy
                if (i, j) in policy:
                    action = policy[(i, j)]
                    if action == 'U':
                        dx, dy = (0, -0.3)
                    elif action == 'D':
                        dx, dy = (0, 0.3)
                    elif action == 'L':
                        dx, dy = (-0.3, 0)
                    elif action == 'R':
                        dx, dy = (0.3, 0)
                    ax.arrow(i - 0.5, mdp.rows - j + 0.5, dx, dy, head_width=0.2, head_length=0.1, fc='blue', ec='blue')

    # Set the limits and display the plot
    ax.set_xlim(0, mdp.cols)
    ax.set_ylim(0, mdp.rows)
    ax.set_aspect('equal')
    ax.invert_yaxis()  
    plt.title(f'Grid MDP Values and Policy with reward ={mdp.default_reward}')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()