# DigNet
This project is an attempt to model a neural network that can find an optimal solution to a traveling salesman problem involving a robot agent (e.g. a Mars rover) that can travel throughout a grid world (e.g. Mars) in search of areas of interest to carry out scientific experiments. 

The environment is a grid world that is pre-programmed to contain a distribution of minerals of various concentrations, represented as integers. The robot can move north, south, east, or west in the grid world, and can also perform a 'dig' operation, which harvests 1 mineral if a mineral is available at that grid location. The mineral count at that location then decreases by 1, until there are no minerals left. Each move and dig operation consumes 1 battery unit. 

The objective is to model the neural network such that an optimal 'travel' and 'dig' schedule is learned in order to maximize the number of minerals harvested while minimizing the amount of battery consumed.

The inspiration behind this project is the Mars _Perseverance_ rover, which has a mission to search for signs of ancient life. I imagine such an agent trained with RL could be useful in programming what the rover's daily schedule should be in order to optimize travel time and battery consumption.

I've tried a variety of different approaches to find an optimal solution to this, with the most optimal thus far being found in gym_digger_dqn_keras-rl2.py.
