########################################################################################
# Destinations of our experiment
#
# Parameters of destinations generated in the hybrid_controller_sim.py
#
# Jack Cooperman, Anuj Suvarna
# November 2024
########################################################################################

import numpy as np

DEST_REACHED = 1
DEST_NOT_REACHED = 0

class Destination():
    def __init__(self, xpos, ypos, radius):
        self.xpos = xpos
        self.ypos = ypos
        self.radius = radius
        self.maxPoints = 10

    def destinationReached(self, agentX, agentY):
        self.distFromDest = np.sqrt((self.xpos - agentX)**2 + (self.ypos - agentY)**2)
        if(self.distFromDest < self.radius):
            return DEST_REACHED
        else:
            return DEST_NOT_REACHED

    


