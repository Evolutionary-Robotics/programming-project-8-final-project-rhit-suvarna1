########################################################################################
# Agent of Simulation
#
# Agent consists of a Braitenberg Machine with 2 sensors and 2 motors
# Methods include: move(), sense(), update(), distanceBetweenTwoPoints()
#
# Jack Cooperman, Anuj Suvarna
# November 2024
########################################################################################

import numpy as np
import matplotlib as ml
import math


class Agent():
    def __init__(self, start=(0,0)):
        self.x, self.y = start
        self.x = 0
        self.y = 0
        #makes the angle smaller(more samples basically)
        self.beta = 0.1
        self.alpha = 1
        self.RM = 0
        self.LM = 0
        self.leftSensor = 0
        self.rightSensor = 0
        self.radius = 1
        self.direction = 0
        self.velocity = 0
        #Sensor position
        self.sensOffset = np.pi/4
        self.leftSensLoc_x = self.radius * np.cos(self.direction - self.sensOffset)
        self.leftSensLoc_y = self.radius * np.sin(self.direction - self.sensOffset)
        self.rightSensLoc_x = self.radius * np.cos(self.sensOffset + self.direction)
        self.rightSensLoc_y = self.radius * np.sin(self.sensOffset + self.direction)

    def distanceBetweenTwoPoints(self):
        self.distFormula = np.sqrt((self.x**2) + (self.y**2))
        return self.distFormula
    
    def sense(self,flag):
        max_distance = math.sqrt(100**2 + 100**2)
        left_dist = np.sqrt((self.leftSensLoc_y - flag.ypos)**2 + (self.leftSensLoc_x - flag.xpos)**2)
        right_dist = np.sqrt((self.rightSensLoc_y - flag.ypos)**2 + (self.rightSensLoc_x - flag.xpos)**2)
        self.leftSensor = 1 - (left_dist/max_distance)
        self.rightSensor = 1 - (right_dist/max_distance)
        destinationReached = flag.destinationReached(self.x, self.y)
        return destinationReached

    def move(self, nn, nn_inputs):
        motorVals = nn.forward(nn_inputs)
        self.LM, self.RM = motorVals[0]
        self.update()
        
    def update(self):
        self.velocity = self.alpha * (self.LM + self.RM)
        self.direction += self.beta * (self.RM - self.LM)
        self.x += self.velocity * np.cos(self.direction)  
        self.y += self.velocity * np.sin(self.direction)
        self.leftSensLoc_x = self.x + self.radius * np.cos(self.direction - self.sensOffset)
        self.leftSensLoc_y = self.y + self.radius * np.sin(self.direction - self.sensOffset)
        self.rightSensLoc_x = self.x + self.radius * np.cos(self.direction + self.sensOffset)
        self.rightSensLoc_y = self.y + self.radius * np.sin(self.direction + self.sensOffset)

    