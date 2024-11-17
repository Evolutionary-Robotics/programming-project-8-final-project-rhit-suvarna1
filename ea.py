########################################################################################
# Microbial Genetic Algorithm
#
# Tournament style evolutionary algorithm
# 
# Eduardo Izquierdo
# September 2024
########################################################################################


import numpy as np
import matplotlib.pyplot as plt

class MGA():

    def __init__(self, fitnessfunction, genesize, popsize, recomprob, mutationprob, tournaments):
        self.genesize = genesize
        self.popsize = popsize
        self.recomprob = recomprob
        self.mutationprob = mutationprob
        self.tournaments = tournaments
        self.fitnessfunction = fitnessfunction
        self.pop = np.random.random((popsize,genesize))*2 - 1
        self.fit = self.calculateFitness()
        # stats
        gens = tournaments//popsize      
        self.bestfit = np.zeros(gens)
        self.avgfit = np.zeros(gens)
        self.worstfit = np.zeros(gens)
        self.bestind = np.zeros(gens)

    def calculateFitness(self):
        fits = np.zeros(self.popsize)
        for i in range(self.popsize):
            fits[i] = self.fitnessfunction(self.pop[i])
        return fits

    def run(self):
        # 1 loop for tour
        gen = 0
        for t in range(self.tournaments):
            # 2 pick two to fight (same could be picked -- fix)
            [a,b] = np.random.choice(np.arange(self.popsize),2,replace=False)
            # 3 pick winner
            if self.fit[a] > self.fit[b]:
                winner = a
                loser = b
            else:
                winner = b
                loser = a
            # 4 transfect winner to loser
            for g in range(self.genesize):
                if np.random.random() < self.recomprob: 
                    self.pop[loser][g] = self.pop[winner][g] 
            # 5 mutate loser
            self.pop[loser] += np.random.normal(0,self.mutationprob,self.genesize)
            self.pop[loser] = np.clip(self.pop[loser],-1,1)
            # Update
            self.fit[loser] = self.fitnessfunction(self.pop[loser])
            # 6 Stats 
            if t % self.popsize == 0:
                self.bestfit[gen] = np.max(self.fit)
                self.avgfit[gen] = np.mean(self.fit)
                self.worstfit[gen] = np.min(self.fit)
                self.bestind[gen] = np.argmax(self.fit)                
                gen += 1

    def showFitness(self):
        plt.plot(self.bestfit,label="Best")
        plt.plot(self.avgfit,label="Avg.")
        plt.plot(self.worstfit,label="Worst")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Evolution")
        plt.show()
                
