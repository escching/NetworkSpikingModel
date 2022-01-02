import numpy as np
import SpikingNeuronModel as m

np.random.seed(0)
loadFromPrev = False
totTime = 7500
timeStep = 0.125
plotStep = round(500/timeStep)
network = 'networkC.txt'

myModel = m.SpikingNeuronModel(loadFromPrev=loadFromPrev)
myModel.initNetwork(network)
myModel.initDynamicalParams()
myModel.initDynamics(totIter=round(totTime/timeStep), timeStep=timeStep, plotStep=plotStep)
myModel.runDynamics()
myModel.saveDynamicsAndPlot()
