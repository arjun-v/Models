import numpy as np

class BackPropogationNetwork:

    @staticmethod
    def sgm(x, derivative=False):
        if not derivative:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            out = BackPropogationNetwork.sgm(x)
            return out * (1.0 - out)

    @staticmethod
    def linear(x, derivative=False):
        if not derivative:
            return x
        else:
            return 1.0

    @staticmethod
    def gaussian(x, derivative=False):
        if not derivative:
            return np.exp(-x**2)
        else:
            return -2*x*np.exp(-x**2)
    
    @staticmethod
    def tanh(x, derivative=False):
        if not derivative:
            return np.tanh(x)
        else:
            return 1.0 - np.tanh(x)**2


    def __init__(self, layerSize, layerFunctions = None):
        self.weights = []
        self.tFuncs = []
        self.layerCount = len(layerSize) - 1
        self.shape = layerSize

        self._layerInput = []
        self._layerOutput = []
        self._previousWeightDelta = []

        if layerFunctions is None:
            lFuncs = []
            for i in range(self.layerCount):
                if i == self.layerCount - 1:
                    lFuncs.append(BackPropogationNetwork.linear)
                else:
                    lFuncs.append(BackPropogationNetwork.sgm)
        else:
            if len(layerSize) != len(layerFunctions):
                raise ValueError("Incompatible list of transfer functions")
            elif layerFunctions[0] is not None:
                raise ValueError("Input layer cannot have a transfer function")
            else:
                lFuncs = layerFunctions[1:]

        self.tFuncs = lFuncs

        for (l1,l2) in zip(layerSize[:-1], layerSize[1:]):
            self.weights.append(np.random.normal(scale=0.2,size=(l2,l1+1)))
            self._previousWeightDelta.append(np.zeros((l2,l1+1)))

        print self.weights

    def run(self, input):
        lnCases = input.shape[0]
        self._layerInput = []
        self._layerOutput = []

        for index in range(self.layerCount):
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1,lnCases])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1,lnCases])]))

            self._layerInput.append(layerInput)
            self._layerOutput.append(self.tFuncs[index](layerInput))

        return self._layerOutput[-1].T


    def trainEpoch(self, input, target, trainingRate = 0.01, momentum = 0.01):
        delta = []
        lnCases = input.shape[0]

        self.run(input)

        for index in reversed(range(self.layerCount)):
            if index == self.layerCount - 1:
                output_delta = self._layerOutput[index] - target.T
                error = np.sum(output_delta**2)
                delta.append(output_delta * self.tFuncs[index](self._layerInput[index], True))
            else:
                delta_pullback = self.weights[index + 1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1,:] * self.tFuncs[index](self._layerInput[index], True))

        for index in range(self.layerCount):
            delta_index = self.layerCount - 1 - index
            if index == 0:
                layerOutput = np.vstack([input.T, np.ones([1,lnCases])])
            else:
                layerOutput = np.vstack([self._layerOutput[index-1], np.ones([1, self._layerOutput[index-1].shape[1]])])

            curWeightDelta = np.sum(layerOutput[None,:,:].transpose(2,0,1) * delta[delta_index][None,:,:].transpose(2,1,0), axis = 0)
            weightDelta = trainingRate * curWeightDelta + momentum * self._previousWeightDelta[index]
            self.weights[index] -= weightDelta
            self._previousWeightDelta[index] = weightDelta

        return error


if __name__ == '__main__':
    input = np.array([[0,0],[1,1],[1,0],[0,1]])
    target = np.array([[0.05],[0.05],[0.95],[0.95]])
    lFuncs = [None, BackPropogationNetwork.gaussian, BackPropogationNetwork.linear]

    bpn = BackPropogationNetwork((2,2,1),lFuncs)
    max = 100000
    lnErr = 1e-5

    for i in range(max+1):
        err = bpn.trainEpoch(input,target)
        if i%2500 == 0:
            print "iteration {0}\tError : {1:0.6f}".format(i,err)
        if err <= lnErr:
            print "Min error reached at {0}".format(i)
            break


    d = np.array([[0,0],[1,1],[1,0],[0,1]])
    output = bpn.run(d)
    for i in range(d.shape[0]):
        print "Input : {0} Output : {1} ".format(d[i],output[i])



