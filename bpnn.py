# Back-Propagation Neural Networks
#!/usr/bin/python
# *- coding: utf-8 -*-

"""Back-propagation neural network for python

Authors: Neil Schemenauer <nas@arctrix.com>, Viator <viator@via-net.org>
License: MIT (http://opensource.org/licenses/MIT)
"""

import math
import random
import cProfile

def profile_it(func):
    """Decorator for cProfile
    """
    cprofile = cProfile.Profile()
    def do_profile(*args, **kargs):
        cprofile.clear()
        returned = cprofile.runcall(func, *args, **kargs)
        cprofile.create_stats()
        cprofile.print_stats(sort = 2)
        return returned
    do_profile.__name__ = func.__name__
    do_profile.__dict__ = func.__dict__
    do_profile.__doc__ = func.__doc__
    return do_profile

class NN:
    """Neural Network

        num_input - number of inputs nodes
        num_hidden - number of hidden layers
        num_output - number of output nodes
        seed - seed for random
        sigmoid - function for sigmoid(x)
        dsigmoid - function for dsigmoid(y)
    """

    def __init__(self, num_input, num_hidden, num_output, seed = 0,
                    sigmoid = math.tanh, dsigmoid = lambda y: 1.0 - y ** 2):
        random.seed(seed)
        rand = random.triangular

        self.sigmoid = sigmoid
        self.dsigmoid = dsigmoid

        # number of input, hidden, and output nodes
        # +1 for bias node
        self.num_input = num_input + 1
        self.num_hidden = num_hidden
        self.num_output = num_output

        # activations for nodes
        self.act_input = [1.0] * self.num_input
        self.act_hidden = [1.0] * self.num_hidden
        self.act_output = [1.0] * self.num_output

        # create weights
        self.weights_input = [[rand(-0.2, 0.2) for h in xrange(self.num_hidden)]\
                                  for i in xrange(self.num_input)]
        self.weights_output = [[rand(-0.2, 0.2) for o in xrange(self.num_output)]\
                                  for h in xrange(self.num_hidden)]

        # last change in weights for momentum
        self.change_input = [ [ 0.0 for h in xrange(self.num_hidden) ] for i in xrange(self.num_input) ]
        self.change_output = [ [ 0.0 for o in xrange(self.num_output) ] for h in xrange(self.num_hidden) ]

    def update(self, inputs):
        """Return output results for inputs nodes and activate neurons
        """
        if len(inputs) != self.num_input-1:
            raise ValueError('wrong number of inputs')

        # input activations
        self.act_input = inputs + self.act_input[-1:]

        # hidden activations
        input_xrange = xrange(self.num_input)
        hidden_xrange = xrange(self.num_hidden)
        output_xrange = xrange(self.num_output)

        for j in hidden_xrange:
            res_sum = 0.0
            for i in input_xrange:
                res_sum += self.act_input[i] * self.weights_input[i][j]
            self.act_hidden[j] = self.sigmoid(res_sum)

        # output activations
        for k in output_xrange:
            res_sum = 0.0
            for j in hidden_xrange:
                res_sum += self.act_hidden[j] * self.weights_output[j][k]
            self.act_output[k] = self.sigmoid(res_sum)

        return self.act_output


    def backPropagate(self, targets, N, M):
        """For train - back propagation method
        """
        if len(targets) != self.num_output:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.num_output
        for k in xrange(self.num_output):
            # * error
            output_deltas[k] = self.dsigmoid(self.act_output[k]) * (targets[k] - self.act_output[k])

        out_xrange = xrange(self.num_output)
        hidden_xrange = xrange(self.num_hidden)

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.num_hidden
        for j in hidden_xrange:
            error = 0.0
            for k in out_xrange:
                error += output_deltas[k] * self.weights_output[j][k]
            hidden_deltas[j] = self.dsigmoid(self.act_hidden[j]) * error

        # update output weights
        for j in hidden_xrange:
            for k in out_xrange:
                change = output_deltas[k] * self.act_hidden[j]
                self.weights_output[j][k] = self.weights_output[j][k] + N * change + M * self.change_output[j][k]
                self.change_output[j][k] = change
                #print N*change, M*self.change_output[j][k]


        # update input weights
        for i in xrange(self.num_input):
            for j in hidden_xrange:
                change = hidden_deltas[j] * self.act_input[i]
                self.weights_input[i][j] = self.weights_input[i][j] + N * change + M * self.change_input[i][j]
                self.change_input[i][j] = change

    def calc_error(self, targets):
        """calculate error
        """
        error = 0.0
        for e, target in enumerate(targets):
            error += 0.5 * (target - self.act_output[e])**2
        return error

    def weights(self):
        print 'Input weights:'
        for i in xrange(self.num_input):
            print i, self.weights_input[i]
        print 'Output weights:'
        for j in xrange(self.num_hidden):
            print j, self.weights_output[j]

    def train(self, patterns, iterations = 1000, learning_rate = 0.7,\
              momentum_factor = 0.7, degrade_count = 7, degrade_factor = 0.9, yield_count = 10):
        """train of NN, iterator

            patterns - list of input and output patterns:
                [
                    line1,
                    line2,
                    ...
                    line_N
                ]
                where line:
                    [
                        [input1, input2, ..., input_N],
                        [output1, outpu2, ..., output_N]
                    ]
                    where inputs and outputs - float or int numbers

            iterations - number of iterations for train
            factors for train:
                learning_rate
                momentum_factor
            degrade_count - count of degrade for learning_rate and momentum_factor
                it's use degrade_factor: learning_rate * degrade_factor
                    example: [1] 0.7 * 0.9 = 0.63
                             [2] 0.63 * 0.9 = 0.567
                                 => etc
            yield_count - mod for yield of progress

            Yield yield_iteration, error, learning_rate, momentum_factor
        """

        iter_yield = iterations / yield_count
        iter_degrade = iterations / degrade_count
        for i in xrange(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                self.backPropagate(targets, learning_rate, momentum_factor)
                error += self.calc_error(targets)
            if iter_yield > 0 and i % iter_yield == 0 and i >= iter_yield:
                yield i / iter_yield, error, learning_rate, momentum_factor
            if iter_degrade > 0 and i % iter_degrade == 0 and i >= iter_degrade:
                learning_rate *= degrade_factor
                momentum_factor *= degrade_factor
        yield yield_count, error, learning_rate, momentum_factor

@profile_it
def demo():
    pat = [
        [
            [0, 0], [0]
        ],
        [
            [0, 1], [1]
        ],
        [
            [1, 0], [1]
        ],
        [
            [1, 1], [0]
        ],
    ]

    for pattern in ('{0} == {1}'.format(' '.join([str(f) for f in item[0]]),\
                    ' '.join([str(f) for f in item[1]])) for item in pat):
        print pattern

    n = NN(2, 4, 1)
    n.weights()
    # train it with some patterns
    for progress in ('iter: {3}, error: {0:-f}, learning rate: {1:-f}, momentum factor: {2:-f}'.\
               format(er, lr, mf, it) for it, er, lr, mf in n.train(pat, 10000)):
        print progress

    # test
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

    print '\n'.join([
        '{0} -> {1}'.format(
            ', '.join(['{0:-f}'.format(v) for v in inp]),
            ', '.join(['{0:-f}'.format(v) for v in n.update(inp)])) for inp in inputs
    ])


if __name__ == '__main__':
    demo()
