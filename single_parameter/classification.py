"""
An example for single parameter classification methods
"""
import common.statistics as stats
from random import random
from math import log, pi, sqrt


#first generate sample data
# x = salary between (1000 - 5000)
# r = car type (1-5)


def discriminant(class_variance, sample_mean, prior, x):
    return (-0.5 * log(2*pi)) - log(sqrt(class_variance))-(((x-sample_mean)**2 )/(2.0*class_variance)) + log(prior)


def get_random_input():
    return random() * 4000.0 + 1000.0


def generate_sample(sample_size):
    x = [0.0] * sample_size
    r = [0] * sample_size

    for i in xrange(sample_size):
        x[i] = get_random_input()
        #not so random!
        r[i] = int((x[i]-1000.0) / 800) + 1

    return x, r


n = 1000
inputs, outputs = generate_sample(n)

priors = [0.0] * 5

sample_set = zip(inputs, outputs)

for i in xrange(1, 6):
    class_count = 0
    for output in outputs:
        if output == i:
            class_count += 1

    priors[i-1] = float(class_count) / n

means = [0.0] * 5
variances = [0.0] * 5

for i in xrange(1, 6):
    sub_set = [input for input, output in sample_set if output == i]
    means[i-1] = stats.sample_mean(sub_set)
    variances[i-1] = stats.sample_variance(sub_set, means[i-1])

sufficient_stats = zip(means, variances)

#suppose it's a gaussian distribution
#and we have new random inputs

for i in xrange(10):
    x = get_random_input()

    discriminants = [0.0] * 5
    max_d = -100000
    max_c = 0
    for j in xrange(5):
        d = discriminant(variances[j], means[j], priors[j], x)
        discriminants[j] = d
        if d > max_d:
            max_d = d
            max_c = j + 1

    print 'For input :', x
    print 'Discriminants'
    print discriminants
    print 'So the most likely class is %(max_c)d' % {'max_c': max_c}
    print '*****'


