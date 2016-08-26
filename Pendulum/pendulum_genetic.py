from concurrent.futures import ProcessPoolExecutor

import copy
import gym
import numpy
import random


class GeneticSearcher:
    def __init__(self, pop_size, problem):
        self.problem = problem
        self.pop = [Network.random_network() for i in range(pop_size)]
        self.fitness_cache = {}
        self.best = None
        self.nt = NetTester(problem)
        self.pp = ProcessPoolExecutor(max_workers=4)
        self.ntf = NetworkTesterFactory(problem)
        self.pop_size = pop_size

    def recalculate_fitness(self):
        nets_to_rate = [net for net in self.pop if net not in self.fitness_cache]
        for net, res in self.pp.map(self.ntf.rate_network, nets_to_rate):
            self.fitness_cache[net] = res

    def selection(self):
        population_fitness = [(net, self.fitness_cache[net]) for net in self.pop]
        population_fitness = sorted(population_fitness, reverse=True, key=lambda x: x[1])
        self.best = population_fitness[0]
        return list(map(lambda x: x[0], population_fitness[:int(self.pop_size / 3)]))

    def crossing(self, parents):
        children = []
        while len(children) < self.pop_size / 3:
            parents = random.sample(set(parents), 2)
            children.append(self.problem.crossing(parents[0], parents[1]))

        return children

    def mutation(self, population):
        mutants = []
        while len(mutants) < 0.3 * self.pop_size:
            mutants.append(self.problem.mutate(random.choice(population)))

        return mutants

    def iteration(self):
        self.recalculate_fitness()
        old_survivors = self.selection()
        children = self.crossing(old_survivors)
        mutants = self.mutation(old_survivors)

        new_generation = old_survivors + children + mutants

        while len(new_generation) < self.pop_size:
            new_generation.append(Network.random_network())

        self.pop = new_generation

        return self.best[1]

    def show_best(self):
        self.nt.test(self.best[0], render=True)


class Network:
    def __init__(self, weights):
        self.weights = weights

    def __hash__(self):
        return hash(frozenset(self.weights))

    def __eq__(self, other):
        return self.weights.__eq__(other.weights)

    def weighted_sum(self, observation):
        s = 0.0
        for i in range(3):
            s += self.weights[i] * observation[i]

        return s + self.weights[3]

    def output(self, observation):
        val = self.weighted_sum(observation) / 2
        if val > 2:
            return 2
        elif val < -2:
            return -2

        return val

    def __str__(self):
        return str(self.weights)

    @staticmethod
    def random_network():
        return Network([random.random() * 2 - 1 for i in range(4)])


class NetTester:
    def __init__(self, problem):
        self.problem = problem
        self.env = problem.make_env()

    def test_n_times_and_return_min(self, net, n):
        results = [self.test(net) for _ in range(n)]
        return min(results)

    def test(self, net, render=False):
        observation = self.env.reset()
        res = 0.0

        for t in range(1000):
            if render:
                self.env.render()

            self.problem.scale_observation(self.env, observation)
            action = numpy.array([net.output(observation)])
            observation, reward, done, info = self.env.step(action)

            res += reward

            if done:
                break

        return res


class NetworkTesterFactory:
    def __init__(self, problem):
        self.problem = problem

    def rate_network(self, net):
        nt = NetTester(self.problem)
        return net, nt.test_n_times_and_return_min(net, 10)


class PendulumV0:
    @staticmethod
    def crossing(net1, net2):
        crossing_point = random.randint(1, 3)
        new_weights = []

        for i in range(crossing_point):
            new_weights.append(net1.weights[i])

        for i in range(crossing_point, 4):
            new_weights.append(net2.weights[i])

        return Network(new_weights)

    @staticmethod
    def mutate(net):
        mutations = random.randint(1, 3)
        mutated_genes = random.sample([0, 1, 2, 3], mutations)
        new_weights = copy.copy(net.weights)

        for idx in mutated_genes:
            new_weights[idx] = random.random() * 2 - 1

        return Network(new_weights)

    @staticmethod
    def make_env():
        return gym.make('Pendulum-v0')

    @staticmethod
    def scale_observation(env, observation):
        for i in range(3):
            observation[i] /= env.observation_space.high[i]


def main():
    gs = GeneticSearcher(100, PendulumV0)

    for i in range(20):
        print('generation {}'.format(i))
        best = gs.iteration()
        print('best: {}'.format(best))

        gs.show_best()


if __name__ == '__main__':
    main()