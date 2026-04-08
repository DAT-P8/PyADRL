from ray.tune import Trainable


class IterativeTraining(Trainable):
    def setup(sef, test_string):
        print(test_string)

    def step(self):
        return self.algo.train()
