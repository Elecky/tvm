import numpy as np
import multiprocessing
import logging
import threading
import lightgbm as lgb
import time

from .tuner import Tuner
from .sa_model_optimizer import random_walk
from .model_based_tuner import knob2point, point2knob
from ..measure import MeasureInput, create_measure_batch
from ..env import GLOBAL_SCOPE

logger = logging.getLogger('autotvm')

class LGBTuner(Tuner):
    def __init__(self, task, early_stop = 1e9):
        super(LGBTuner, self).__init__(task)
        self.space = task.config_space
        self.dims = [len(x) for x in self.space.space_map.values()]
        self.visited = set([])
        self.parallel_size = multiprocessing.cpu_count()
        self.preiter = 0
        self.points = []
        self.scores = []
        self.params = {
            'patch_size': 6,
            'learning_rate': 0.1,
            'max_depth': 3,
            'num_leaves': 2,
            'reg_alpha': 0,
            'reg_lambda': 0
        }
        self.process_points = []
        self.new_points = []
        self.process_score = [0 for _ in range(self.params['patch_size'])]
        self.warn = [0 for _ in range(self.params['patch_size'])]
        self.f = open('score.csv', 'a+')
        self.f.write('this test start at: ' + str(time.asctime( time.localtime(time.time()))) + '\n')

    def isLegalPoint(self, point):
        return (knob2point(point, self.dims) >= 0 and knob2point(point, self.dims) <= len(self.space))

    def PointInit(self, scale):
        tmppoints = []
        for _ in range(scale):
            tmp = point2knob(np.random.randint(len(self.space)), self.dims)
            while (not self.isLegalPoint(tmp)) or (knob2point(tmp, self.dims) in self.visited):
                tmp = point2knob(np.random.randint(len(self.space)), self.dims)
            tmppoints.append(tmp)
        return tmppoints

    def update(self, inputs, results):
        if self.preiter != self.best_iter:
            self.preiter = self.best_iter
            self.f.write(str(self.best_iter) + ',' + str(self.best_flops) + '\n')
        
        i=0
        for inp, res in zip(inputs, results):
            if res.error_no == 0:
                y = inp.task.flop / np.mean(res.costs)
                if y/1e9 > self.process_score[i]:
                    self.process_score[i] = y/1e9
                    self.process_points[i] = self.new_points[i]
                    self.warn[i] = 0
                else:
                    self.warn[i] +=1
                    if self.warn[i] >= 50:
                        self.process_points[i] = self.PointInit(1)[0]
                        self.warn[i] = 0
                        self.process_score[i] = 0
                self.scores.append(y/1e9)
            else:
                self.scores.append(0.0)
            i+=1
        
    def SortElem(self, elem):
        return elem[1]

    def next_batch(self, batch_size):
        configs = []
        
        # init points when visited is empty
        if len(self.visited) == 0:
            init_points = self.PointInit(self.params['patch_size'])
            for point in init_points:
                self.points.append(point)
                configs.append(self.space.get(knob2point(point, self.dims)))
                self.visited.add(knob2point(point, self.dims))
                self.process_points.append(point)
                self.new_points.append(point)
            return configs

        model = lgb.LGBMRegressor(num_threads=self.parallel_size, num_leaves=self.params['num_leaves'], max_depth=self.params['max_depth'], learning_rate=self.params['learning_rate'], reg_alpha=self.params['reg_alpha'], reg_lambda=self.params['reg_lambda'], objective='regression')

        x_train = np.array(self.points)
        y_train = np.array(self.scores)
        model.fit(x_train, y_train)

        # test_points = self.PointInit(600)
        # test_scores = model.predict(test_points)
        # sort_test_scores = [(test_points[i], test_scores[i]) for i in range(len(test_points))]
        # sort_test_scores.sort(key = self.SortElem)
        # print(sort_test_scores[0][1])
        # for k in range(6):
        #     configs.append(self.space.get(knob2point(sort_test_scores[len(test_points)-1-k][0], self.dims)))
        #     self.points.append(sort_test_scores[len(test_points)-1-k][0])
        #     self.visited.add(knob2point(sort_test_scores[len(test_points)-1-k][0], self.dims))

        self.new_points = []
        for k in range(len(self.process_points)):
            test_points = []
            hc = [-1 for _ in range(len(self.dims))]
            while True:
                isEnd = True
                for i in range(len(self.dims)):
                    if hc[i] != 1:
                        hc[i] += 1
                        isEnd = False
                        break
                    elif i != len(self.dims) - 1:
                        hc[i] = -1
                    else:
                        break
                if isEnd:
                    break
                tmp = [a + b for a, b in zip(hc,self.process_points[k])]
                if knob2point(tmp, self.dims) not in self.visited:
                    test_points.append(tmp)   

            test_scores = model.predict(test_points)
            sort_test_scores = [(test_points[i], test_scores[i], i) for i in range(len(test_points))]
            sort_test_scores.sort(key = self.SortElem)
            configs.append(self.space.get(knob2point(sort_test_scores[len(test_points)-1][0], self.dims)))
            self.points.append(sort_test_scores[len(test_points)-1][0])
            self.new_points.append(sort_test_scores[len(test_points)-1][0])
            self.visited.add(knob2point(sort_test_scores[len(test_points)-1][0], self.dims))
        
        return configs

    def has_next(self):
        # return self.trial_pt < self.n_trial
        return True