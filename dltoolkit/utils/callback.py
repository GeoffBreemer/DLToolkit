"""Keras BaseLogger callback producing figures after each epoch"""
from .visual import plot_history
from keras.callbacks import BaseLogger
import json, os


class TrainingMonitor(BaseLogger):
    def __init__(self, fig_path, json_path=None, start_epoch=0):
        super(TrainingMonitor, self).__init__()
        self.fig_path = fig_path
        self.json_path = json_path
        self.start_epoch = start_epoch
        self.hist = {}

    def on_train_begin(self, logs={}):
        self.hist = {}

        if self.json_path is not None:
            if os.path.exists(self.json_path):
                self.hist = json.loads(open(self.json_path).read())

                if self.start_epoch > 0:
                    for h in self.hist.keys():
                        self.hist[h] = self.hist[h][:self.start_epoch]

    def on_epoch_end(self, epoch, logs={}):
        for (key, value) in logs.items():
            l = self.hist.get(key, [])
            l.append(value)
            self.hist[key] = l

        # Save the history value to a JSON file
        if self.json_path is not None:
            f = open(self.json_path, "w")
            f.write(json.dumps(self.hist))
            f.close()

        # Create and save the accuracy/loss plot
        if len(self.hist["loss"]) > 1:
            plot_history(self.hist, len(self.hist["loss"]), False, self.fig_path)
