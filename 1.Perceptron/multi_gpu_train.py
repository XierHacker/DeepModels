import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
import perceptron
from utility import preprocessing


MAX_EPOCH=10
BATCH_SIZE=20
LEARNING_RATE=0.001
MODEL_SAVING_PATH="./saved_models/model.ckpt"
N_GPU=2

