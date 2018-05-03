import numpy as np
import tensorflow as tf
import pandas as pd
from perceptron import Perceptron

'''


'''


# load model
percept = Perceptron()
percept.fit(X=train_frame.values,y=train_labels_frame.values,print_log=False)

# predict
#result = percept.predict(X=test_frame.values)
#print(result)
