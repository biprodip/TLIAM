import pickle
import numpy as np
from loader import *
from utils import *
import tensorflow as tf


class EnvArgs:
  def __init__(self):
    self.MODEL_NAME = 'model_tliam_'            
    self.FILE_NAME =  'mc_outcomes_tliam.pkl'   
    self.BAYESIAN_FLAG = 1                     
    self.BINS = 10                              
    self.EPOCH_TS = 100
    self.AVG_OPT = 'macro'
    self.SAVE_RES = True


if __name__ == "__main__":
    args = EnvArgs()

    y_test_int, mean_test_mc_logits, mean_test_mc_probs, test_mc_probs, le_name_mapping = load_var(args)

    T = tf.Variable(initial_value=1.39, trainable=True, dtype=tf.float32)

    get_performance(y_test_int, mean_test_mc_probs, le_name_mapping, args)

    produce_cal_u1(args, mean_test_mc_probs, mean_test_mc_logits, y_test_int, T)

    produce_cal_u2(test_mc_probs, y_test_int, T)