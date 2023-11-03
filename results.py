import pickle
import numpy as np
from loader import *
from utils import *
import tensorflow as tf
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default='mc_outcomes_tliam.pkl')
    parser.add_argument('--bayesian_flag', type=int, default=1)

    args = parser.parse_args()
    
    y_test_int, mean_test_mc_logits, mean_test_mc_probs, test_mc_probs, le_name_mapping = load_var(args)

    T = tf.Variable(initial_value=1.39, trainable=True, dtype=tf.float32)

    get_performance(y_test_int, mean_test_mc_probs, le_name_mapping, args)

    produce_cal_u1(args, mean_test_mc_probs, mean_test_mc_logits, y_test_int, T)

    produce_cal_u2(test_mc_probs, y_test_int, T)