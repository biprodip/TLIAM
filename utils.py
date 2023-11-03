import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix


def get_performance(y_test_tmp, mean_test_mc_probs, le_name_mapping, args):
    if args.bayesian_flag == 1:
        y_hat = mean_test_mc_probs.argmax(axis=1)   # total_test_sample

    e_acc = accuracy_score(y_test_tmp, y_hat)
    e_cm = confusion_matrix(y_test_tmp, y_hat)
    e_pr = precision_score(y_test_tmp, y_hat,average='macro',labels=np.unique(y_hat))
    e_rc = recall_score(y_test_tmp, y_hat, average='macro',labels=np.unique(y_hat))
    e_f1 = f1_score(y_test_tmp, y_hat, average='macro',labels=np.unique(y_hat))
    e_auc = roc_auc_score(y_test_tmp, mean_test_mc_probs, multi_class='ovr')
    
    labels = np.unique(y_hat)
    FalsePositive = []
    for i in range(len(labels)):
        FalsePositive.append(sum(e_cm[:,i]) - e_cm[i,i])
    FP_percentage = FalsePositive[le_name_mapping['Benign']]*100/sum(e_cm[:,le_name_mapping['Benign']])
    
    print("MC-ensemble accuracy: {:.1%}".format(e_acc))
    print("MC-ensemble precision: {:.1%}".format(e_pr))
    print("MC-ensemble recall: {:.1%}".format(e_rc))
    print("MC-ensemble f1: {:.1%}".format(e_f1))
    print("MC-ensemble auc: {:.1%}".format(e_auc))
    print(f'False positive for Benign is : {FP_percentage:0.4f}')


def scale_logits(mean_mc_logits, T):
    ts_mean_mc_logits = tf.math.divide(mean_mc_logits, T)
    ts_logits = tf.convert_to_tensor(ts_mean_mc_logits, dtype=tf.float32, name='scaled_logits')
    ts_mean_mc_probs = tf.nn.softmax(ts_logits)
    return ts_logits, ts_mean_mc_probs


def produce_cal_u1(args, mean_test_mc_probs, mean_mc_logits, y_test_tmp, T):
    if args.bayesian_flag == 1:
        y_hat = mean_test_mc_probs.argmax(axis=1)   

    epsilon = 1e-10 
    _, ts_mean_mc_probs = scale_logits(mean_mc_logits, T)

    ts_test_u1 = -1 * np.sum(ts_mean_mc_probs * np.log(ts_mean_mc_probs + epsilon), axis=1)
    c_normalized_u1 = ts_test_u1 * (1/np.log(ts_mean_mc_probs.shape[1]))
    c_avg_u1 = np.sum(c_normalized_u1)/float(y_hat.shape[0])
    c_corr_u1 = ts_test_u1[y_hat==y_test_tmp]
    c_mis_u1 = ts_test_u1[y_hat!=y_test_tmp]
    print(f'\nAvg calibrated correct classification U1: {np.sum(c_corr_u1)/sum(y_hat==y_test_tmp)} \nStdDev: {np.std(c_corr_u1)}')
    print(f'Avg calibrated mis classification U1: {np.sum(c_mis_u1)/sum(y_hat!=y_test_tmp)},\nStdDev: {np.std(c_mis_u1)}')


def produce_cal_u2(test_mc_probs, y_test_tmp, T):
    N_MC_SAMPLES = np.array(test_mc_probs).shape[0]             
    ts_mc_probs = test_mc_probs/T.numpy() 
    ts_mean_mc_probs = np.array(ts_mc_probs).mean(axis=0)  

    y_hat = ts_mean_mc_probs.argmax(axis=1)

    tmp = (ts_mc_probs-ts_mean_mc_probs)**2   
    tmp_sum = tmp.sum(axis=0)
    c_u2_tmp = tmp_sum/N_MC_SAMPLES     
    c_u2 = np.array([c_u2_tmp[i,y_hat[i]] for i in range(y_hat.shape[0])])  
    c_corr_u2 = c_u2[y_hat==y_test_tmp]
    c_mis_u2 = c_u2[y_hat!=y_test_tmp]
    print(f'\nAvg calibrated correct classification U2: {np.sum(c_corr_u2)/sum(y_hat==y_test_tmp)}  \n StdDev: {np.std(c_corr_u2)}')
    print(f'Avg calibrated  mis classification U2: {np.sum(c_mis_u2)/sum(y_hat!=y_test_tmp)},\n StdDev: {np.std(c_mis_u2)}')