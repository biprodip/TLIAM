import pickle
import numpy as np

def load_var(args):
    with open(args.FILE_NAME,'rb') as f:  # Python 3: open(..., 'rb')
        _, _, test_mc_logits, test_mc_probs, y_test_oh, _, _, le_name_mapping = pickle.load(f)
        #val_mc_logits, val_mc_probs, test_mc_logits, test_mc_probs, y_test_oh, y_val_oh, M, le_name_mapping

    y_test_int = y_test_oh.argmax(axis = 1) #label_true
    #y_val_int = y_val_oh.argmax(axis = 1)

    #get mean  logit and softmaxes
    if args.BAYESIAN_FLAG:
      mean_test_mc_logits = (np.array(test_mc_logits)).mean(axis=0)            
      mean_test_mc_probs = np.array(test_mc_probs).mean(axis=0)                

    #   mean_val_mc_logits = (np.array(val_mc_logits)).mean(axis=0)              
    #   mean_val_mc_probs = np.array(val_mc_probs).mean(axis=0)   

    return y_test_int, mean_test_mc_logits, mean_test_mc_probs, test_mc_probs, le_name_mapping 