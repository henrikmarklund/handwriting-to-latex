import distance
import numpy as np
#Exact_match and token_accuracy is adapted from the evaluation file
# in the tutorial on Neural Machine Translation: https://github.com/tensorflow/nmt

def exact_match(labels, predictions, max_token_check=None):
    #"""Compute exact match"""
    match = 0.0
    count = 0.0
    if max_token_check == None:
        max_token_check = 1000
    for idx in range(len(labels)):
        if np.all(labels[idx][:max_token_check] == predictions[idx][:max_token_check]):
            match += 1

        count += 1
    return 100 * match / count


def token_accuracy(labels, predictions, max_token_check=None):
    #"""Compute accuracy on per word basis."""
    total_acc, total_count = 0., 0.
    if max_token_check==None:
        m_length = 1000
    else:
        m_length = max_token_check

    for idx, target_sentence in enumerate(labels):
        prediction = predictions[idx]

        match = 0.0

        total_count += 1 
        for pos in range(min(len(target_sentence), len(prediction), m_length)):
            label = target_sentence[pos]
            pred = prediction[pos]
            if label == pred:
                match += 1
        
             
        if max_token_check==None:
            total_acc += 100 * match / max(len(target_sentence), len(prediction))
        else:
            total_acc += 100 * match / min(len(target_sentence), len(prediction), m_length)
            
    return total_acc / total_count


def lev_dist(labels, predictions, max_token_check=None):
    
    # In our report we report this distance * 10. (to get distance per 10 tokens for easier interpretation)
    avg_distance = 0
    count = 0.0
  
    for idx in range(len(labels)):

        if max_token_check is not None:
            lev_distance = distance.levenshtein(labels[idx][:max_token_check], predictions[idx][:max_token_check])
            lev_distance = lev_distance / (min(len(labels[idx]), max_token_check))
        else:
            lev_distance = distance.levenshtein(labels[idx], predictions[idx])
            lev_distance = lev_distance / (len(labels[idx]))

        avg_distance = avg_distance + lev_distance
        count += 1

    avg_distance = float(avg_distance) / count
    return avg_distance


def get_metrics(labels, predictions, max_token_check=None):
    exact_match_avg = exact_match(labels, predictions, max_token_check)
    token_accuracy_avg = token_accuracy(labels, predictions, max_token_check)
    edit_distance_avg = lev_dist(labels, predictions, max_token_check)
    edit_distance_avg = edit_distance_avg * 10
    return {'Exact Match: ': exact_match_avg, 'Token accuracy': token_accuracy_avg, 'Edit distance (per 10 tokens)': edit_distance_avg }

