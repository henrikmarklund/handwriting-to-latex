import distance
import numpy as np


# CHECK
def exact_match(labels, predictions, max_token_check=None):
  """Compute exact match"""

  match = 0.0
  count = 0.0
  if max_token_check == None:
    max_token_check = 1000

  for idx in range(len(labels)):

    if np.all(labels[idx, :max_token_check] == predictions[idx,:max_token_check]):
      match += 1
    count += 1
  return 100 * match / count


def token_accuracy(labels, predictions, max_token_check=None):
  """Compute accuracy on per word basis."""

  total_acc, total_count = 0., 0.

  if max_token_check==None:
    max_token_check = 1000

  for idx, target_sentence in enumerate(labels):
    
    prediction = predictions[idx]

    match = 0.0


    for pos in range(min(len(target_sentence), len(prediction), max_token_check)):
      label = target_sentence[pos]
      pred = prediction[pos]
      if label == pred:
        match += 1
    total_acc += 100 * match / max(len(target_sentence), len(prediction))
    total_count += 1
  return total_acc / total_count


#https://pypi.python.org/pypi/Distance/

# Takes a numpy array as input
def lev_dist(labels, predictions, max_token_check=None):
    
  avg_distance = 0
  count = 0.0
  

  for idx in range(len(labels)):


    if max_token_check is not None:
      lev_distance = distance.levenshtein(labels[idx,:max_token_check], predictions[idx,:max_token_check])
    else:
      lev_distance = distance.levenshtein(labels[idx], predictions[idx])

    avg_distance = avg_distance + lev_distance
    count += 1



  avg_distance = float(avg_distance) / count

  return avg_distance



def get_metrics(labels, predictions, max_token_check=None):



  exact_match_avg = exact_match(labels, predictions, max_token_check)
  token_accuracy_avg = token_accuracy(labels, predictions, max_token_check)

  edit_distance_avg = lev_dist(labels, predictions, max_token_check)

  return exact_match_avg, token_accuracy_avg, edit_distance_avg







# TEST
#predictions_batch1 = [[3,4,5], [6,1,6], [1,4], [51,5,6], [6,61,616,4] ]
#labels_batch1 = [[3,2,5], [6,1,6], [1,4], [6,61,616,4], [51,5,6]]

#predictions_batch2 = [[3,4,5]]
#labels_batch2 = [[3,2,5]]

#predictions_batches = [predictions_batch1, predictions_batch2]
#labels_batches = [labels_batch1, labels_batch2]


#print(get_metrics(labels_batches, predictions_batches))
#print(token_accuracy(labels_batches, predictions_batches))

#print(lev_dist(labels_batches, predictions_batches))