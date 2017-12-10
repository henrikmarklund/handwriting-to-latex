import distance
import numpy as np


# CHECK
def exact_match(labels_batches, predictions_batches):
  """Compute exact match"""

  match = 0.0
  count = 0.0
  num_batches = len(labels_batches)
  


  for i in range(num_batches):
    labels_batch = labels_batches[i]
    predictions_batch = predictions_batches[i]

    for idx in range(len(labels_batch)):

      if np.all(labels_batch[idx] == predictions_batch[idx]):
        match += 1
      count += 1
  return 100 * match / count


def token_accuracy(labels_batches, predictions_batches):
  """Compute accuracy on per word basis."""

  total_acc, total_count = 0., 0.

  num_batches = len(labels_batches)

  for i in range(num_batches):
    labels_batch = labels_batches[i]
    predictions_batch = predictions_batches[i]

    for idx, target_sentence in enumerate(labels_batch):
      
      prediction = predictions_batch[idx]

      match = 0.0

      for pos in range(min(len(target_sentence), len(prediction))):
        label = target_sentence[pos]
        pred = prediction[pos]
        if label == pred:
          match += 1
      total_acc += 100 * match / max(len(target_sentence), len(prediction))
      total_count += 1
  return total_acc / total_count


#https://pypi.python.org/pypi/Distance/

# Takes a numpy array as input
def lev_dist(labels_batches, predictions_batches):
    
  avg_distance = 0
  count = 0.0
  
  num_batches = len(labels_batches)

  for i in range(num_batches):
    labels_batch = labels_batches[i]
    predictions_batch = predictions_batches[i]
    batch_size = len(labels_batch)

    for idx in range(batch_size):
      label = labels_batch[idx] #.to_list()
      prediction = predictions_batch[idx] #.to_list()
      lev_distance = distance.levenshtein(label, prediction)

      avg_distance = avg_distance + lev_distance
      count += 1

  avg_distance = float(avg_distance) / count

  return avg_distance



def get_metrics(labels_batches, predictions_batches):

  exact_match_avg = exact_match(labels_batches, predictions_batches)
  token_accuracy_avg = token_accuracy(labels_batches, predictions_batches)

  edit_distance_avg = lev_dist(labels_batches, predictions_batches)

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