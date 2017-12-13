import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from matplotlib import pyplot as plt
import cv2
import datetime
import json
import pdb
from operator import itemgetter
from random import shuffle
from tensor2tensor.layers.common_attention import add_timing_signal_nd

### Outline


## CONFIG:
hparams = {}
hparams['num_epochs'] = 16
hparams['max_token_length'] = 70
hparams['mini_batch_size'] = 16
hparams['max_train_num_samples'] = 16
hparams['max_val_num_samples'] = 16
hparams['use_attention'] = True
hparams['use_encoding_average_as_initial_state'] = False
hparams['num_units'] = 512  # LSTM number of units
hparams['OVERFIT_TO_SMALL_SAMPLE'] = False

# Learning rate config
hparams['warm_up_rate'] = 0.00001
hparams['num_epochs_warm_up'] = 3
hparams['base_learning_rate'] = 0.0004
hparams['num_epochs_constant_lrate'] = 3
hparams['num_decay_epochs'] = 30
hparams['target_rate'] = 0.00001
calculate_val_loss = True

RESTORE_FROM_CHECKPOINT = False
# whether or not to create the 'dataset' structure freshly from the image files or from a cached pickle
LOAD_FRESHLY = True

## FLOYDHUB CONFIG
#data = '../data/'
data = '../data/'
output = '../output/'

#todo change these three back before cloud run
ON_FLOYDHUB = False
if (ON_FLOYDHUB):
    data = '/data/'
    output = '/output/'

buckets_dict = {(40, 160): 0,
                (40, 200): 1,
                (40, 240): 2,
                (40, 280): 3,
                (40, 320): 4,
                (40, 360): 5,
                (50, 120): 6,
                (50, 200): 7,
                (50, 240): 8,
                (50, 280): 9,
                (50, 320): 10,
                (50, 360): 11,
                (50, 400): 12,
                (60, 360): 13,
                (100, 360): 14,
                (100, 500): 15,
                (160, 400): 16,
                (200, 500): 17,
                (800, 800): 18}

BIG_BUCKETS = [17, 18]


def get_max_shape(data_batch):
    max_height = 0
    max_width = 0

    for sample in data_batch:
        image = sample[0]

        if image.shape[0] > max_height:
            max_height = image.shape[0]

        if image.shape[1] > max_width:
            max_width = image.shape[1]

    return (max_height, max_width)


def pad_images(data_batch):
    new_data_batch = data_batch
    target_shape = get_max_shape(data_batch)
    new_height = target_shape[0]
    new_width = target_shape[1]

    for idx, sample in enumerate(data_batch):
        padded_image = np.ones((new_height, new_width)) * 255

        image = sample[0]  # A sample consist of an image (0), a target text (1), and a sequence length (2)

        h = image.shape[0]
        w = image.shape[1]

        padded_image[:h, :w] = image

        new_data_batch[idx][0] = padded_image

    return new_data_batch


def sort_key(bucket):
    if len(bucket) == 0:
        return 0
    else:
        if len(bucket[0]):
            return bucket[0][0].shape[1] * 1000 + bucket[0][0].shape[0]
        else:
            return 0


def load_raw_data(dataset_name, mini_batch_size, max_token_length=400, max_num_samples=5000):
    buckets = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

    dataset = []

    if dataset_name == "small":
        image_folder = data + 'tin/tiny/'
        formula_file_path = data + 'tin/tiny.formulas.norm.txt'
    elif dataset_name == "test":
        image_folder = data + 'images_test/'
        formula_file_path = data + 'test.formulas.norm.txt'
    elif dataset_name == "train":
        image_folder = data + 'images_train/'
        formula_file_path = data + 'train.formulas.norm.txt'
    elif dataset_name == "val":
        image_folder = data + 'images_val/'
        formula_file_path = data + 'val.formulas.norm.txt'
    elif dataset_name == "digital_numbers":
        image_folder = 'datasets/digital_numbers/images/'
        formula_file_path = "datasets/digital_numbers/number_sequences.txt"

    in_counter = 0
    examples_counter = 0
    with open(formula_file_path, "r") as myfile:

        for idx, token_sequence in enumerate(myfile):
            examples_counter += 1
            # Check token size:
            token_sequence = token_sequence.rstrip('\n')
            tokens = token_sequence.split()

            file_name = str(idx) + '.png'
            image = cv2.imread(image_folder + file_name, cv2.IMREAD_GRAYSCALE)

            if image is None:
                # what does this even mean?
                print("Id was none: ", idx)
                continue
            image = image.astype(np.uint8)
            if len(tokens) <= max_token_length:

                token_sequence = '**start** ' + token_sequence
                # is **end** already there?
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Grey scale

                seq_length = len(token_sequence.split())

                relevant_bucket_id = buckets_dict[image.shape]

                if relevant_bucket_id in BIG_BUCKETS:
                    continue

                buckets[relevant_bucket_id].append([image, token_sequence, seq_length])

                if len(buckets[buckets_dict[image.shape]]) == mini_batch_size:
                    data_batch = np.array(buckets[buckets_dict[image.shape]])
                    dataset.append(data_batch)
                    buckets[buckets_dict[image.shape]] = []

                in_counter += 1

            if in_counter == max_num_samples:
                break

        # put what's left in the buckets into batches (padding will be needed)

        counter = 0

        data_batch = []

        for idx, bucket in enumerate(buckets):

            for j, sample in enumerate(bucket):
                data_batch.append(sample)
                if len(data_batch) == mini_batch_size:
                    padded_data_batch = pad_images(data_batch)
                    padded_data_batch = np.array(padded_data_batch)
                    dataset.append(padded_data_batch)
                    data_batch = []

        if (len(data_batch) == mini_batch_size):
            # for some reason the algorithm generates empty data batches sometimes
            padded_data_batch = pad_images(data_batch)
            padded_data_batch = np.array(padded_data_batch)
            dataset.append(padded_data_batch)

    for k in range(len(dataset)):
        assert (len(dataset[k]) == mini_batch_size)
    return dataset


# dataset is list of all batches containing (image, target_text, sequence_length)
# we split that up into three lists
# Adam code review note: this seems verbose and I don't quite get it
def split_dataset(dataset):
    encoder_input_data_batches = []
    target_texts_batches = []
    sequence_lengths_batches = []

    for batch in range(len(dataset)):
        temp = dataset[batch]
        if (len(temp) == 0):
            bob = None
        image_batch = temp[:, 0]
        image_batch = image_batch.tolist()
        image_batch = np.array(image_batch)
        # Add one dimension so that the conv net can take it (it expects four dimensions)
        image_batch = np.reshape(image_batch, (image_batch.shape[0], image_batch.shape[1], image_batch.shape[2], 1))
        image_batch = image_batch.astype('uint8')

        encoder_input_data_batches.append(image_batch)

        target_text = temp[:, 1]
        target_texts_batches.append(target_text)

        decoder_length = temp[:, 2]
        decoder_length = np.array(decoder_length, dtype=np.uint16)
        sequence_lengths_batches.append(decoder_length)

    # Make sure we have equal number of batches
    assert (len(encoder_input_data_batches) == len(dataset))
    assert (len(target_texts_batches) == len(dataset))
    assert (len(sequence_lengths_batches) == len(dataset))

    return encoder_input_data_batches, target_texts_batches, sequence_lengths_batches


def get_vocabulary(dataset):
    print("HI MOM! ", data)
    if dataset == "small":
        vocab = [line for line in open(data + 'tin/tiny_vocab.txt')]
    elif dataset == "test":
        vocab = [line for line in open(data + 'vocab.txt')]
    elif dataset == "train":
        vocab = [line for line in open(data + 'vocab.txt')]

    vocab = [x.strip('\n') for x in vocab]
    return vocab


def create_output_int_sequences(target_texts_batches, sequence_lengths_batches, target_token_index):
    decoder_input_data_batches = []
    decoder_target_data_batches = []

    for idx, target_texts_batch in enumerate(target_texts_batches):

        # get max dec seq length for that batch
        max_decoder_seq_length = max(sequence_lengths_batches[idx])

        batch_size = len(target_texts_batch)

        decoder_input_data = np.zeros(
            (batch_size, max_decoder_seq_length),
            dtype='uint16')
        decoder_target_data = np.zeros(
            (batch_size, max_decoder_seq_length),
            dtype='uint16')

        num_other = 0

        for i, target_text in enumerate(target_texts_batch):
            for t, token in enumerate(target_text.split()):

                if token in target_token_index:
                    # decoder_target_data is ahead of decoder_input_data by one timestep

                    decoder_input_data[i, t] = target_token_index[token]

                    if t > 0:
                        # decoder_target_data will be ahead by one timestep
                        # and will not include the start character.
                        decoder_target_data[i, t - 1] = target_token_index[token]

                else:
                    print("Token %s in %d not in V" % (token, idx))
                    num_other = num_other + 1
                    decoder_input_data[i, t] = target_token_index['**unknown**']

                    if t > 0:
                        # decoder_target_data will be ahead by one timestep
                        # and will not include the start character.

                        decoder_target_data[i, t - 1] = target_token_index['**unknown**']

            decoder_target_data[i, len(target_text.split()) - 1] = target_token_index['**end**']

        decoder_input_data_batches.append(decoder_input_data)
        decoder_target_data_batches.append(decoder_target_data)

    return decoder_input_data_batches, decoder_target_data_batches


def dump_data_by_size(name, sorted_data):
    current_indices = None
    current_shape = None
    for idx, img_batch in enumerate(sorted_data[0]):
        if current_indices is None:
            # first iteration, do not save anything
            current_indices = [idx]
            current_shape = img_batch[0].shape
            continue

        if img_batch[0].shape == current_shape:
            current_indices.append(idx)
        else:
            current_data_set = [
                itemgetter(*current_indices)(sorted_data[0]),
                itemgetter(*current_indices)(sorted_data[1]),
                itemgetter(*current_indices)(sorted_data[2]),
                itemgetter(*current_indices)(sorted_data[3]),
                itemgetter(*current_indices)(sorted_data[4])
            ]
            dump_data_set(current_data_set, name + str(current_shape))
            current_indices = [idx]
            current_shape = img_batch[0].shape
            # set the current mini batch equal to the current_mini batch


def load_data(dataset_name, mini_batch_size, max_token_length, max_num_samples, target_token_index):
    dataset = load_raw_data(dataset_name, mini_batch_size, max_token_length=max_token_length,
                            max_num_samples=max_num_samples)

    #dataset.sort(key=sort_key)
    encoder_input_data_batches, target_texts_batches, sequence_lengths_batches = split_dataset(dataset)
    decoder_input_data_batches, decoder_target_data_batches = create_output_int_sequences(target_texts_batches,
                                                                                          sequence_lengths_batches,
                                                                                          target_token_index)
    return encoder_input_data_batches, target_texts_batches, sequence_lengths_batches, decoder_input_data_batches, decoder_target_data_batches


# Learning rate_schedule

# Epoch 0 - 2: Warmup with a lower learning rate: (1e-4)
# Epoch 3 - 6: (5e-4)
# Epoch 7 - 20: Exponentially decaying from (5e-4) to (1e-5)


def get_learning_rate(global_step, num_train_batches):
    epoch = int(float(global_step) / num_train_batches)

    if hparams['OVERFIT_TO_SMALL_SAMPLE'] == True:

        if epoch < 20:
            # Warm up
            lr_rate = 0.0001
        elif epoch < 40:
            lr_rate = 0.0005
        elif epoch < 500:
            # Over 10 epochs decay learning rate from 0.0005 to 0.00001
            decay_rate = 0.00001 / 0.0005
            decay_steps = hparams['num_epochs']
            lr_rate = 0.0005 * decay_rate ** (float((global_step - num_train_batches * 6)) / decay_steps)
        else:
            # after 16 epochs of decay, set a new fixed rate
            lr_rate = 0.00001
    else:

        warm_up_rate = hparams['warm_up_rate']
        num_epochs_warm_up = hparams['num_epochs_warm_up']

        base_learning_rate = hparams['base_learning_rate']
        num_epochs_constant_lrate = hparams['num_epochs_constant_lrate']

        num_decay_epochs = hparams['num_decay_epochs']
        target_rate = hparams['target_rate']

        if epoch < num_epochs_warm_up:
            # Warm up
            lr_rate = warm_up_rate
        elif epoch < num_epochs_warm_up + num_epochs_constant_lrate:
            lr_rate = base_learning_rate
        elif epoch < num_epochs_warm_up + num_epochs_constant_lrate + num_decay_epochs:
            # Over 10 epochs decay learning rate from 0.0005 to 0.00001

            decay_rate = target_rate / base_learning_rate
            decay_steps = num_train_batches * num_decay_epochs
            lr_rate = base_learning_rate * decay_rate ** (
                    float((
                                  global_step - num_train_batches * num_epochs_warm_up + num_epochs_constant_lrate)) / decay_steps)
        else:

            lr_rate = target_rate

    return lr_rate


def get_validation_loss(num_val_batches,
                        img, val_encoder_input_data_batches,
                        decoder_lengths, val_sequence_lengths_batches,
                        decoder_inputs, val_decoder_input_data_batches,
                        decoder_outputs, val_decoder_target_data_batches,
                        train_loss, sess):
    # num_val_batches = len(val_sequence_lengths_batches)
    val_loss = 0
    for i in range(num_val_batches):
        input_data = {img: val_encoder_input_data_batches[i],
                      decoder_lengths: val_sequence_lengths_batches[i],
                      decoder_inputs: val_decoder_input_data_batches[i],
                      decoder_outputs: val_decoder_target_data_batches[i],
                      }

        output_tensors = [train_loss]
        loss = sess.run(output_tensors,
                        feed_dict=input_data)

        print(loss)
        val_loss = val_loss + loss[0]

    val_loss = val_loss / len(val_decoder_input_data_batches[i])
    return val_loss


def get_id_for_bucket(img_batches):
    shapes_already_found = []
    batch_ids = []
    for idx, batch in enumerate(img_batches):
        image = img_batches[0][0]
        shape = np.squeeze(image).shape

        for j in range(18, -1, -1):

            if buckets_dict[shape] == j:
                # print("Found batch with shape: ", buckets_dict[shape])
                # print("Batch id: ", idx)
                shapes_already_found.append(shape)

        batch_ids.append(idx)

    return batch_ids


def dump_data_set(set, name):
    filename = output + 'pickles/' + name + '.pkl'
    if not os.path.exists(output + 'pickles'):
        os.makedirs(output + 'pickles')
    f = open(filename, 'wb+')
    pickle.dump(set, f)
    print('dumped: ' + name)
    f.close()


def load_data_pickle(name):
    print('loading: ', name)
    filename = data + name + '.pkl'
    f = open(filename, 'rb')
    data_set = pickle.load(f)
    f.close()
    return data_set


def get_data_somehow(name, fresh, _mini_batch_size, _max_token_length, _max_train_num_samples, _target_token_index):
    if (fresh):
        print('fresh data coming up')
        _set = load_data(name, _mini_batch_size, _max_token_length, _max_train_num_samples, _target_token_index)
        dump_data_set(_set, name)
    else:
        print('yesterday\'s data half price')
        _set = load_data_pickle(name)

    return _set


def get_loss(img, encoder_input_data_batches,
             decoder_lengths, sequence_lengths_batches,
             decoder_inputs, decoder_input_data_batches,
             decoder_outputs, decoder_target_data_batches,
             train_loss, sess):
    num_batches = len(sequence_lengths_batches)
    avg_loss = 0
    for i in range(num_batches):
        input_data = {img: encoder_input_data_batches[i],
                      decoder_lengths: sequence_lengths_batches[i],
                      decoder_inputs: decoder_input_data_batches[i],
                      decoder_outputs: decoder_target_data_batches[i],
                      }

        output_tensors = [train_loss]
        loss = sess.run(output_tensors,
                        feed_dict=input_data)

        avg_loss = avg_loss + loss[0]

    avg_loss = avg_loss / num_batches
    return avg_loss


def std_ocr_convnet(img):
    img = tf.cast(img, tf.float32) / 255.

    out = tf.layers.conv2d(img, 64, 3, 1, "SAME", activation=tf.nn.relu)
    out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

    out = tf.layers.conv2d(out, 128, 3, 1, "SAME", activation=tf.nn.relu)
    out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

    out = tf.layers.conv2d(out, 256, 3, 1, "SAME", activation=tf.nn.relu)

    out = tf.layers.conv2d(out, 256, 3, 1, "SAME", activation=tf.nn.relu)
    out = tf.layers.max_pooling2d(out, (2, 1), (2, 1), "SAME")

    out = tf.layers.conv2d(out, 512, 3, 1, "SAME", activation=tf.nn.relu)
    out = tf.layers.max_pooling2d(out, (1, 2), (1, 2), "SAME")

    # encoder representation, shape = (batch size, height', width', 512)
    out = tf.layers.conv2d(out, 512, 3, 1, "VALID", activation=tf.nn.relu)

    return out


def create_graph(token_vocab_size, num_units, use_attention, use_encoding_average_as_initial_state, training=True):
    # Encoder
    # Adam Added skip connection to one of Genthails's encoder implementations (from paper)
    img = tf.placeholder(tf.uint8, [None, None, None, 1], name='img')
    batch_size = tf.shape(img)[0]

    cast_img = tf.cast(img, tf.float32) / 255.

    out = tf.layers.conv2d(cast_img, 64, 3, 1, "SAME", activation=tf.nn.relu, name='conv1')
    out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

    out = tf.layers.conv2d(out, 128, 3, 1, "SAME", activation=tf.nn.relu, name='conv2')
    out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

    out = tf.layers.conv2d(out, 128, 3, 1, "SAME", activation=tf.nn.relu, name='conv3')

    out = tf.layers.conv2d(out, 256, 3, 1, "SAME", activation=tf.nn.relu, name='conv4')
    out = tf.layers.max_pooling2d(out, (2, 1), (2, 1), "SAME")

    out = tf.layers.conv2d(out, 512, 3, 1, "SAME", activation=tf.nn.relu, name='conv5')
    out = tf.layers.max_pooling2d(out, (1, 2), (1, 2), "SAME")

    # encoder representation, shape = (batch size, height', width', 512)
    out = tf.layers.conv2d(out, 512, 3, 1, "VALID", activation=tf.nn.relu)

    ## Out is now a H'*W' encoding of the image

    ## We want to turn this into a sequence of vectors: (e1, e2 ... en)
    # H= out.shape[1]
    # W= out.shape[2]
    # C= out.shape[3]

    H = tf.shape(out)[1]
    W = tf.shape(out)[2]

    out = add_timing_signal_nd(out)
    seq = tf.reshape(tensor=out, shape=[-1, H * W, 512])

    # First state of the decoder consists of two vectors, the hidden state (h0) and the memory (c0).
    # Usually the hidden state refers to [h0, c0]. So a little bit of overloading of hidden state (I think)
    # This is how Genthail implements it

    if use_encoding_average_as_initial_state:
        img_mean = tf.reduce_mean(seq, axis=1)

        # img_mean = tf.layers.batch_normalization(img_mean)

        W = tf.get_variable("W", shape=[512, num_units])
        b = tf.get_variable("b", shape=[num_units])
        h0 = tf.tanh(tf.matmul(img_mean, W) + b)

        W_ = tf.get_variable("W_", shape=[512, num_units])
        b_ = tf.get_variable("b_", shape=[num_units])
        c0 = tf.tanh(tf.matmul(img_mean, W_) + b_)

        encoder_state = tf.contrib.rnn.LSTMStateTuple(c0, h0)

    # attention_states: [batch_size, max_time, num_units]
    attention_states = seq

    attention_depth = num_units

    # Create an attention mechanism
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        attention_depth, attention_states, scale=True)  # Can try scale = False

    # Decoder: from seq2seq tutorial
    embedding_size = 80  # In Genthail's paper he says he has 80 embeddings which I believe corresponds to embedding_size

    decoder_inputs = tf.placeholder(tf.uint16, [None, None],
                                    name='decoder_inputs')  # Supposed to be a sequence of numbers corresponding to the different tokens in the sentence
    decoder_inputs = tf.cast(decoder_inputs, tf.int32)
    # Embedding of target tokens

    # Embedding matrix
    embedding_decoder = tf.get_variable(
        "embedding_encoder", [token_vocab_size, embedding_size],
        tf.float32)  # tf.float32 was default in the NMT tutorial

    # Look up embedding:
    #   decoder_inputs: [max_time, batch_size]
    #   decoder_emb_inp: [max_time, batch_size, embedding_size]
    decoder_emb_inp = tf.nn.embedding_lookup(
        embedding_decoder, decoder_inputs)

    # Build RNN cell
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

    # Using this instead compared to NMT tutorial so we can initialize with orthogonal intializer (like Genthail)
    # decoder_cell = tf.nn.rnn_cell.LSTMCell(
    # num_units,
    # initializer=tf.orthogonal_initializer,
    # )

    if use_attention:
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism,
            attention_layer_size=512)

        ## Set initial state of decoder to zero (possible to use previous state)

        if use_encoding_average_as_initial_state:
            decoder_initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)
        else:
            decoder_initial_state = decoder_cell.zero_state(batch_size, tf.float32)

    else:
        decoder_initial_state = encoder_state

    decoder_lengths = tf.placeholder(tf.uint16, [None])
    decoder_lengths = tf.cast(decoder_lengths, tf.int32)

    # Helper
    helper = tf.contrib.seq2seq.TrainingHelper(
        decoder_emb_inp, decoder_lengths, time_major=False)

    # Projection layer
    projection_layer = layers_core.Dense(token_vocab_size, use_bias=False,
                                         name="output_projection")  # Said layers_core before

    # Decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, helper, decoder_initial_state,
        output_layer=projection_layer)

    # Dynamic decoding
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      output_time_major=False)  ## Understand parameter Impute finished
    logits = outputs.rnn_output

    global_step = tf.Variable(0, trainable=False)  ## IMPORTANT

    # target_weights = tf.placeholder(tf.int8, [None, None], name='target_weights')
    # target_weights = tf.cast(target_weights, tf.float32)

    # Supposed to be a sequence of numbers corresponding to the different tokens in the sentence
    decoder_outputs = tf.placeholder(tf.uint16, [None, None], name='decoder_outputs')
    decoder_outputs = tf.cast(decoder_outputs, tf.int32)
    learning_rate = tf.placeholder(tf.float32, shape=[])

    # Loss function

    # HYPERPARAMETER: Should we divide by sequence length on each
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=decoder_outputs, logits=logits)

    # Create the target_weights (the masking)
    max_seq_length = tf.shape(decoder_outputs)[1]
    target_weights = tf.sequence_mask(decoder_lengths, max_seq_length, dtype=logits.dtype)

    train_loss = tf.reduce_sum(crossent * target_weights) / tf.cast(batch_size, tf.float32)

    tf.summary.scalar('loss', train_loss)

    # Calculate and clip gradients
    params = tf.trainable_variables()
    gradients = tf.gradients(train_loss, params)

    max_gradient_norm = 3  # Usually a number between 1 and 5. Set to 5 in the NMT.

    clipped_gradients, global_norm = tf.clip_by_global_norm(
        gradients, max_gradient_norm)

    tf.summary.scalar('global_norm', global_norm)

    # Optimization
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_step = optimizer.apply_gradients(
        zip(clipped_gradients, params), global_step=global_step)

    param_names = [v.name for v in params]

    gradient_names = [g.name for g in gradients]

    gradient_norms = [tf.norm(gradient) for gradient in gradients]

    grads = list(zip(gradients, params))

    for grad, var in grads:
        tf.summary.histogram(var.name + '/gradient', grad)

    for param in params:
        to_summary = tf.summary.histogram(param.name + '/weight', param)

        # config=tf.ConfigProto(log_device_placement=True) logs whether it runs on the gpus
        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    merged = tf.summary.merge_all()

    graph = {'merged': merged,
              'update_step': update_step,
              'train_loss': train_loss,
              'optimizer': optimizer,
              'global_norm': global_norm,
              'gradient_norms': gradient_norms,
              'global_step': global_step,
              'img': img,
              'decoder_lengths': decoder_lengths,
              'decoder_inputs': decoder_inputs,
              'decoder_outputs': decoder_outputs,
              'learning_rate': learning_rate,
              'embedding_decoder': embedding_decoder,
              'decoder_cell': decoder_cell,
              'decoder_initial_state': decoder_initial_state,
              'projection_layer': projection_layer
        }
    return graph


def inference_tensor(target_token_index,
                     inference_batch_size,
                     embedding_decoder,
                     decoder_cell,
                     decoder_initial_state,
                     projection_layer,
                     maximum_iterations=hparams['max_token_length']):
    """
    :param target_token_index:
    :param batch_for_inference:
    :param embedding_decoder:
    :param decoder_cell:
    :param decoder_initial_state:
    :param projection_layer:
    :param maximum_iterations:
    :return: RETURNS A TENSOR THE CALLER USES FOR INFERENCE ON 1 BATCH
    """
    tgt_sos_id = target_token_index['**start**']  # 1
    tgt_eos_id = target_token_index['**end**']  # 0

    # Helper
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder,
                                                                tf.fill([inference_batch_size], tgt_sos_id), tgt_eos_id)

    # Decoder
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, inference_helper, decoder_initial_state,
        output_layer=projection_layer)
    # Dynamic decoding
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        inference_decoder, maximum_iterations=maximum_iterations)
    translations = outputs.sample_id
    logits = outputs.rnn_output
    return translations, logits


def predict_batch(sess,
                  batch,
                  target_token_index,
                  embedding_decoder,
                  decoder_cell,
                  decoder_initial_state,
                  projection_layer,
                  img,
                  maximum_iterations=hparams['max_token_length']):
    # for b in batches:
    batch_len = batch.shape[0]
    translation_t, logits_t = inference_tensor(target_token_index,
                                               batch_len,
                                               embedding_decoder,
                                               decoder_cell,
                                               decoder_initial_state,
                                               projection_layer)
    translation, logits = sess.run([translation_t, logits_t], feed_dict={img: batch})
    return translation, logits


def initialize_variables(sess, saver, restore, path):
    if restore:
        print('restoring')
        saver.restore(sess, path)
    else:
        print('reinitializing')
        sess.run(tf.global_variables_initializer())


def create_hparams_log():
    file = open(output + "hparams.txt", "w")
    file.write(json.dumps(hparams, indent=4))

    file.close()


def create_metric_output_files():
    file = open(output + "metrics.txt", "w")

    file.write("Train loss" + "\t" + "Val loss" + "\t" + "Learning rate" + "\t" + "Global grad norm" + "\n")

    file.close()


def main():
    create_hparams_log()

    num_epochs = hparams['num_epochs']
    max_token_length = hparams['max_token_length']
    mini_batch_size = hparams['mini_batch_size']
    max_train_num_samples = hparams['max_train_num_samples']
    max_val_num_samples = hparams['max_val_num_samples']
    use_attention = hparams['use_attention']
    use_encoding_average_as_initial_state = hparams['use_encoding_average_as_initial_state']
    num_units = hparams['num_units']  # LSTM number of units

    # Create the vocabulary
    target_tokens = ["**end**", "**start**", "**unknown**"]

    target_tokens.extend(get_vocabulary("train"))
    token_vocab_size = len(target_tokens)
    target_token_index = dict(
        [(token, i) for i, token in enumerate(target_tokens)])

    reverse_target_token_index = dict(
        (i, char) for char, i in target_token_index.items())
    print("\n ======================= Loading Data =======================")
    # new cell
    print('load freshly: ', LOAD_FRESHLY)
    train_dataset = get_data_somehow('train', LOAD_FRESHLY, mini_batch_size, max_token_length,
                                     max_train_num_samples,
                                     target_token_index)
    val_dataset = get_data_somehow('val', LOAD_FRESHLY, mini_batch_size, max_token_length,
                                   max_val_num_samples,
                                   target_token_index)

    #dump_data_by_size('train', train_dataset)
    #dump_data_by_size('val', val_dataset)
    train_encoder_input_data_batches = train_dataset[0]
    train_target_texts_batches = train_dataset[1]
    train_sequence_lengths_batches = train_dataset[2]
    train_decoder_input_data_batches = train_dataset[3]
    train_decoder_target_data_batches = train_dataset[4]
    val_encoder_input_data_batches = val_dataset[0]
    val_target_texts_batches = val_dataset[1]
    val_sequence_lengths_batches = val_dataset[2]
    val_decoder_input_data_batches = val_dataset[3]
    val_decoder_target_data_batches = val_dataset[4]
    print("\n ======================= Data Loaded =======================")

    num_train_batches = len(train_target_texts_batches)
    num_val_batches = len(val_target_texts_batches)
    num_train_samples = (num_train_batches - 1) * mini_batch_size + train_target_texts_batches[-1].shape[0]
    num_val_samples = (num_val_batches - 1) * mini_batch_size + val_target_texts_batches[-1].shape[0]

    # new cell
    print("Num train batches: ", num_train_batches)
    print("Num val batches: ", num_val_batches)

    print("Num train samples: ", num_train_samples)
    print("Num val samples: ", num_val_samples)

    g = create_graph(token_vocab_size, num_units, use_attention, use_encoding_average_as_initial_state)
    merged = g['merged']
    update_step = g['update_step']
    train_loss = g['train_loss']
    optimizer = g['optimizer']
    global_norm = g['global_norm']
    gradient_norms = g['gradient_norms']
    global_step = g['global_step']
    img = g['img']
    decoder_lengths = g['decoder_lengths']
    decoder_inputs = g['decoder_inputs']
    decoder_outputs = g['decoder_outputs']
    learning_rate = g['learning_rate']

    sess = tf.Session()
    tf_saver = tf.train.Saver(name='Henrik', allow_empty=False)
    initialize_variables(sess, saver=tf_saver, restore=RESTORE_FROM_CHECKPOINT, path="/Users/adamjensen/project-environments/handwriting-to-latex-env/output/checkpoints/model_9.ckpt")

    train_writer = tf.summary.FileWriter(output + 'summaries/train/', sess.graph)

    print("Num batches: ", len(
        train_sequence_lengths_batches))  # (Note: they are not necessarily equal size towards the end (this will fix later))

    train_decoder_target_data_batches

    print(train_sequence_lengths_batches[0])

    print(train_encoder_input_data_batches[0].shape)

    # train_decoder_input_data_batches[0][0] = np.array(train_decoder_input_data_batches[0][0])
    train_decoder_target_data_batches[0][0] = np.array(train_decoder_target_data_batches[0][0])

    # print(train_decoder_input_data_batches[0].shape)

    print(train_sequence_lengths_batches[0].shape)
    print(train_decoder_target_data_batches[0].shape)
    print(train_decoder_input_data_batches[0].shape)

    learning_rates = []

    for step in range(num_train_batches * 20):
        learning_rates.append(get_learning_rate(step, num_train_batches))


    total_parameters = 0
    # Get total number of parameters

    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters

    print("Total number of parameters: ", total_parameters)

    print("Num batches: ", num_train_batches)

    glob_step = sess.run(
        global_step)  # Get what global step we are at in training already (so that the learning_rate is set correct)
    # _list = get_id_for_bucket(train_encoder_input_data_batches)

    create_metric_output_files()

    if hparams['OVERFIT_TO_SMALL_SAMPLE'] == True:
        #only train on the first 2 batches
        print('only training on the first 1 batches')
        num_train_batches = 1

    for epoch in range(num_epochs + 1):
        print("planning: %d epochs.  Starting epoch: %d" % (num_epochs, epoch))

        # Train on data sorted by image_width for the first 15 epochs, and random orderings after that
        #if epoch > 10:
        #    print('shuffled')
        #    idxs = [x for x in range(len(train_encoder_input_data_batches))]
        #    shuffle(idxs)

        #    train_encoder_input_data_batches = itemgetter(*idxs)(train_encoder_input_data_batches)
        #    train_target_texts_batches = itemgetter(*idxs)(train_target_texts_batches)
        #    train_sequence_lengths_batches = itemgetter(*idxs)(train_sequence_lengths_batches)
        #    train_decoder_input_data_batches = itemgetter(*idxs)(train_decoder_input_data_batches)
        #    train_decoder_target_data_batches = itemgetter(*idxs)(train_decoder_target_data_batches)

        for i in range(num_train_batches):
            # Calculate running time for batch
            start_time = datetime.datetime.now()

            # Calculate the right learning rate for this step.
            lrate = get_learning_rate(glob_step, num_train_batches)
            images = train_encoder_input_data_batches[i]
            input_data = {img: images,
                          decoder_lengths: train_sequence_lengths_batches[i],
                          decoder_inputs: train_decoder_input_data_batches[i],
                          decoder_outputs: train_decoder_target_data_batches[i],
                          learning_rate: lrate
                          }
            # Write to tensorboard
            #pdb.set_trace()
            if glob_step % 200 == 0:

                output_tensors = [merged, update_step, train_loss, optimizer._lr, global_norm, gradient_norms,
                                  global_step]
                summary, _, loss, lr_rate, global_grad_norm, grad_norms, glob_step = sess.run(output_tensors,
                                                                                              feed_dict=input_data)
                train_writer.add_summary(summary, glob_step)
                print("global step: %d loss: %f" %(glob_step, loss))
            else:
                output_tensors = [update_step, train_loss, global_norm, global_step, optimizer._lr]
                _, loss, global_grad_norm, glob_step, lr_rate = sess.run(output_tensors,
                                                                         feed_dict=input_data)

                if loss <= 0.01:
                    break

            if i == 0:
                print("global step: %d loss: %f" % (glob_step, loss))
                validation_loss = get_loss(img, val_encoder_input_data_batches,
                                           decoder_lengths, val_sequence_lengths_batches,
                                           decoder_inputs, val_decoder_input_data_batches,
                                           decoder_outputs, val_decoder_target_data_batches,
                                           train_loss, sess)

                training_loss = get_loss(img, train_encoder_input_data_batches,
                                         decoder_lengths, train_sequence_lengths_batches,
                                         decoder_inputs, train_decoder_input_data_batches,
                                         decoder_outputs, train_decoder_target_data_batches,
                                         train_loss, sess)

                file = open(output + "metrics.txt", "a")
                lrate_to_file = ('%s' % ('%.8g' % lrate))
                file.write(str(training_loss) + "\t" + str(validation_loss) + "\t" + lrate_to_file + "\t" + str(
                    global_grad_norm) + "\n")

                file.close()
        # Run the following in terminal to get up tensorboard: tensorboard --logdir=summaries/train

        save_path = tf_saver.save(sess, output + 'checkpoints/model_a_' + str(epoch) + '.ckpt', global_step=glob_step)
        print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    main()
