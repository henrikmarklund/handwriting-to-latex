import numpy as np
import pickle
import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from matplotlib import pyplot as plt
import cv2
import datetime

### Outline

# 1. Encoder
# 2. Decoder
# 3. Optimization and training

# 4. Inference

## FLOYDHUB CONFIG
data = '../data/'
output = '../output/'


ON_FLOYDHUB = False
if (ON_FLOYDHUB):
    data = '/data/'
    output = '/output/'

## CONFIG:
num_epochs = 10
max_token_length = 50
mini_batch_size = 16
max_train_num_samples = 100000
max_val_num_samples = 1000
use_attention = True # I have not tried without attention so not sure if it breaks
use_encoding_average_as_initial_state = True  #Only relevant when use_attention is True.
num_units = 512 # LSTM number of units
calculate_val_loss = False

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



# 2. Try runnin it on the GPU for 1 hour (I suggest having a cap of token length 50, but increase the num samples to a lot) (All in the Config up top). If you want to play with different learning rates, I've not created a config for this yet. Rather there is a function called: get_learning_rate that handles it all.
# 3. My sense is that it is probably way to slow but by running it on a GPU we can get a sense of how slow it is, and how much faster it needs to become. Right now I've got no sense, I but I've started tracking it for a batch.

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
                #what does this even mean?
                print("Id was none: ", idx)
                continue
            image = image.astype(np.uint8)
            if len(tokens) <= max_token_length:

                token_sequence = '**start** ' + token_sequence
                # is **end** already there?
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Grey scale


                seq_length = len(token_sequence.split())

                relevant_bucket_id = buckets_dict[image.shape]

                if relevant_bucket_id == 18 or relevant_bucket_id == 17:
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
        if (len(data_batch) != 0):
            # for some reason the algorithm generates empty data batches sometimes
            padded_data_batch = pad_images(data_batch)
            padded_data_batch = np.array(padded_data_batch)
            dataset.append(padded_data_batch)


    return dataset


# dataset is list of all batches containing (image, target_text, sequence_length)
# we split that up into three lists
# Adam code review note: this seems verbose and I don't quite get it
def split_dataset(dataset):
    encoder_input_data_batches = []
    target_texts_batches = []
    sequence_lengths_batches = []

    for batch in range(len(dataset) ):
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


def load_data(dataset_name, mini_batch_size, max_token_length, max_num_samples, target_token_index):
    dataset = load_raw_data(dataset_name, mini_batch_size, max_token_length=max_token_length,
                            max_num_samples=max_num_samples)

    for k in range(len(dataset) - 1):
        assert (len(dataset[k]) == mini_batch_size)
    encoder_input_data_batches, target_texts_batches, sequence_lengths_batches = split_dataset(dataset)
    decoder_input_data_batches, decoder_target_data_batches = create_output_int_sequences(target_texts_batches,
                                                                                          sequence_lengths_batches, target_token_index)
    return encoder_input_data_batches, target_texts_batches, sequence_lengths_batches, decoder_input_data_batches, decoder_target_data_batches


# Learning rate_schedule

# Epoch 0 - 2: Warmup with a lower learning rate: (1e-4)
# Epoch 3 - 6: (5e-4)
# Epoch 7 - 20: Exponentially decaying from (5e-4) to (1e-5)


def get_learning_rate(global_step, num_train_batches):
    epoch = int(float(global_step) / num_train_batches)

    base_learning_rate = 0.0005

    if epoch < 3:
        # Warm up
        lr_rate = 0.0001
    elif epoch < 6:
        lr_rate = 0.0005
    elif epoch < 16:
        # Over 10 epochs decay learning rate from 0.0005 to 0.00001

        decay_rate = 0.00001 / 0.0005
        decay_steps = num_train_batches * 10
        lr_rate = base_learning_rate * decay_rate ** (float((global_step - num_train_batches * 6)) / decay_steps)
    else:
        # after 16 epochs of decay, set a new fixed rate
        lr_rate = 0.00001
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
                #print("Found batch with shape: ", buckets_dict[shape])
                #print("Batch id: ", idx)
                shapes_already_found.append(shape)

        batch_ids.append(idx)

    return batch_ids


def dump_data_set(set, name):
    filename = output + 'pickles/' + name + '.pkl'
    if not os.path.exists(output + 'pickles'):
        os.makedirs(output + 'pickles')
    f = open(filename, 'wb+')
    pickle.dump(set, f)
    print('dumped')
    f.close()


def load_data_pickle(name):
    print('loading: ', name)
    filename = output + 'pickles/' + name + '.pkl'
    f = open(filename, 'rb')
    data_set = pickle.load(f)
    f.close()
    return data_set

def get_data_somehow(name, fresh, _mini_batch_size, _max_token_length, _max_train_num_samples, _target_token_index):  
    if (fresh):
        _set = load_data(name, _mini_batch_size, _max_token_length, _max_train_num_samples, _target_token_index)        
        dump_data_set(_set, name)
    else:
        _set = load_data_pickle(name)

    return _set


def main():
    # Create the vocabulary
    token_vocabulary = ["**end**", "**start**", "**unknown**"]

    token_vocabulary.extend(get_vocabulary("train"))

    target_tokens = token_vocabulary  # TODO: Refactor this. Currently duplicate naming

    token_vocab_size = len(target_tokens)
    # todo: document what was lifted from tutorials and what we wrote ourselves
    target_token_index = dict(
        [(token, i) for i, token in enumerate(target_tokens)])

    reverse_target_token_index = dict(
        (i, char) for char, i in target_token_index.items())
    print("\n ======================= Loading Data =======================")
    #new cell

    train_dataset = get_data_somehow('train', False, mini_batch_size, max_token_length, max_train_num_samples, target_token_index)
    val_dataset = get_data_somehow('val', True, mini_batch_size, max_token_length, max_train_num_samples, target_token_index)
    
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

    #new cell
    print("Num train batches: ", num_train_batches)
    print("Num val batches: ", num_val_batches)

    print("Num train samples: ", num_train_samples)
    print("Num val samples: ", num_val_samples)

    #Encoder
    #One of Genthails's encoder implementations (from paper)
    img = tf.placeholder(tf.uint8, [None, None, None, 1], name='img')

    img = tf.cast(img, tf.float32) / 255

    batch_size = tf.shape(img)[0]

    # Conv + max pooling
    out = tf.layers.conv2d(img, 64, 3, 1, "SAME", activation=tf.nn.relu)
    # Conv + max pooling
    out = tf.layers.conv2d(out, 128, 3, 1, "SAME", activation=tf.nn.relu)

    out = tf.layers.conv2d(out, 256, 3, 1, "SAME", activation=tf.nn.relu)  # regular conv -> id
    out = tf.layers.batch_normalization(out)

    out = tf.layers.conv2d(out, 256, 3, 1, "SAME", activation=tf.nn.relu)  # regular conv -> id
    out = tf.layers.max_pooling2d(out, (2, 1), (2, 1), "SAME")

    out = tf.layers.conv2d(out, 512, 3, 1, "SAME", activation=tf.nn.relu)  # regular conv -> id
    out = tf.layers.max_pooling2d(out, (1, 2), (1, 2), "SAME")

    # Conv valid
    out = tf.layers.conv2d(out, 512, 3, 1, "VALID", activation=tf.nn.relu, name="last_conv_layer")  # conv
    out = tf.layers.batch_normalization(out)

    ## Out is now a H'*W' encoding of the image

    ## We want to turn this into a sequence of vectors: (e1, e2 ... en)
    # H= out.shape[1]
    # W= out.shape[2]
    # C= out.shape[3]

    H = tf.shape(out)[1]
    W = tf.shape(out)[2]

    # out = add_timing_signal_nd(out)
    seq = tf.reshape(tensor=out, shape=[-1, H * W, 512])

    # TODO: Add positional encodings

    # First state of the decoder consists of two vectors, the hidden state (h0) and the memory (c0).
    # Usually the hidden state refers to [h0, c0]. So a little bit of overloading of hidden state (I think)
    # This is how Genthail implements it

    # tf.reset_default_graph()


    if use_encoding_average_as_initial_state:
        img_mean = tf.reduce_mean(seq, axis=1)

        img_mean = tf.layers.batch_normalization(img_mean)

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
        "embedding_encoder", [token_vocab_size, embedding_size], tf.float32,
        initializer=tf.orthogonal_initializer)  # tf.float32 was default in the NMT tutorial

    # Look up embedding:
    #   decoder_inputs: [max_time, batch_size]
    #   decoder_emb_inp: [max_time, batch_size, embedding_size]
    decoder_emb_inp = tf.nn.embedding_lookup(
        embedding_decoder, decoder_inputs)

    print(decoder_emb_inp.shape)

    # Build RNN cell
    # decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)


    # Using this instead compared to NMT tutorial so we can initialize with orthogonal intializer (like Genthail)
    decoder_cell = tf.nn.rnn_cell.LSTMCell(
        num_units,
        initializer=tf.orthogonal_initializer,
    )

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
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess = tf.Session()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(output + 'summaries/train/',
                                         sess.graph)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    sess.run(init)

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

    #plt.title('Learning rate (10^) over the steps')
    #plt.ylabel('learning rate (10 ^)')
    #plt.xlabel('steps #')
    #plt.plot(np.log10(learning_rates))
    #plt.savefig(output + 'learning_rate.png')
    #plt.close()

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

    val_losses = []
    print("Num batches: ", num_train_batches)

    glob_step = sess.run(
        global_step)  # Get what global step we are at in training already (so that the learning_rate is set correct)

    gogo_gadget_saver = tf.train.Saver()

    log = pd.DataFrame([['step', 'epoch', 'i', 'loss', 'time', 'norm']])

    #_list = get_id_for_bucket(train_encoder_input_data_batches)

    for epoch in range(num_epochs):
        print("Epoch: ", epoch)

        for i in range(num_train_batches):
        #for i in _list:

            # Calculate running time for batch
            start_time = datetime.datetime.now()

            # Calculate the right learning rate for this step.

            lrate = get_learning_rate(glob_step, num_train_batches)

            input_data = {img: train_encoder_input_data_batches[i],
                          decoder_lengths: train_sequence_lengths_batches[i],
                          decoder_inputs: train_decoder_input_data_batches[i],
                          decoder_outputs: train_decoder_target_data_batches[i],
                          learning_rate: lrate
                          }

            output_tensors = [update_step, train_loss, global_norm, global_step, optimizer._lr]
            _, loss, global_grad_norm, glob_step, lr_rate = sess.run(output_tensors,
                                                                         feed_dict=input_data)


            end_time = datetime.datetime.now()
            delta = end_time - start_time
            # print("Time for batch in seconds: %.1f" % delta.total_seconds())
            stats = [glob_step, epoch, i, loss, delta, global_grad_norm]

            log.append(stats)

            # Write to tensorboard
            if glob_step % 200 == 0:
                _s = datetime.datetime.now()
                pair = zip(log[0], stats)
                for x, y in pair:
                    print(x,y)

                if not os.path.exists(output + 'pickles'):
                    os.makedirs(output + 'pickles')

                log.to_csv(output + 'pickles/robin-thicke-logs-a-big.csv', mode='w+')

                output_tensors = [merged, update_step, train_loss, optimizer._lr, global_norm, gradient_norms,
                                  global_step]
                summary, _, loss, lr_rate, global_grad_norm, grad_norms, glob_step = sess.run(output_tensors,
                                                                                              feed_dict=input_data)
                train_writer.add_summary(summary, glob_step)
                save_path = gogo_gadget_saver.save(sess, output + 'checkpoints/inner.ckpt')
                _e = datetime.datetime.now()
                print("Model saved in file: %s" % save_path)
                print("checkpoint cost: ", _e - _s)

        # Run the following in terminal to get up tensorboard: tensorboard --logdir=summaries/train
        save_path = gogo_gadget_saver.save(sess, output + 'checkpoints/model_'+str(epoch)+'.ckpt')
        print("Model saved in file: %s" % save_path)
        if calculate_val_loss:
            val_loss = get_validation_loss(num_val_batches,
                                           img, val_encoder_input_data_batches,
                                           decoder_lengths, val_sequence_lengths_batches,
                                           decoder_inputs, val_decoder_input_data_batches,
                                           decoder_outputs, val_decoder_target_data_batches)
            val_losses.append(val_loss)

            print("Val loss: ", val_loss)
            with open(output + 'validation-loss.txt', 'w') as f:
                f.write(repr(val_losses))
                # todo: get some notion of training set performance each time.


if __name__ == '__main__':
    main()
