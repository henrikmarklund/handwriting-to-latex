## Work in progress. Code adapted from https://github.com/tensorflow/nmt

def inference_tensor_beam(target_token_index,
              inference_batch_size,
              embedding_decoder,
              decoder_cell,
              decoder_initial_state,
              projection_layer,
              beam_width,
              maximum_iterations = max_token_length):
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


    decoder_initial_state = tf.contrib.seq2seq.tile_batch(
        decoder_initial_state, multiplier=hparams.beam_width)

    decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        cell=decoder_cell,
        embedding=embedding_decoder,
        start_tokens=tf.fill([inference_batch_size], tgt_sos_id),
        end_token=tgt_eos_id,
        initial_state=decoder_initial_state,
        beam_width=beam_width,
        output_layer=projection_layer,
        length_penalty_weight=0.0)

    # Dynamic decoding
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        inference_decoder, maximum_iterations=maximum_iterations)
    translations = outputs.sample_id
    logits = outputs.rnn_output
    return translations,logits

def predict_batch_beam(sess,
                  batch,
                  target_token_index,
                  embedding_decoder,
                  decoder_cell,
                  decoder_initial_state,
                  projection_layer,
                  img,
                  beam_size=4,
                  maximum_iterations=max_token_length):
    #for b in batches:
    batch_len = batch.shape[0]
    translation_t, logits_t = inference_tensor(target_token_index,
              batch_len,
              embedding_decoder,
              decoder_cell,
              decoder_initial_state,
              projection_layer,
              beam_size)
    translation, logits = sess.run([translation_t, logits_t], feed_dict={img: batch})
    return translation,logits