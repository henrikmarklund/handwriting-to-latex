def inference_tensor_beam(target_token_index,
              inference_batch_size,
              embedding_decoder,
              decoder_cell,
              decoder_initial_state,
              projection_layer,
              beam_size,
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

    # Helper
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder,
        tf.fill([inference_batch_size], tgt_sos_id), tgt_eos_id)

    # Decoder
    decoder_initial_state = cell.zero_state(batch_size, tf.float32)

    decoder_initial_state = tf.contrib.seq2seq.tile_batch(
        decoder_initial_state, multiplier=beam_size)


    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=decoder_cell,
            embedding=embedding_decoder,
            start_tokens=tgt_sos_id,
            end_token=tgt_eos_id,
            decoder_initial_state=decoder_initial_state,
            beam_width=beam_size,
            output_layer=projection_layer,
            lenth_penalty_weight=0.0)

    # Dynamic decoding
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        inference_decoder, maximum_iterations=maximum_iterations)
    translations = outputs.sample_id
    return translations

def predict_batch_beam(sess,
                  batch,
                  target_token_index,
                  embedding_decoder,
                  decoder_cell,
                  decoder_initial_state,
                  projection_layer,
                  img,
                  beam_size,
                  maximum_iterations=max_token_length):
    #for b in batches:
    batch_len = batch.shape[0]
    translation_t = inference_tensor(target_token_index,
              batch_len,
              embedding_decoder,
              decoder_cell,
              decoder_initial_state,
              projection_layer,
              beam_size)
    translation = sess.run(translation_t, feed_dict={img: batch})
    return translation