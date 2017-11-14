import numpy as np

def get_decoder_data(target_texts, target_characters, num_decoder_tokens, max_decoder_seq_length, target_token_index):

	decoder_input_data = np.zeros(
	    (len(target_texts), max_decoder_seq_length, num_decoder_tokens),
	    dtype='float32')
	decoder_target_data = np.zeros(
	    (len(target_texts), max_decoder_seq_length, num_decoder_tokens),
	    dtype='float32')

	num_other = 0

	for i, target_text in enumerate(target_texts):
	    for t, token in enumerate(target_text.split()):

			if token in target_token_index:
		        # decoder_target_data is ahead of decoder_input_data by one timestep
				decoder_input_data[i, t, target_token_index[token]] = 1.

				if t > 0:
					#decoder_target_data will be ahead by one timestep
		            # and will not include the start character.
					decoder_target_data[i, t - 1, target_token_index[token]] = 1.
			else:
				print("Hello")
				num_other = num_other + 1
				decoder_input_data[i, t, target_token_index['other']] = 1.

				if t > 0:
					#decoder_target_data will be ahead by one timestep
					# and will not include the start character.
					decoder_target_data[i, t - 1, target_token_index['other']] = 1.
		      


	return [decoder_input_data, decoder_target_data]


