
def preprocess_function_wrapper(tokenizer, max_length, doc_stride):
    def preprocess_function(examples):
        """
        Tokenize the contexts and questions for QA,
        and find the start/end positions of the answer in tokenized form.
        """
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        # trim to the to the size of examples
        # TODO(Hue): this is a workaround. fix this
        for k in inputs.keys():
          inputs[k] = inputs[k][:len(examples["question"])]

        sample_mapping = inputs.pop("overflow_to_sample_mapping")
        offset_mapping = inputs["offset_mapping"]

        start_positions = []
        end_positions = []
        
        for i, offsets in enumerate(offset_mapping):
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # find where the context (sequence_ids == 1) starts/ends
            sequence_ids = inputs.sequence_ids(i)
            
            # the context starts where sequence_ids == 1
            token_start_index = 0
            while token_start_index < len(sequence_ids) and sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # the context ends where sequence_ids == 1 changes back to None
            token_end_index = len(sequence_ids) - 1
            while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # if the answer is outside the current window
            if not (offsets[token_start_index][0] <= start_char and 
                    offsets[token_end_index][1] >= end_char):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # otherwise, pinpoint start/end tokens of the answer
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    return preprocess_function
