import numpy as np

def postprocess_qa_predictions(examples, features, raw_predictions):
    """
    Convert model outputs into text answers using offsets.
    (Simplified approach. For thoroughness, see HF's official script.)
    """
    all_start_logits, all_end_logits = raw_predictions
    # mapping from feature to example
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = [[] for _ in range(len(examples["id"]))]

    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["id"]]].append(i)

    predictions = {}
    # for each example, pick the best span across all features
    for example_index, feature_indices in enumerate(features_per_example):
        context = examples["context"][example_index]

        # best prediction
        best_score = -float("inf")
        best_answer = ""

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features["offset_mapping"][feature_index]
            
            start_index = int(np.argmax(start_logits))
            end_index = int(np.argmax(end_logits))

            if start_index < len(offset_mapping):
                start_char = offset_mapping[start_index][0]
            else:
                start_char = 0

            if end_index < len(offset_mapping):
                end_char = offset_mapping[end_index][1]
            else:
                end_char = 0

            # reconstruct answer text
            answer_text = context[start_char: end_char]
            score = start_logits[start_index] + end_logits[end_index]
            
            if score > best_score:
                best_score = score
                best_answer = answer_text
        
        predictions[examples["id"][example_index]] = best_answer
    
    # convert to squad_metric format
    formatted_predictions = [
        {"id": k, "prediction_text": v} for k, v in predictions.items()
    ]
    references = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in examples
    ]
    return formatted_predictions, references
