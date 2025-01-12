from datasets import load_dataset
from evaluate import load
from transformers import (
    AutoTokenizer,
)

from preprocess import preprocess_function_wrapper
from postprocess import postprocess_qa_predictions
from consts import MAX_LENGTH, DOC_STRIDE, METRIC_NAME, MODEL_NAME

def test(trainer):
    xquad_es_dataset = load_dataset("xquad", "xquad.es")["validation"].shuffle(seed=42).select(range(20))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    xquad_es_dataset = xquad_es_dataset.map(
        preprocess_function_wrapper(tokenizer=tokenizer, max_length=MAX_LENGTH, doc_stride=DOC_STRIDE),
        batched=True)

    squad_metric = load(METRIC_NAME)

    # evaluate on English Validation (SQuAD)
    print("Evaluating on SQuAD validation (English)...")
    eval_results = trainer.evaluate()
    print(eval_results)

    # zero-Shot Evaluate on Spanish (XQuAD)
    print("Evaluating zero-shot on XQuAD (Spanish)...")

    # we must do a post-processing step for XQuAD similar to SQuAD
    raw_predictions = trainer.predict(xquad_es_dataset)
    predictions, references = postprocess_qa_predictions(
        xquad_es_dataset,
        xquad_es_dataset,
        raw_predictions.predictions
    )

    # evaluate with SQuAD metric structure
    results_xquad_es = squad_metric.compute(predictions=predictions, references=references)
    print("Zero-Shot Results on XQuAD (es):", results_xquad_es)
