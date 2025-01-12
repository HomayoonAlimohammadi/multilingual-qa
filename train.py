from datasets import load_dataset
from evaluate import load
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer
)

from consts import METRIC_NAME
from preprocess import preprocess_function_wrapper
from postprocess import postprocess_qa_predictions


def compute_metrics_wrapper(valid_dataset, squad_metric):
    def compute_metrics(p):
        """
        Evaluate predictions with the SQuAD metric (F1 and EM).
        """
        examples = valid_dataset
        features = valid_dataset
        predictions, references = postprocess_qa_predictions(examples, features, p.predictions)
        return squad_metric.compute(predictions=predictions, references=references)

    return compute_metrics

def train():
    squad_dataset = load_dataset("squad")
    squad_dataset = squad_dataset.remove_columns("title")

    train_dataset = squad_dataset["train"].shuffle(seed=42).select(range(80))
    valid_dataset = squad_dataset["validation"].shuffle(seed=42).select(range(20))

    model_name = "xlm-roberta-base"  # or "xlm-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    max_length = 512   # maximum total input sequence length
    doc_stride = 256   # overlap between chunks when splitting context

    train_dataset = train_dataset.map(
        preprocess_function_wrapper(tokenizer=tokenizer, max_length=max_length, doc_stride=doc_stride),
        batched=True)
    valid_dataset = valid_dataset.map(
        preprocess_function_wrapper(tokenizer=tokenizer, max_length=max_length, doc_stride=doc_stride),
        batched=True)
    
    
    training_args = TrainingArguments(
        output_dir="./_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_steps=5,
        push_to_hub=False,
        report_to="none"
    )

    squad_metric = load(METRIC_NAME)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_wrapper(valid_dataset=valid_dataset, squad_metric=squad_metric)
    )

    # train the Model on SQuAD (English)
    trainer.train()

    return trainer

