# **Multilingual Question-Answering with Zero-Shot Cross-Lingual Transfer**

This project demonstrates **fine-tuning** a **multilingual Transformer** (e.g., [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base)) on a **subset** of [SQuAD v1.1](https://huggingface.co/datasets/squad) (English), then evaluating **zero-shot** performance on the [XQuAD dataset](https://huggingface.co/datasets/xquad) (Spanish). By training only on **English** data, we test how well the model **transfers** its knowledge to **another language** without further fine-tuning.

---

## **Project Overview**

1. **Load and Subset SQuAD**  
   - SQuAD v1.1 is a large English QA dataset.  
   - For demonstration and to keep **training time** low, we use a **small subset** (e.g., 200 examples for training, 50 for validation).

2. **Load and Subset XQuAD**  
   - XQuAD is a **multilingual** variant of SQuAD, translated into multiple languages.  
   - We focus on the **Spanish** split and again use a small subset (e.g., 50 examples) to reduce runtime.

3. **Preprocess**  
   - Tokenize `(context, question)` pairs and compute **start/end** positions for answers.  
   - Keep crucial fields (`offset_mapping`, `example_id`, `context_text`) to **reconstruct** answers during post-processing.

4. **Train**  
   - We use **Hugging Face’s `Trainer`** API to **fine-tune** `xlm-roberta-base` for a few epochs on the English subset of SQuAD.

5. **Evaluate**  
   - Evaluate the model on (a) the English SQuAD validation subset, and (b) **zero-shot** on the Spanish XQuAD subset.  
   - Post-process predictions to convert logits into text answers, and compute **Exact Match (EM)** and **F1** with the official [SQuAD metric](https://huggingface.co/metrics/squad).

---

## **File Structure**

A typical structure might look like this:

```
.
├── .
│   ├── main.py                <- Main script
|   ├── train.py
│   ├── test.py
│   ├── consts.py
│   ├── README.md              <- (this file)
│   └── requirements.txt       <- optional, if you want to list exact dependencies
```

- **` main.py`**: The main code script with all the data loading, preprocessing, training, and evaluation logic.  
- **`README.md`**: This documentation.  
- **`requirements.txt`**: (Optional) A list of Python dependencies (e.g., `transformers`, `datasets`, `torch`, etc.).

---

## **Installation**

1. **Create and activate** a virtual environment (recommended, but optional):

    ```bash
    python -m venv env
    source env/bin/activate
    ```

2. **Install** the requirements:

   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

1. **Run** the script:

   ```bash
   python main.py
   ```

3. **Expected output**:
   - The script will **load** a small subset of SQuAD (200 training examples, 50 validation examples).  
   - It will **tokenize** and compute **start/end** positions, then train for a **small number of epochs** (e.g., 1 epoch).  
   - It will **evaluate** on the English validation subset, printing out **Exact Match** and **F1** scores.  
   - Finally, it will **predict** on the Spanish XQuAD subset (50 examples) and compute the same metrics **zero-shot**.

## **Customizing**

1. **Training Data Size**  
   - In `train.py`, you’ll see something like:
     ```python
     train_dataset = squad_dataset["train"].select(range(200))
     valid_dataset = squad_dataset["validation"].select(range(50))
     ```
     Increase (or decrease) these values for more (or fewer) examples.

2. **Number of Epochs**  
   - Adjust `num_train_epochs` in `TrainingArguments` to train longer on the small subset (or fewer if you just want a quick test).

3. **Batch Size**  
   - Change `per_device_train_batch_size` and `per_device_eval_batch_size` to accommodate your GPU memory.

4. **Different Languages**  
   - The script loads **Spanish** from XQuAD by specifying `"es"`.  
   - You can try `"ar"`, `"hi"`, `"zh"`, etc., if available in XQuAD or other multilingual QA datasets.

5. **Larger Models**  
   - Swap `model_name = "xlm-roberta-base"` for `"xlm-roberta-large"` or `"bert-base-multilingual-cased"`, if you have sufficient resources.

6. **Using the Full Dataset**  
   - Remove or adjust the `.select(...)` lines if you want to train on all of SQuAD. But note that training time will significantly increase.

---

## **References**

- [Hugging Face Transformers](https://github.com/huggingface/transformers)  
- [Hugging Face Datasets (SQuAD)](https://huggingface.co/datasets/squad)  
- [Hugging Face Datasets (XQuAD)](https://huggingface.co/datasets/xquad)  
- [SQuAD v1.1 Paper](https://arxiv.org/abs/1606.05250)  
- [XQuAD Paper](https://arxiv.org/abs/2003.11080)

---

## **License**

This code references open-source datasets (SQuAD, XQuAD) and uses the [Apache 2.0 license](https://github.com/huggingface/transformers/blob/main/LICENSE) from Hugging Face Transformers. Ensure compliance with any dataset-specific licenses if you plan to use them for production or commercial research.

---

### **Happy QA Experimentation!**

Feel free to open issues or discussions if you have questions about **multilingual QA**, **zero-shot transfer**, or **fine-tuning** larger language models. Contributions and ideas for improvement are always welcome!
