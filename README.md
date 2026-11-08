# As the Model Size more than 100, the model itself not included in this repo.
# DistilBERT Sentiment Analysis

A Transformer-based NLP project that classifies IMDb movie reviews as
**positive** or **negative**. The project covers data preparation,
fine-tuning, evaluation, error analysis, model saving, reusable
inference, and deployment using **Azure Machine Learning Workspace**.

## Project Overview

-   **Task:** Binary sentiment classification
-   **Dataset:** IMDb movie reviews (`stanfordnlp/imdb`)
-   **Pretrained model:** `distilbert-base-uncased`
-   **Final input configuration:** Maximum sequence length of 512 tokens
-   **Cloud platform:** Azure Machine Learning Workspace

## Technologies

-   Python
-   PyTorch
-   Hugging Face Transformers and Datasets
-   Scikit-learn
-   Azure Machine Learning

## Workflow

1.  **Data preparation** -- Loaded the IMDb dataset, inspected labels,
    checked exact-text overlap between training and test data, and
    prepared training, validation, and test splits.
2.  **Tokenization** -- Converted reviews into model inputs and
    experimented with maximum sequence lengths of 256 and 512 tokens.
3.  **Fine-tuning** -- Fine-tuned the pretrained DistilBERT model for
    positive/negative sentiment classification.
4.  **Evaluation** -- Measured accuracy, precision, recall, and
    F1-score, and inspected incorrectly classified reviews.
5.  **Model persistence** -- Saved the fine-tuned model and tokenizer,
    then reloaded them to verify that inference worked independently of
    the training session.
6.  **Cloud deployment** -- Deployed the trained model using Azure
    Machine Learning Workspace.

## Results

  Experiment               Evaluation split   Accuracy   F1-score
  ---------------------- ------------------ ---------- ----------
  256-token baseline                   Test     91.15%     91.23%
  512-token experiment           Validation     92.81%     92.80%

**Note:** These results come from different evaluation splits, so they
should not be treated as a direct, controlled comparison. The 512-token
result is a validation result, not a final test-set result.

## Example Prediction

The saved model was tested with a new review:

> "The acting was brilliant and the story kept me engaged throughout."

Example output:

``` json
{
  "sentiment": "Positive",
  "confidence": 0.9963
}
```

The confidence value is the model's softmax probability for the
predicted class; it is not a guarantee that the prediction is correct.

## Repository Contents

The notebook and model files can be added to this repository as
appropriate. The trained model directory contains the model weights,
configuration, and tokenizer files needed to reload the model.

## Running Locally

Create and activate a Python environment, then install the main
dependencies:

``` bash
pip install torch transformers datasets accelerate scikit-learn
```

Load the saved model and tokenizer from the model directory using
Hugging Face Transformers:

``` python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "./sentiment-distilbert-512"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
```

The original training workflow was developed in a GPU-enabled notebook
environment. Training time and hardware requirements will vary depending
on the available system.

## Azure Machine Learning

The trained model was deployed using Azure Machine Learning Workspace
for cloud-based inference.

Deployment configuration, endpoint URL, authentication details, and
request examples are not included here. Add only non-sensitive
deployment instructions that match your Azure setup. **Never commit
access keys, tokens, credentials, or other secrets to GitHub.**

## Future Improvements

-   Evaluate both sequence-length configurations on the same validation
    split and reserve the test set for final evaluation.
-   Add a documented inference API request example for the Azure
    deployment.
-   Measure inference latency and resource usage.
-   Evaluate probability calibration and handling of uncertain
    predictions.

## Author

Sunny Saha
