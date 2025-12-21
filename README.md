# Automatic Ticket Classification using Many-to-One RNN and LLM Response Generation
 
##  Project Overview
 
This project implements an automated customer support ticket classification system using Many-to-One RNN (LSTM) architecture combined with Google Gemini API for generating empathetic customer responses. The system can automatically categorize thousands of support tickets and draft polite acknowledgment responses.
 
**Domain:** Natural Language Processing (NLP), Generative AI
 
**Project Type:** GUVI HCL Final Project
 
---
 
## Problem Statement
 
Organizations receive thousands of customer support tickets daily, making manual categorization challenging. Misclassification leads to:

-  Delayed resolution times

-  Customer frustration

-  Increased operational costs
 
**Solution:** Build a Many-to-One RNN model that automatically classifies tickets into their respective queues based on ticket body text, followed by automated response generation using Gemini API.
 
##  Business Use Cases
 
1. **Customer Support Automation** – Route tickets to correct departments (Billing, Technical Support, Account Services)

2. **Faster Ticket Resolution** – Reduce response times with pre-drafted acknowledgments

3. **Cost Optimization** – Minimize manual triage requirements

4. **Customer Satisfaction** – Provide instant, empathetic replies
 
 
##  Skills & Technologies
 
### Skills Gained

- Text Preprocessing & Tokenization

- Sequence Modeling using RNN/LSTM

- Model Evaluation (Accuracy, Precision, Recall, F1-Score)

- Integration of ML with Generative AI

- Prompt Engineering
 
### Technical Stack

- **Deep Learning:** TensorFlow/Keras, LSTM, Bidirectional RNN

- **NLP:** Text preprocessing, tokenization, padding

- **Generative AI:** Google Gemini 2.0 Flash API

- **Data:** Hugging Face Datasets

- **Evaluation:** scikit-learn metrics

- **Visualization:** Matplotlib, Seaborn
 
##  Dataset Information
 
**Source:** [Hugging Face - Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
 
**Format:** JSON/CSV-like structure
 
**Loading Dataset:**

python

from datasets import load_dataset

ds = load_dataset("Tobi-Bueck/customer-support-tickets")
 
 
##  Installation
 
### Prerequisites

- Python 3.8+

- pip package manager

- Google Gemini API key
 
### Setup Instructions
 
1. **Clone or download the project**

bash

cd ticket-classification-project
 
2. **Install dependencies**

bash

pip install -r requirements.txt
 
 
3. **Get Gemini API Key**

   - Visit: https://makersuite.google.com/app/apikey

   - Create a new API key

   - Keep it secure for later use
 
 
## 🎬 Usage
 
The script will:

1.  Load dataset from Hugging Face

2.  Preprocess and tokenize text

3.  Build and train LSTM model

4.  Evaluate performance with metrics

5.  Generate visualizations

6.  Integrate Gemini API for response generation
 
### Step-by-Step Execution
 
**1. Data Exploration

from src.data_loader import DataLoader
 
loader = DataLoader()

df = loader.load_data()

stats = loader.get_basic_stats()

print(stats)
 
**2. Preprocessing
 
from src.preprocessing import TextPreprocessor
 
preprocessor = TextPreprocessor()

preprocessor.fit_tokenizer(X_train)

X_train_seq = preprocessor.transform_texts(X_train)
 
 
**3. Model Training 

python

from src.model import TicketClassifier
 
classifier = TicketClassifier(num_classes=num_classes)

classifier.build_model()

history = classifier.train(X_train_seq, y_train_enc, X_test_seq, y_test_enc)
 
 
**4. Evaluation 

python

from src.evaluate import ModelEvaluator
 
evaluator = ModelEvaluator(y_true, y_pred, class_names)

metrics = evaluator.calculate_metrics()

evaluator.plot_confusion_matrix()
 
 
**5. Gemini Integration

python

from src.gemini_integration import GeminiResponseGenerator
 
gemini = GeminiResponseGenerator(api_key="YOUR_API_KEY")

response = gemini.generate_response(ticket_text, predicted_queue)
 
 
### Enhanced Many-to-One Bidirectional LSTM
 
MODEL_CONFIG = {

    'vocab_size': 10000,

    'embedding_dim': 128,

    'lstm_units': 128,

    'dropout_rate': 0.5,

    'recurrent_dropout': 0.3,

    'max_sequence_length': 200,

    'batch_size': 64,

    'epochs': 20,

    'learning_rate': 0.001,

    'validation_split': 0.2

}
 
 
##  Evaluation Metrics
 
### Primary Metrics

- **Accuracy** - Overall classification accuracy (Target: >90%)

- **Precision** - Per-class and weighted average

- **Recall** - Per-class and weighted average

- **F1-Score** - Harmonic mean of precision and recall

- **Confusion Matrix** - Visual representation of misclassifications
 
### Outputs Generated

1. **Training History Plot** - Accuracy and loss curves

2. **Confusion Matrix** - Heatmap of predictions vs actual

3. **Classification Report** - Detailed per-class metrics

4. **Sample Predictions** - CSV with predicted queues and responses
 
 
##  Gemini API Integration
 
### Response Generation Process
 
1. **Input:** Ticket text + Predicted queue

2. **Prompt Engineering:** Structured prompt for empathetic responses

3. **Generation:** Gemini 2.0 Flash API call

4. **Output:** Polite, professional acknowledgment
 
### Example Response
 
**Ticket:** "I can't log into my account. Reset password link not working."
 
**Queue:** Technical Support
 
**Generated Response:**

"Thank you for reaching out to us. We understand how frustrating login issues can be. Your ticket has been received and forwarded to our Technical Support team. Our specialists will investigate the password reset issue and get back to you as soon as possible with a solution."
 
### Configuration

```python

GEMINI_CONFIG = {

    'model_name': 'gemini-2.0-flash-exp',

    'temperature': 0.7,

    'max_tokens': 300

}
 
 
##  Expected Results
 
### Model Performance

-  **Accuracy:** >90%

-  **Precision (Weighted):** >0.88

-  **Recall (Weighted):** >0.88

-  **F1-Score (Weighted):** >0.88

-  **Training Time:** 10-15 minutes
 
### Deliverables

1.  Trained LSTM model files

2.  Saved tokenizer and label encoder

3.  Evaluation plots (confusion matrix, training curves)

4.  Sample predictions with Gemini responses

5.  Complete source code

6.  Project documentation
 
 
## 🔧 Hyperparameter Tuning Tips
 
To achieve >90% accuracy:
 
1. **Increase vocabulary size** if needed

   python

   'vocab_size': 15000  # Try larger vocab

 
2. **Adjust LSTM units**

    python

   'lstm_units': 256  # More capacity

 
3. **Modify sequence length**

    python

   'max_sequence_length': 250  # Capture longer tickets

 
4. **Fine-tune dropout**

   'dropout_rate': 0.4  # Reduce if underfitting
 
 
5. **Extend training epochs** with early stopping

   'epochs': 30

   'early_stopping_patience': 7
 
 
##  Security Best Practices
 
1. **Never commit API keys** to version control

2. **Use environment variables**

   ```bash

   export GEMINI_API_KEY="your_api_key_here"
 
3. **Add to .gitignore**
 
   .env

   *.pickle

   *.h5
 
 
##  Troubleshooting
 
### Common Issues
 
**1. Dataset loading fails**

```python

# Check internet connection

# Try manual download

ds = load_dataset("Tobi-Bueck/customer-support-tickets", download_mode="force_redownload")
 
**2. Low accuracy (<90%)**

- Increase model complexity (more LSTM units)

- Train for more epochs

- Check for class imbalance

- Verify preprocessing steps
 
**3. Gemini API errors**
 
# Check API key validity

# Verify quota limits

# Implement fallback responses
 
 
**4. Memory issues**

```python

# Reduce batch size

'batch_size': 32  # Instead of 64
 
 
##  Dependencies
 
torch==2.1.0

tensorflow==2.15.0

datasets==2.14.0

transformers==4.35.0

numpy==1.24.3

pandas==2.1.0

scikit-learn==1.3.0

matplotlib==3.8.0

seaborn==0.12.2

google-generativeai==0.3.1

tqdm==4.66.1

python-dotenv==1.0.0
 
### Learning Resources

- [Hugging Face Datasets Docs](https://huggingface.co/docs/datasets/)

- [TensorFlow LSTM Guide](https://www.tensorflow.org/guide/keras/rnn)

- [Google Gemini API Docs](https://ai.google.dev/docs)

- [Text Classification Best Practices](https://developers.google.com/machine-learning/guides/text-classification)
 
 
##  Project Guidelines Checklist
 
-  Modular coding practices

-  Best practices in model evaluation

-  Secure API key management

-  Clear documentation of preprocessing steps

-  Comprehensive evaluation metrics

-  Sample outputs with Gemini responses

-  Trained model files saved

-  Confusion matrix visualization

-  Classification report generated
 
 
 
**Happy Coding! 🎉**
 
For best results, ensure you have a stable internet connection for dataset download and Gemini API calls.
 
# Automatic Ticket Classification using Many-to-One RNN and LLM Response Generation
 
##  Project Overview
 
This project implements an automated customer support ticket classification system using Many-to-One RNN (LSTM) architecture combined with Google Gemini API for generating empathetic customer responses. The system can automatically categorize thousands of support tickets and draft polite acknowledgment responses.
 
**Domain:** Natural Language Processing (NLP), Generative AI
 
**Project Type:** GUVI HCL Final Project
 
---
 
## Problem Statement
 
Organizations receive thousands of customer support tickets daily, making manual categorization challenging. Misclassification leads to:

-  Delayed resolution times

-  Customer frustration

-  Increased operational costs
 
**Solution:** Build a Many-to-One RNN model that automatically classifies tickets into their respective queues based on ticket body text, followed by automated response generation using Gemini API.
 
##  Business Use Cases
 
1. **Customer Support Automation** – Route tickets to correct departments (Billing, Technical Support, Account Services)

2. **Faster Ticket Resolution** – Reduce response times with pre-drafted acknowledgments

3. **Cost Optimization** – Minimize manual triage requirements

4. **Customer Satisfaction** – Provide instant, empathetic replies
 
 
##  Skills & Technologies
 
### Skills Gained

- Text Preprocessing & Tokenization

- Sequence Modeling using RNN/LSTM

- Model Evaluation (Accuracy, Precision, Recall, F1-Score)

- Integration of ML with Generative AI

- Prompt Engineering
 
### Technical Stack

- **Deep Learning:** TensorFlow/Keras, LSTM, Bidirectional RNN

- **NLP:** Text preprocessing, tokenization, padding

- **Generative AI:** Google Gemini 2.0 Flash API

- **Data:** Hugging Face Datasets

- **Evaluation:** scikit-learn metrics

- **Visualization:** Matplotlib, Seaborn
 
##  Dataset Information
 
**Source:** [Hugging Face - Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
 
**Format:** JSON/CSV-like structure
 
**Loading Dataset:**

python

from datasets import load_dataset

ds = load_dataset("Tobi-Bueck/customer-support-tickets")
 
 
##  Installation
 
### Prerequisites

- Python 3.8+

- pip package manager

- Google Gemini API key
 
### Setup Instructions
 
1. **Clone or download the project**

bash

cd ticket-classification-project
 
2. **Install dependencies**

bash

pip install -r requirements.txt
 
 
3. **Get Gemini API Key**

   - Visit: https://makersuite.google.com/app/apikey

   - Create a new API key

   - Keep it secure for later use
 
 
## 🎬 Usage
 
The script will:

1.  Load dataset from Hugging Face

2.  Preprocess and tokenize text

3.  Build and train LSTM model

4.  Evaluate performance with metrics

5.  Generate visualizations

6.  Integrate Gemini API for response generation
 
### Step-by-Step Execution
 
**1. Data Exploration

from src.data_loader import DataLoader
 
loader = DataLoader()

df = loader.load_data()

stats = loader.get_basic_stats()

print(stats)
 
**2. Preprocessing
 
from src.preprocessing import TextPreprocessor
 
preprocessor = TextPreprocessor()

preprocessor.fit_tokenizer(X_train)

X_train_seq = preprocessor.transform_texts(X_train)
 
 
**3. Model Training 

python

from src.model import TicketClassifier
 
classifier = TicketClassifier(num_classes=num_classes)

classifier.build_model()

history = classifier.train(X_train_seq, y_train_enc, X_test_seq, y_test_enc)
 
 
**4. Evaluation 

python

from src.evaluate import ModelEvaluator
 
evaluator = ModelEvaluator(y_true, y_pred, class_names)

metrics = evaluator.calculate_metrics()

evaluator.plot_confusion_matrix()
 
 
**5. Gemini Integration

python

from src.gemini_integration import GeminiResponseGenerator
 
gemini = GeminiResponseGenerator(api_key="YOUR_API_KEY")

response = gemini.generate_response(ticket_text, predicted_queue)
 
 
### Enhanced Many-to-One Bidirectional LSTM
 
MODEL_CONFIG = {

    'vocab_size': 10000,

    'embedding_dim': 128,

    'lstm_units': 128,

    'dropout_rate': 0.5,

    'recurrent_dropout': 0.3,

    'max_sequence_length': 200,

    'batch_size': 64,

    'epochs': 20,

    'learning_rate': 0.001,

    'validation_split': 0.2

}
 
 
##  Evaluation Metrics
 
### Primary Metrics

- **Accuracy** - Overall classification accuracy (Target: >90%)

- **Precision** - Per-class and weighted average

- **Recall** - Per-class and weighted average

- **F1-Score** - Harmonic mean of precision and recall

- **Confusion Matrix** - Visual representation of misclassifications
 
### Outputs Generated

1. **Training History Plot** - Accuracy and loss curves

2. **Confusion Matrix** - Heatmap of predictions vs actual

3. **Classification Report** - Detailed per-class metrics

4. **Sample Predictions** - CSV with predicted queues and responses
 
 
##  Gemini API Integration
 
### Response Generation Process
 
1. **Input:** Ticket text + Predicted queue

2. **Prompt Engineering:** Structured prompt for empathetic responses

3. **Generation:** Gemini 2.0 Flash API call

4. **Output:** Polite, professional acknowledgment
 
### Example Response
 
**Ticket:** "I can't log into my account. Reset password link not working."
 
**Queue:** Technical Support
 
**Generated Response:**

"Thank you for reaching out to us. We understand how frustrating login issues can be. Your ticket has been received and forwarded to our Technical Support team. Our specialists will investigate the password reset issue and get back to you as soon as possible with a solution."
 
### Configuration

```python

GEMINI_CONFIG = {

    'model_name': 'gemini-2.0-flash-exp',

    'temperature': 0.7,

    'max_tokens': 300

}
 
 
##  Expected Results
 
### Model Performance

-  **Accuracy:** >90%

-  **Precision (Weighted):** >0.88

-  **Recall (Weighted):** >0.88

-  **F1-Score (Weighted):** >0.88

-  **Training Time:** 10-15 minutes
 
### Deliverables

1.  Trained LSTM model files

2.  Saved tokenizer and label encoder

3.  Evaluation plots (confusion matrix, training curves)

4.  Sample predictions with Gemini responses

5.  Complete source code

6.  Project documentation
 
 
## 🔧 Hyperparameter Tuning Tips
 
To achieve >90% accuracy:
 
1. **Increase vocabulary size** if needed

   python

   'vocab_size': 15000  # Try larger vocab

 
2. **Adjust LSTM units**

    python

   'lstm_units': 256  # More capacity

 
3. **Modify sequence length**

    python

   'max_sequence_length': 250  # Capture longer tickets

 
4. **Fine-tune dropout**

   'dropout_rate': 0.4  # Reduce if underfitting
 
 
5. **Extend training epochs** with early stopping

   'epochs': 30

   'early_stopping_patience': 7
 
 
##  Security Best Practices
 
1. **Never commit API keys** to version control

2. **Use environment variables**

   ```bash

   export GEMINI_API_KEY="your_api_key_here"
 
3. **Add to .gitignore**
 
   .env

   *.pickle

   *.h5
 
 
##  Troubleshooting
 
### Common Issues
 
**1. Dataset loading fails**

```python

# Check internet connection

# Try manual download

ds = load_dataset("Tobi-Bueck/customer-support-tickets", download_mode="force_redownload")
 
**2. Low accuracy (<90%)**

- Increase model complexity (more LSTM units)

- Train for more epochs

- Check for class imbalance

- Verify preprocessing steps
 
**3. Gemini API errors**
 
# Check API key validity

# Verify quota limits

# Implement fallback responses
 
 
**4. Memory issues**

```python

# Reduce batch size

'batch_size': 32  # Instead of 64
 
 
##  Dependencies
 
torch==2.1.0

tensorflow==2.15.0

datasets==2.14.0

transformers==4.35.0

numpy==1.24.3

pandas==2.1.0

scikit-learn==1.3.0

matplotlib==3.8.0

seaborn==0.12.2

google-generativeai==0.3.1

tqdm==4.66.1

python-dotenv==1.0.0
 
### Learning Resources

- [Hugging Face Datasets Docs](https://huggingface.co/docs/datasets/)

- [TensorFlow LSTM Guide](https://www.tensorflow.org/guide/keras/rnn)

- [Google Gemini API Docs](https://ai.google.dev/docs)

- [Text Classification Best Practices](https://developers.google.com/machine-learning/guides/text-classification)
 
 
