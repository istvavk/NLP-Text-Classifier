# Sports News Text Classifier

Modular Python application for classifying sports news articles into **previews** and **match reports** using **natural language processing (NLP)** and **deep learning** techniques.

The project is designed for academic purposes and combines advanced Python programming concepts with neural network–based text classification. It provides both a command-line workflow and a graphical user interface (GUI) for interactive testing.

---

## Project Overview

The goal of this project is to automatically distinguish between:
- **Preview articles** – texts describing upcoming sports events
- **Report articles** – texts summarizing already played matches

The system processes raw text input, applies NLP preprocessing, and uses a trained **LSTM neural network** to predict the article category along with a confidence score.

---

## Key Features

- Modular project architecture (models, training, utilities, GUI)
- Text preprocessing and tokenization
- Deep learning model (LSTM) implemented using PyTorch
- Train / validation / test split with proper evaluation
- F1-macro metric to address class imbalance
- Saved trained model for immediate inference
- Graphical user interface for user interaction
- Unit tests and doctests
- Use of advanced Python programming techniques:
  - decorators
  - closures
  - abstract base classes (ABC)
  - protocols
  - concurrency (ThreadPoolExecutor / asyncio)
  - type annotations (PEP484, PEP526)

---

## Technologies Used

- Python 3
- PyTorch
- Tkinter
- scikit-learn
- NumPy
- pandas

---

## Project Structure

textClassifier/
│
├── models/ # Baseline and neural network models
├── training/ # Training, evaluation, dataset splitting
├── utils/ # Preprocessing, decorators, concurrency utilities
├── gui/ # Graphical user interface
├── tests/ # Unit tests
├── saved_models/ # Trained LSTM model
├── requirements.txt
└── README.md

---

## Installation

It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

Install the required dependencies:

pip install -r requirements.txt

---

## Training the Model

From the project root directory, run:

python -m training.train

The model with the best validation F1-macro score is automatically saved

---

## Model Evaluation

To evaluate the trained model on the test dataset:

python -m training.evaluate

The script prints accuracy, F1-macro score, and a detailed classification report

---

## Running the GUI

Launch the graphical user interface with:

python -m gui.app

Enter a sports news article into the text field to obtain the predicted class and confidence score

---

## Testing

Run unit tests:

python -m unittest discover -s tests -v

Run doctests:

python -m doctest -v utils/preprocessing.py

---

## Notes

Due to the small dataset size and heuristic labeling, evaluation results may exhibit higher variance

F1-macro is used as the primary evaluation metric because of class imbalance

The project is intended for educational and academic use

## License

This project is provided for academic purposes only
