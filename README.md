# Neural-Name-Generator
A character-level neural network in PyTorch that learns from real names and generates new, realistic samples using a interactive Dash web interface to let users click a button and instantly generate names from the model.

![image alt](https://github.com/gcaballero1/Neural-Name-Generator/blob/main/screenshot.png?raw=true)

## How It Works
This model is a character-level neural language model trained on thousands of real names. It Converts names into input/output pairs using a fixed context window. Next, it runs gradient descent for thousands of iterations to adjust the parameters. Then, saves the learned parameters to models/params.pt for later use.

**Architecture:**
   - Embedding layer (C): learns vector representations for each character
   - Hidden layer (W1, b1): combines character embeddings into context-aware features
   - Output layer (W2, b2): predicts the probability distribution for the next character
   - Training: uses cross-entropy loss to optimize the network parameters
   - The model learns character sequences and produces plausible new names that resemble the training data but are not simple copies

**User Interface:**
The project includes two interactive web apps built with [Dash](https://dash.plotly.com/):

- *Faker App:* Uses the [Faker](https://faker.readthedocs.io/en/master/) library to instantly generate realistic random names. A button click refreshes the output, showing 5 new names each time.  
- *Neural App:* Loads the trained PyTorch model and provides a button-driven interface that samples names from the learned probability distribution. This demonstrates how deep learning models can be wrapped in a user-friendly UI.  

Both apps run locally in your browser and display results in real time, making it easy to explore both rule-based and neural-generated names.

## Dependencies
1. Python 3.8 or higher
2. pip
3. Libraries:  
   - [Dash](https://dash.plotly.com/)  
   - [Faker](https://faker.readthedocs.io/en/master/)  
   - [PyTorch](https://pytorch.org/)  
   - [NumPy](https://numpy.org/)  
   - [Requests](https://requests.readthedocs.io/en/latest/)

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download dataset
python get_data.py

# Train (quick demo)
python train_from_scratch.py --steps 5000

# Run Dash app (Faker demo on port 8050, Neural demo on port 8051)
python faker_app.py
python neural_app.py
```

## Structure
- `faker_app.py` – Dash + Faker app
- `train_from_scratch.py` – Trains neural model
- `neural_app.py` – Dash UI for neural generator
- `get_data.py` – downloads names.txt
