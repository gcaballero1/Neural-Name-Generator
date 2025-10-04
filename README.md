# Neural-Name-Generator
A character-level neural network in PyTorch that learns from real names and generates new, realistic samples using a interactive Dash web interface to let users click a button and instantly generate names from the model.

![image alt](https://github.com/gcaballero1/Neural-Name-Generator/blob/main/screenshot.png?raw=true)

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
