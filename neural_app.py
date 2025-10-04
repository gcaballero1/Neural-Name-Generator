# src/neural_app.py
# Author: Gabriel Caballero
from dash import Dash, html, callback, Output, Input
import torch, os, torch.nn.functional as F

MODEL_PATH = "models/params.pt"

def load_params():
    return torch.load(MODEL_PATH, map_location="cpu")

def sample_name(params, g):
    C,W1,b1,W2,b2 = params["C"],params["W1"],params["b1"],params["W2"],params["b2"]
    itos=params["itos"]; block_size=params["block_size"]
    out=[]; context=[0]*block_size
    while True:
        emb=C[torch.tensor([context])]
        h=torch.tanh(emb.view(1,-1)@W1+b1)
        logits=h@W2+b2
        probs=F.softmax(logits,dim=1)
        ix=torch.multinomial(probs,1,generator=g).item()
        context=context[1:]+[ix]
        out.append(ix)
        if ix==0: break
    return ''.join(itos[i] for i in out).capitalize()

app=Dash(__name__)
server=app.server

app.layout=html.Div([
    html.H1("Neural Name Generator"),
    html.Button("Generate", id="btn", n_clicks=0),
    html.Div(id="out")
])

@callback(Output("out","children"), Input("btn","n_clicks"))
def gen(n):
    if not os.path.exists(MODEL_PATH):
        return "Train model first (python src/train_from_scratch.py)"
    params=load_params()
    g=torch.Generator().manual_seed(2147483647+int(n or 0))
    names=[sample_name(params,g) for _ in range(5)]
    return "Generated Names: "+", ".join(names)

if __name__=="__main__":
    app.run(port=8051, debug=True)
