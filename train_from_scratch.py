# train_from_scratch.py
# Simplified training script for neural name generator
# Author: Gabriel Caballero
import argparse, os, torch, torch.nn.functional as F

def build_dataset(words, stoi, block_size):
    X, Y = [], []
    for w in words:
        context = [0]*block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/names.txt")
    p.add_argument("--block-size", type=int, default=3)
    p.add_argument("--embed", type=int, default=10)
    p.add_argument("--hidden", type=int, default=200)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--out", default="models/params.pt")
    args = p.parse_args()

    words = open(args.data).read().splitlines()
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}; stoi['.']=0
    itos = {i:s for s,i in stoi.items()}
    X,Y = build_dataset(words, stoi, args.block_size)
    g = torch.Generator().manual_seed(2147483647)

    C = torch.randn((27, args.embed), generator=g)
    W1 = torch.randn((args.block_size*args.embed, args.hidden), generator=g)
    b1 = torch.randn(args.hidden, generator=g)
    W2 = torch.randn((args.hidden,27), generator=g)
    b2 = torch.randn(27, generator=g)
    params = [C,W1,b1,W2,b2]
    for p_ in params: p_.requires_grad=True

    for i in range(args.steps):
        ix = torch.randint(0, X.shape[0], (args.batch,), generator=g)
        emb = C[X[ix]]
        h = torch.tanh(emb.view(-1, args.block_size*args.embed)@W1+b1)
        logits = h@W2+b2
        loss = F.cross_entropy(logits, Y[ix])
        for p_ in params: p_.grad=None
        loss.backward()
        lr=0.1 if i<100000 else 0.01
        for p_ in params: p_.data += -lr*p_.grad
        if (i+1)%1000==0: print(f"step {i+1}/{args.steps} loss={loss.item():.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save({"C":C,"W1":W1,"b1":b1,"W2":W2,"b2":b2,"block_size":args.block_size,"embed":args.embed,"hidden":args.hidden,"itos":itos}, args.out)

if __name__=="__main__":
    main()
