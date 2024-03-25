from dataset import Config, get_dataset
from torch.utils.data import DataLoader
from model import Transformer

import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config(ds_name="opus_books", src_lang="en", tgt_lang="es", split=0.9)

train_ds, valid_ds = get_dataset(config)

vocab_size = train_ds.src_tok.get_vocab_size()

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=32, shuffle=False)


transformer = Transformer(
    num_heads=8,
    d_model=512,
    d_k=64,
    d_v=64,
    d_ff=2048,
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
).to(device)


def train(model, epochs, train_dl, valid_dl):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(epochs):
        model.train()
        for o in train_dl:
            src = o["src_seq"]
            dec_inp = o["tgt_seq_shifted"]
            tgt = o["tgt_seq_labels"]

            # we need to make src and tgt the same size
            # we can do this by padding the src tensor
            # we will pad the src tensor with the <pad> token

            # print(src.shape, tgt.shape)
            src = src.to(device)
            dec_inp = dec_inp.to(device)
            tgt = tgt.to(device)

            out = model(src, dec_inp)

            # print(out.shape)

            loss = criterion(
                out.view(-1, out.size(-1)), tgt.view(-1)
            )  # flatten the output and target tensors

            print(loss.item())

            loss.backward()

            # update the weights
            optimizer.step()

            # clear the gradients
            optimizer.zero_grad()


# train(transformer, 1, train_dl, valid_dl)
