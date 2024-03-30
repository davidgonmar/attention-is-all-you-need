from dataset import Config, get_dataset, get_padding_mask
from torch.utils.data import DataLoader
from model import Transformer

import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config(ds_name="opus_books", src_lang="en", tgt_lang="es", split=0.9)

train_ds, valid_ds = get_dataset(config)

vocab_size = train_ds.src_tok.get_vocab_size()

print("Vocab size:", vocab_size)


train_dl = DataLoader(train_ds, batch_size=12, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=16, shuffle=False)


transformer = Transformer(
    num_heads=8,
    d_model=512,
    d_k=64,
    d_v=64,
    d_ff=2048,
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
).to(device)

if config.model_path:
    try:
        transformer.load_state_dict(torch.load(config.model_path))
    except:
        print("Model not found, training from scratch")


def train(model, epochs, train_dl, valid_dl):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=train_ds.tgt_tok.token_to_id("<pad>")) # ignore padding token
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    for epoch in range(epochs):
        model.train()
        for i, o in enumerate(train_dl):
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

            src_mask = get_padding_mask(src, train_ds.src_tok.token_to_id("<pad>")).to(device)
            tgt_mask = get_padding_mask(dec_inp, train_ds.tgt_tok.token_to_id("<pad>")).to(device)

            
            out = model(src, dec_inp, src_mask, tgt_mask)

            # decoded out
            decoded_out = torch.argmax(out, dim=-1)
            decoded_out = decoded_out.cpu().numpy()
            decoded_out = [train_ds.tgt_tok.decode(x) for x in decoded_out]
            #print("decoded_out:", decoded_out)
            loss = criterion(
                out.view(-1, out.size(-1)), tgt.view(-1)
            )  # flatten the output and target tensors

            print("iter:", i, " out of ", len(train_dl), " epoch:", epoch, " loss:", loss.item())

            loss.backward()

            # update the weights
            optimizer.step()

            # clear the gradients
            optimizer.zero_grad()
            if i % 30 == 0:
                torch.save(model.state_dict(), "transformer.pth")

if config.train:
    train(transformer, 100, train_dl, valid_dl)


