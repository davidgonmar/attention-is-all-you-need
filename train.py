from dataset import Config, get_dataset
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

            #print(src, dec_inp, tgt)
            out = model(src, dec_inp)


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


def translate(model, src_sentence, src_tok, tgt_tok, max_len=config.seq_len):
    model.eval()
    

    # we'll pad both the src and tgt tensors, up to max_len
    src = src_tok.encode(src_sentence).ids
    src = torch.tensor([src], dtype=torch.int64)

    # pad the src tensor
    pad_len = max_len - len(src[0]) - 2  # -2 for sos and eos
    src = torch.cat(
        [
            torch.tensor([[src_tok.token_to_id("<s>")]]),
            src,
            torch.tensor([[src_tok.token_to_id("</s>")]]),
            torch.tensor([[src_tok.token_to_id("<pad>")]] * pad_len).view(1, -1),
        ],
        dim=-1,
    ).to(device)

    # we'll start with the <s> token
    dec_inp = torch.tensor([[tgt_tok.token_to_id("<s>")]], dtype=torch.int64).to(device)
    print("dec_inp:", dec_inp.shape)
    out = None
    for i in range(max_len - 1):
        # first, pad dec_inp to max_len
        _dec_inp = dec_inp
        pad_len = max_len - dec_inp.size(1)
        _dec_inp = torch.cat(
            [
                _dec_inp,
                torch.tensor([[tgt_tok.token_to_id("<pad>")]] * pad_len, dtype=torch.int64).view(1, -1).to(device),
            ],
            dim=-1,
        ).to(device)
        assert _dec_inp.size(1) == max_len


        # now, we can get the output
        out = model(src, _dec_inp)

        # get the last token
        last_tok = out[0, -1, :].argmax().item()
        dec_inp = torch.cat([dec_inp, torch.tensor([[last_tok]], dtype=torch.int64).to(device)], dim=-1)

        if last_tok == tgt_tok.token_to_id("</s>"):
            break

        
    

    print("Input:", src_sentence)
    print("Encoded input:", src)
    print("Output tensor:", out)
    print("Output tensor argmax:", out.argmax(dim=-1).squeeze().tolist())
    out = out.argmax(dim=-1).squeeze().tolist()
    out = tgt_tok.decode(out)
    print("Output:", out)

    print("Output:", "<", out.join(" "), ">")



translate(transformer, "I am a student", train_ds.src_tok, train_ds.tgt_tok)
translate(transformer, "Yesterday I was trying to avoid this", train_ds.src_tok, train_ds.tgt_tok)

# save the model

torch.save(transformer.state_dict(), "transformer.pth")