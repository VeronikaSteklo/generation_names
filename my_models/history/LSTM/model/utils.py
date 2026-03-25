import torch


def generate_title(model, vocab, device, text, max_len=20):
    model.eval()

    encoded_text = [vocab.stoi["<SOS>"]] + vocab.encode(" ".join(text.split()[:40])) + [vocab.stoi["<SEP>"]]
    input_tensor = torch.tensor([encoded_text]).to(device)

    result = []
    hidden = None

    with torch.no_grad():
        logits, hidden = model(input_tensor, hidden)

        last_token = torch.argmax(logits[0, -1, :]).item()

        for _ in range(max_len):
            if last_token == vocab.stoi["<EOS>"]:
                break
            result.append(last_token)

            curr_input = torch.tensor([[last_token]]).to(device)
            logits, hidden = model(curr_input, hidden)
            last_token = torch.argmax(logits[0, -1, :]).item()

    return vocab.decode(result)