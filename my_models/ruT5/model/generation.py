import torch


def generate_title(text, model, tokenizer, device):
    model.eval()
    input_text = "make_title: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=32,
            num_beams=5,
            no_repeat_ngram_size=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)