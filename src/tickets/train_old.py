import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tickets.data import load_data

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)


class SequenceAveragingModel(nn.Module):
    def __init__(self):
        super().__init__()
        embedding_dim = 768
        output_dim = 2
        self.embedding_module = bert_model.distilbert.embeddings.word_embeddings
        self.a_linear_module = nn.Linear(embedding_dim, output_dim)

    def forward(self, x, attention_mask):
        embeddings = self.embedding_module(x)
        averaged_embeddings = torch.mean(embeddings, dim=1)
        output = self.a_linear_module(averaged_embeddings)
        return output

#model = SequenceAveragingModel()
model = bert_model

data = load_data()
train_data = list(data["train"].take(32))


def evaluate_model(model, data):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for item in data:
            inputs = tokenizer(item["content"], return_tensors="pt", truncation=True)
            output = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
            output = output.logits
            predicted_label = torch.argmax(output, dim=1).item()
            if predicted_label == item["label"]:
                correct += 1
            total += 1
    print(f"Accuracy: {correct / total:.2%}")


evaluate_model(model, train_data)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    for item in train_data:
        model.zero_grad()
        # TODO
        inputs = tokenizer(item['content'], return_tensors="pt")
        label = torch.tensor([item['label']])
        output = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        output = output.logits
        loss = loss_fn(output, label)
        # TODO: gradient descent
        loss.backward() #compute gradients
        optimizer.step() #update parameters

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

evaluate_model(model, train_data)

#total = sum(p.numel() for p in model.parameters())