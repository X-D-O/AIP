import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
import math


# 定义位置编码（Positional Encoding）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


# 定义Transformer模型
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_classes, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(src.size(1))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.fc(output.mean(dim=0))  # 分类任务，取平均值进行预测
        return output


# 数据集定义（这里以简单的文本分类任务为例）
class TextDataset(Dataset):
    def __init__(self, data, labels, vocab, tokenizer):
        self.data = data
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer

.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.data[idx])
        token_ids = [self.vocab[token] for token in tokens]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# 生成数据集和词汇表
def build_vocab(tokenizer, data):
    vocab = build_vocab_from_iterator(map(tokenizer, data), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab


# 训练模型
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        outputs = model(inputs.T)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# 测试模型
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = [x.to(device) for x in batch]
            outputs = model(inputs.T)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy


# 主函数
if __name__ == '__main__':
    # 准备数据
    train_data = ["this is a positive review", "this is a negative review", "i loved the movie", "i hated the movie"]
    test_data = ["it was a good movie", "it was a bad movie"]
    train_labels = [1, 0, 1, 0]
    test_labels = [1, 0]

    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab(tokenizer, train_data)

    train_dataset = TextDataset(train_data, train_labels, vocab, tokenizer)
    test_dataset = TextDataset(test_data, test_labels, vocab, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: x)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: x)

    # 模型定义
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerClassifier(input_dim=len(vocab), embed_dim=64, num_heads=4, num_classes=2, num_encoder_layers=2,
                                  dim_feedforward=128).to(device)

    # 优化器与损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 训练与评估
    for epoch in range(10):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        test_loss, accuracy = evaluate_model(model, test_loader, criterion, device)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')
