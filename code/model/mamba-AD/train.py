import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class BallMillDataset(Dataset):
    def __init__(self, csv_path, seq_len=128):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        self.data = torch.FloatTensor(df.values)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return self.data[idx : idx + self.seq_len]

# 实例化
dataset = BallMillDataset('processed_1min_data_0_to_400.csv')
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = MambaAD(c_in=17, seq_len=128).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# 训练循环
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.cuda()
        optimizer.zero_grad()
        
        # 自重构任务：输入 X，预测/还原 X
        output = model(batch)
        loss = criterion(output, batch)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.6f}")