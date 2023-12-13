import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset, Planetoid, LRGBDataset
from torch_geometric.transforms import NormalizeFeatures, Constant, OneHotDegree
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv, GPSConv, global_mean_pool, Linear
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
from torch_geometric.loader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_file = 'output.txt'

##########################################################################################################################
# Cora
##########################################################################################################################

dataset = Planetoid(root='.', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

print('Running Cora')

train_dataset = data.train_mask
test_dataset = data.test_mask

dataset = dataset.shuffle()

train_mask = data.train_mask
test_mask = data.test_mask

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=-1)

class GIN(torch.nn.Module):
    def __init__(self):
        super(GIN, self).__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(dataset.num_features, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16))
        nn2 = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=-1)


class GATv2(torch.nn.Module):
    def __init__(self):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(dataset.num_features, 16)
        self.conv2 = GATv2Conv(16, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=-1)

def train(model):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = torch.nn.functional.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

def test(model, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)
    correct = pred[mask].eq(data.y[mask]).sum().item()
    acc = correct / mask.sum().item()
    return acc

hidden_channels = [64, 64, 64]
models = [GCN(), GIN(), GATv2()]

for model in models:
    with open(output_file, 'a') as file:
        file.write(f"Training {model.__class__.__name__}" + '\n')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(501):
        loss = train(model)
        train_acc = test(model, data.train_mask)
        test_acc = test(model, data.test_mask)
        if epoch % 100 == 0:
            with open(output_file, 'a') as file:
                file.write(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}' + '\n')

class GPSConvNet(nn.Module):
    def __init__(self, hidden_channels, heads=1, dropout=0.0, act='relu'):
        super(GPSConvNet, self).__init__()
        self.GPSConvs = nn.ModuleList()
        prev_channels = dataset.num_node_features
        h = hidden_channels[0]
        self.preprocess = nn.Sequential(Linear(dataset.num_node_features, 2 * h), nn.GELU(), Linear(2 * h, h), nn.GELU())
        for h in hidden_channels:
            gatv2_conv = GATv2Conv(h, h // heads, heads=heads, dropout=dropout)
            gps_conv = GPSConv(h, gatv2_conv, heads=4, dropout=dropout, act=act)
            self.GPSConvs.append(gps_conv)
            prev_channels = h
        self.final_conv = GATv2Conv(prev_channels, dataset.num_classes, heads=1, dropout=dropout)
        
    def forward(self, x, edge_index):
        x = self.preprocess(x)
        for gps_conv in self.GPSConvs:
            x = x.float()
            x = F.relu(gps_conv(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
        x = self.final_conv(x, edge_index)
        return F.log_softmax(x, dim=1)

hidden_channels = [64, 64, 64]

model = GPSConvNet(hidden_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = torch.nn.functional.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

def test(model, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)
    correct = pred[mask].eq(data.y[mask]).sum().item()
    acc = correct / mask.sum().item()
    return acc

with open(output_file, 'a') as file:
        file.write(f"Training {model.__class__.__name__}" + '\n')

for epoch in range(501):
    loss = train()
    train_acc = test(model, data.train_mask)
    test_acc = test(model, data.test_mask)
    if epoch % 100 == 0:
        with open(output_file, 'a') as file:
                file.write(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}' + '\n')



##########################################################################################################################
# Enzymes
##########################################################################################################################

print('Running Enzymes')

dataset = TUDataset(root='.', name='ENZYMES')
data = dataset[0]

train_dataset = dataset[:450]
data = dataset[0]
dataset = dataset.shuffle()

train_dataset = dataset[:450]
test_dataset = dataset[450:]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x
    
class GIN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GIN, self).__init__()
        self.conv1 = GINConv(Linear(dataset.num_node_features, hidden_channels))
        self.conv2 = GINConv(Linear(hidden_channels, hidden_channels))
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)       
        return x

class GATv2(torch.nn.Module):
    def __init__(self, hidden_channels, heads):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(dataset.num_node_features, hidden_channels, heads=heads)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads)
        self.lin = Linear(hidden_channels * heads, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x) 
        return x
        

gcn_model = GCN(hidden_channels=64)
gin_model = GIN(hidden_channels=64)
gatv2_model = GATv2(hidden_channels=64, heads=8)

models = [gcn_model, gin_model, gatv2_model]

def train(model):
    model.train()
    for data in train_loader:
         out = model(data.x, data.edge_index, data.batch) 
         loss = criterion(out, data.y)
         loss.backward()
         optimizer.step()
         optimizer.zero_grad()

def test(model, loader):
     model.eval()
     correct = 0
     for data in loader:
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)
         correct += int((pred == data.y).sum())
     return correct / len(loader.dataset)

for model in models:

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    with open(output_file, 'a') as file:
        file.write(f"Training {model.__class__.__name__}" + '\n')


    for epoch in range(1, 501):
        train(model)
        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        if epoch % 100 == 0:
            with open(output_file, 'a') as file:
                file.write(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}' + '\n')


class GPSConvNet(nn.Module):
    def __init__(self, hidden_channels, heads=1, dropout=0.0, act='relu'):
        super(GPSConvNet, self).__init__()
        self.GPSConvs = nn.ModuleList()
        h = hidden_channels[0]
        self.preprocess = nn.Sequential(Linear(dataset.num_node_features, 2 * h), nn.GELU(), Linear(2 * h, h), nn.GELU())
        for h in hidden_channels:
            gatv2_conv = GATv2Conv(h, h // heads, heads=heads, dropout=dropout)
            gps_conv = GPSConv(h, gatv2_conv, heads=4, dropout=dropout, act=act)
            self.GPSConvs.append(gps_conv)  
        self.final_lin = Linear(hidden_channels[-1], dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.preprocess(x)
        for gps_conv in self.GPSConvs:
            x = x.float()
            x = F.relu(gps_conv(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.final_lin(x)
        return F.log_softmax(x, dim=1)

hidden_channels = [64, 64, 64]
data = dataset[0]

model = GPSConvNet(hidden_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

train_dataset = dataset[:450]
test_dataset = dataset[450:]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def train():
    model.train()

    for data in train_loader:
         out = model(data.x, data.edge_index, data.batch)
         loss = criterion(out, data.y)
         loss.backward()
         optimizer.step()
         optimizer.zero_grad()

def test(loader):
     model.eval()

     correct = 0
     for data in loader:
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)
         correct += int((pred == data.y).sum())
     return correct / len(loader.dataset)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
with open(output_file, 'a') as file:
    file.write(f"Training {model.__class__.__name__}" + '\n')

for epoch in range(1, 501):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    if epoch % 100 == 0:
       with open(output_file, 'a') as file:
            file.write(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}' + '\n')



##########################################################################################################################
# IMDB
##########################################################################################################################
print('Running IMDB')
dataset = TUDataset(root='.', name='IMDB-BINARY')

data = dataset[0]

train_dataset = dataset[:850]
test_dataset = dataset[850:]
data = dataset[0]


dataset = dataset.shuffle()

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x
    
class GIN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GIN, self).__init__()
        self.conv1 = GINConv(Linear(dataset.num_node_features, hidden_channels))
        self.conv2 = GINConv(Linear(hidden_channels, hidden_channels))
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

class GATv2(torch.nn.Module):
    def __init__(self, hidden_channels, heads):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(dataset.num_node_features, hidden_channels, heads=heads)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads)
        self.lin = Linear(hidden_channels * heads, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)    
        return x
        

gcn_model = GCN(hidden_channels=64)
gin_model = GIN(hidden_channels=64)
gatv2_model = GATv2(hidden_channels=64, heads=8)

models = [gcn_model, gin_model, gatv2_model]

def train(model):
    model.train()

    for data in train_loader:
         out = model(data.x, data.edge_index, data.batch)
         loss = criterion(out, data.y)
         loss.backward()
         optimizer.step()
         optimizer.zero_grad()

def test(model, loader):
     model.eval()

     correct = 0
     for data in loader:
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)
         correct += int((pred == data.y).sum())
     return correct / len(loader.dataset)

for model in models:

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    with open(output_file, 'a') as file:
        file.write(f"Training {model.__class__.__name__}" + '\n')

    for epoch in range(1, 501):
        train(model)
        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        if epoch % 100 == 0:
            with open(output_file, 'a') as file:
                file.write(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}' + '\n')


class GPSConvNet(nn.Module):
    def __init__(self, hidden_channels, heads=1, dropout=0.0, act='relu'):
        super(GPSConvNet, self).__init__()
        self.GPSConvs = nn.ModuleList()
        h = hidden_channels[0]
        self.preprocess = nn.Sequential(Linear(dataset.num_node_features, 2 * h), nn.GELU(), Linear(2 * h, h), nn.GELU())
        for h in hidden_channels:
            gatv2_conv = GATv2Conv(h, h // heads, heads=heads, dropout=dropout)
            gps_conv = GPSConv(h, gatv2_conv, heads=4, dropout=dropout, act=act)
            self.GPSConvs.append(gps_conv)  
        self.final_lin = Linear(hidden_channels[-1], dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.preprocess(x)
        for gps_conv in self.GPSConvs:
            x = x.float()
            x = F.relu(gps_conv(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.final_lin(x)
        return F.log_softmax(x, dim=1)

hidden_channels = [64, 64, 64]
dataset = TUDataset(root='.', name='IMDB-BINARY', transform=OneHotDegree(max_degree=140))
data = dataset[0]

model = GPSConvNet(hidden_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

train_dataset = dataset[:850]
test_dataset = dataset[850:]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def train():
    model.train()

    for data in train_loader:
         out = model(data.x, data.edge_index, data.batch)
         loss = criterion(out, data.y)
         loss.backward()
         optimizer.step()
         optimizer.zero_grad()

def test(loader):
     model.eval()

     correct = 0
     for data in loader:
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)
         correct += int((pred == data.y).sum())
     return correct / len(loader.dataset)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
with open(output_file, 'a') as file:
    file.write(f"Training {model.__class__.__name__}" + '\n')

for epoch in range(1, 501):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    if epoch % 100 == 0:
        with open(output_file, 'a') as file:
            file.write(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}' + '\n')



##########################################################################################################################
# PascalVOC-SP
##########################################################################################################################

print('Running PascalVOC-SP')
dataset = LRGBDataset(root='.', name='PascalVOC-SP')
data = dataset[0]

train_dataset = dataset[:7000]
test_dataset = dataset[7000:]

data = dataset[0]
dataset = dataset.shuffle()

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
class GIN(torch.nn.Module):
    def __init__(self):
        super(GIN, self).__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(dataset.num_node_features, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(16)
        nn2 = torch.nn.Sequential(torch.nn.Linear(16, dataset.num_classes))
        self.conv2 = GINConv(nn2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)    

class GATv2(torch.nn.Module):
    def __init__(self):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(dataset.num_features, 8, heads=8)
        self.conv2 = GATv2Conv(8 * 8, dataset.num_classes, heads=1)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)

def train(model, optimizer, train_loader):
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

def test(model, test_loader):
    model.eval()
    test_correct = 0
    total_nodes = 0
    for data in test_loader:
        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=-1)
            test_correct += pred.eq(data.y).sum().item()
            total_nodes += data.num_nodes
    test_acc = test_correct / total_nodes
    return test_acc


models = [GCN(), GIN(), GATv2()]

for model in models:
    with open(output_file, 'a') as file:
        file.write(f"Training {model.__class__.__name__}" + '\n')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(501):
        train(model, optimizer, train_loader)
        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        if epoch % 100 == 0:
            with open(output_file, 'a') as file:
                file.write(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}' + '\n')


class GPSConvNet(nn.Module):
    def __init__(self, hidden_channels, heads=1, dropout=0.0, act='relu'):
        super(GPSConvNet, self).__init__()
        self.GPSConvs = nn.ModuleList()
        h = hidden_channels[0]
        self.preprocess = nn.Sequential(Linear(dataset.num_node_features, 2 * h), nn.GELU(), Linear(2 * h, h), nn.GELU())
        for h in hidden_channels:
            gatv2_conv = GATv2Conv(h, h // heads, heads=heads, dropout=dropout)
            gps_conv = GPSConv(h, gatv2_conv, heads=4, dropout=dropout, act=act)
            self.GPSConvs.append(gps_conv)  
        self.final_lin = Linear(hidden_channels[-1], dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.preprocess(x)
        for gps_conv in self.GPSConvs:
            x = x.float()
            x = F.relu(gps_conv(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.final_lin(x)
        return F.log_softmax(x, dim=1)

hidden_channels = [64, 64, 64]
data = dataset[0]

model = GPSConvNet(hidden_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def train():
    model.train()

    for data in train_loader:
         out = model(data.x, data.edge_index, data.batch)
         loss = criterion(out, data.y)
         loss.backward()
         optimizer.step()
         optimizer.zero_grad()

def test(loader):
     model.eval()

     correct = 0
     for data in loader:
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)
         correct += int((pred == data.y).sum())
     return correct / len(loader.dataset)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
with open(output_file, 'a') as file:
    file.write(f"Training {model.__class__.__name__}" + '\n')
for epoch in range(1, 501):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    if epoch % 100 == 0:
        with open(output_file, 'a') as file:
            file.write(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}' + '\n')


print('Finished, check output.txt for details.')