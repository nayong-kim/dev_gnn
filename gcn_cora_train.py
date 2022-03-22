#%%
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetold', name='Cora',
                    transform=NormalizeFeatures())
 
print(f"number of graphs: {len(dataset)}")
print(f"number of features: {dataset.num_features}")
print(f"numboer of classes: {dataset.num_classes}")
print(50*'=')

data = dataset[0]

print(data)
print(f"number of nodes: {data.num_nodes}")
print(f"number of edges: {data.num_edges}")
print(f"number of training nodes: {data.train_mask.sum()}")
print(f"training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}")
print(f"is undirected: {data.is_undirected()}")


# %%
print(data.x.shape)
# %%
data.x[0][:50]
# %%
data.edge_index.t()

# %%
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        
        self.Conv1 = GCNConv(data.num_features, hidden_channels)
        self.Conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, dataset.num_classes)
        
    def forward(self, x, edge_index):
        # first message passing layer
        x = self.Conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        
        #second message passing layer
        x = self.Conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training = self.training)
        
        # Output layer
        
        x = F.softmax(self.out(x), dim = 1)
        return x
    
    
model = GCN(hidden_channels= 6)
print(model)
        
        
# %%

# Initialize model
model = GCN(hidden_channels= 16)

# use CPU
device = torch.device("cpu")
model = model.to(device)
data = data.to(device)

# initialze Optimizer
learning_rate = 0.01
decay = 5e-4
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay= decay)

# define loss function
# (cross entrophyLoss 
# for classification problems with probability distributions)
criterion = torch.nn.CrossEntropyLoss()

# %%
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss
# %%
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct =pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum())/int(data.test_mask.sum())
    return test_acc
    

# %%
losses = []
for epoch in range(1, 1001):
    loss = train()
    losses.append(loss)
    if epoch % 10 == 0:
       print(f"Epoch:{epoch:03d}, loss = {loss:.4f}")
        
    
# %%
import seaborn as sns
# %%
import numpy as np

sample = 9
sns.set_theme(style="whitegrid")
print(model(data.x, data.edge_index).shape)
pred = model(data.x, data.edge_index)
sns.barplot(x=np.array(range(7)), y=pred[sample].detach().cpu().numpy())

# %%
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plt2arr(fig):
    rgb_str = fig.canvas.tostring_rgb()
    (w, h) = fig.canvas.get_width_height()
    rgba_arr = np.fromstring(rgb_str, dtype=np.uint8, sep='').reshape((w, h, -1))
    return rgba_arr

def visualize(h, color, epoch):
    fig = plt.figure(figsize=(5,5), frameon=False)
    fig.suptitle(f"Epoch = {epoch}")
    
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
    
    # Create scatterpoint from embedding
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0],
                z[:, 1],
                s=70,
                c=color.detach().cpu().numpy(),
                cmap="Set2")
    fig.canvas.draw()
    return plt2arr(fig)


for layer in model.children():
    if hasattr(layer, "reset_parameters"):
        layer.reset_parameters()

# %%
import warnings
warnings.filterwarnings('ignore')

#Train the model and sava visualizations 
images = []
for epoch in range(0, 2000):
    loss = train()
    if epoch % 50 == 0 :
        out = model(data.x, data.edge_index)
        images.append(visualize(out, color=data.y, epoch=epoch))
print("TSNE Visualization finished")


# %%
## !pip install moviepy
from moviepy.editor import ImageSequenceClip
fps = 1
filename = "./embeddings.gif"
clip = ImageSequenceClip(images, fps=fps)
clip.write_gif(filename, fps=fps)
# %%
