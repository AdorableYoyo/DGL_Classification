

from dgl.data import MiniGCDataset
import matplotlib.pyplot as plt
import networkx as nx 



dataset = MiniGCDataset(80,10,20)



#check how the dataset looks like 
#dataset[5:10]



graph, label = dataset[0]
nx.draw(graph.to_networkx())
plt.title(f'Class :{label}')



import torch
import dgl
def collate(samples):
    #print(f"samples datatype{type(samples)}")
    graphs, labels = map(list, zip(*samples))
    #print(f"graphs and lables after map{graphs,labels},type{graphs}")
    batched_graph =dgl.batch(graphs)
    
    return batched_graph,torch.tensor(labels)



# test how mini batch forms: 
#test_sam = MiniGCDataset(5,10,20)

#collate(test_sam)

# summery : it adds all the nodes and edges



# how in_degrees works 
# g_test = test_sam[1][0]
#print(g_test.in_degrees(),g_test) 



from dgl.nn.pytorch import GraphConv



# aagregation and classification 

import torch.nn as nn
import torch.nn.functional as F 

class Classifier(nn.Module):
    def __init__(self,in_dim,hidden_dim,n_classes):
        super(Classifier,self).__init__()
        self.conv1 = GraphConv(in_dim,hidden_dim)
        self.conv2 = GraphConv(hidden_dim,hidden_dim)
        self.classify = nn.Linear(hidden_dim,n_classes)

    def forward(self,g): 
        h =g.in_degrees().view(-1,1).float()
        #print(f"starting h{h},{h.shape}")
        h = F.relu(self.conv1(g,h))
        h = F.relu(self.conv2(g,h))
        g.ndata['h']=h

        hg = dgl.mean_nodes(g,'h')
        #print(f"after mean{h},{h.shape}")
        hg = self.classify(hg)
        #print(f"hg size {hg.shape},hg {hg}")
        return hg
        
        



import torch.optim as optim
from torch.utils.data import DataLoader

trainset = MiniGCDataset(320,10,20)
testset = MiniGCDataset(80,10,20)

data_loader = DataLoader(trainset,batch_size=32, shuffle= True , collate_fn= collate)


model = Classifier(1,256,trainset.num_classes)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)
model.train()
epochs = 80

losses = []

for epoch in range(epochs):
    
    for iter, (bg,label) in enumerate(data_loader):
        prediction = model(bg)
        loss = loss_func(prediction,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    print('Epoch {}, loss {:.4f}'.format(epoch, loss.item()))



plt.title('cross entropy averaged over minibatches')
plt.plot(range(1,epochs+1),losses)



model.eval()
# Convert a list of tuples to two lists
test_X, test_Y = map(list, zip(*testset))
test_bg = dgl.batch(test_X)
test_Y = torch.tensor(test_Y).float().view(-1, 1)
probs_Y = torch.softmax(model(test_bg), 1)
sampled_Y = torch.multinomial(probs_Y, 1)
argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
print('Accuracy of sampled predictions on the test set: {:.4f}%'.format((test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
print('Accuracy of argmax predictions on the test set: {:4f}%'.format((test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))


