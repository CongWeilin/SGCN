from utils import *
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, num_classes, layers, dropout):
        super(GCN, self).__init__()
        self.layers = layers
        self.nhid = nhid

        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat,  nhid))
        for _ in range(layers-1):
            self.gcs.append(GraphConvolution(nhid,  nhid))
        self.linear = nn.Linear(nhid, num_classes)
        self.relu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjs):
        for ell in range(len(self.gcs)):
            x = self.gcs[ell](x, adjs[ell])
            x = self.relu(x)
            x = self.dropout(x)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

    def partial_grad(self, x, adjs, targets):
        """
        Function to compute the stochastic gradient
        args : input, loss_function
        return : loss
        """
        outputs = self.forward(x, adjs)
        # compute the partial loss
        loss = F.binary_cross_entropy(outputs, targets)

        # compute gradient
        loss.backward()
        return loss.detach().cpu()

    def calculate_loss_grad(self, x, adjs, targets, batch_nodes):
        """
        Function to compute the large-batch loss and the large-batch gradient
        args : dataset, loss function, number of samples to be calculated
        return : total_loss
        """

        outputs = self.forward(x, adjs)
        # compute the partial loss
        loss = F.binary_cross_entropy(
            outputs[batch_nodes], targets[batch_nodes])

        # compute gradient
        loss.backward()

        full_grad_loss = 0.0
        # aggregate the norm of the batch gradient
        for p_net in self.parameters():
            full_grad_loss += p_net.grad.data.norm(2) ** 2
        # calculate the norm of the batch gradient
        full_grad_loss = torch.sqrt(full_grad_loss.cpu())

        return loss.detach().cpu(), full_grad_loss.detach().cpu()

    def calculate_f1(self, x, adjs, targets, batch_nodes):
        outputs = self.forward(x, adjs)
        outputs[outputs > 0.5] = 1
        outputs[outputs <= 0.5] = 0
        return f1_score(outputs[batch_nodes].cpu().detach(), targets[batch_nodes].cpu().detach(), average="micro")