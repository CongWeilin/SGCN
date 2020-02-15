from utils import *

class GraphConvolution(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(GraphConvolution, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)

    def forward(self, x, adj):
        support = self.linear(x)
        output = torch.spmm(adj, support)
        return output
