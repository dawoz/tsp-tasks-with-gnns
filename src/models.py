from torch import nn
import torch
import torch_geometric as pyg

from src.gnn import GATWConv, GATv2WConv, GINEWConv
from torch_geometric.nn import MLP, GCNConv

# ______________________________________________________________________________________________________________________

# TRANSFORMER # 

class SelfAttention(nn.MultiheadAttention):
    def forward(self, x, *args, **kwargs):
        return super().forward(
            query=x, key=x, value=x, need_weights=False, *args, **kwargs)[0]


class SkipConnection(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.normalizer = nn.BatchNorm1d(embedding_dim, affine=True)

    def forward(self, input):
        return self.normalizer(input.swapaxes(1, 2)).swapaxes(1, 2)


class TransformerLayer(nn.Sequential):
    def __init__(self, num_attention_heads, embedding_dim, feed_forward_dim):
        super().__init__(
            SkipConnection(
                SelfAttention(
                    embed_dim=embedding_dim,
                    num_heads=num_attention_heads,
                    batch_first=True
                )
            ),
            Normalization(embedding_dim=embedding_dim),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(in_features=embedding_dim, out_features=feed_forward_dim),
                    nn.ReLU(),
                    nn.Linear(in_features=feed_forward_dim, out_features=embedding_dim)
                )
            ),
            Normalization(embedding_dim=embedding_dim),
        )

class Transformer(nn.Module):
    def __init__(self, embedding_dim, feed_forward_dim, num_attention_heads, num_attention_layers):
        super().__init__()
        self.proj = nn.Linear(2, embedding_dim)
        self.layers = nn.Sequential(*(
            TransformerLayer(num_attention_heads, embedding_dim, feed_forward_dim)
            for _ in range(num_attention_layers)
        ))
        
    def forward(self, instance):
        x = self.proj(instance)
        x = self.layers(x)
        return x

# ______________________________________________________________________________________________________________________

# POINTER NETWORK # 

class PointerNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.gru = nn.GRUCell(embedding_dim, embedding_dim)
        self.w1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w2 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v = nn.Linear(embedding_dim, 1, bias=False)
        self.logsoftmax = nn.LogSoftmax(-1)

    def forward(self, x, embs, hidden=None, mask=None, mask_fill_value=-10000):
        hidden = self.gru(x, hidden)
        x = self.w1(embs) + self.w2(hidden.unsqueeze(1))
        x = self.v(torch.tanh(x)).squeeze(-1)
        x = self.logsoftmax(x.masked_fill(mask, mask_fill_value) if mask is not None else x)
        return x, hidden
    
# ______________________________________________________________________________________________________________________

# GRAPH NEURAL NETWORKS # 

class GNN(nn.Module):
    def __init__(self,
                 embedding_dim=128,
                 num_layers=3,
                 gnn_layer_class=GCNConv,
                 add_residual_connections=True,
                 activation_function=torch.nn.functional.tanh,
                 batch_norm=True,
                 add_self_loops=True,
                 edge_embeddings=False,
                 edge_weights=True,
                 **kwargs):
        super().__init__()
        
        self.proj_node = nn.Linear(2, embedding_dim)
        
        gnn_constructor = {
            GCNConv: lambda: GCNConv(embedding_dim, embedding_dim, add_self_loops=add_self_loops),
            GATWConv: lambda: GATWConv(embedding_dim, embedding_dim, heads=8, concat=False, add_self_loops=add_self_loops),
            GATv2WConv: lambda: GATv2WConv(embedding_dim, embedding_dim, heads=8, concat=False, add_self_loops=add_self_loops, edge_dim=embedding_dim if edge_embeddings else None),
            GINEWConv: lambda: GINEWConv(kwargs.get('MLP', MLP([embedding_dim, embedding_dim, embedding_dim])), train_eps=True),
        }.get(gnn_layer_class, None)
        assert gnn_constructor is not None, f"Unknown GNN class: {gnn_layer_class}"

        self.layers = nn.ModuleList([gnn_constructor() for _ in range(num_layers)])
        self.batch_norm_h = pyg.nn.GraphNorm(embedding_dim) if batch_norm else None
        self.batch_norm_e = pyg.nn.GraphNorm(embedding_dim) if (batch_norm and edge_embeddings) else None
        self.add_residual_connections = add_residual_connections
        self.activation_function = activation_function
        self.edge_weights = edge_weights
        
        if edge_embeddings:
            self.proj_edge = nn.Linear(1, embedding_dim)
            self.edge_embeddings = edge_embeddings
            self.edge_mlp = MLP([3 * embedding_dim, embedding_dim, embedding_dim])
        else:
            self.edge_embeddings = False

    def forward(self, instance):
        # complete graph
        edge_index = torch.triu_indices(instance.shape[1], instance.shape[1], offset=1, device=instance.device)
        edge_index = pyg.utils.to_undirected(edge_index)
        
        # compute distances
        dist = torch.cdist(instance, instance, p=2)
        mask = torch.triu(torch.ones(dist.shape[1:], dtype=bool, device=instance.device), diagonal=1)
        mask = mask + mask.t()
        dist = dist[:, mask]
        data_list = [pyg.data.Data(x=instance[i], edge_index=edge_index.clone().detach(), weight=dist[i])
                     for i in range(instance.shape[0])]
        batch = pyg.data.Batch.from_data_list(data_list)
        
        h = self.proj_node(batch.x)
        
        if self.edge_embeddings:
            edge_embs = self.proj_edge(batch.weight.unsqueeze(-1))
        
        for layer in self.layers:
            res_h = h
    
            if self.edge_embeddings:
                # Pipeline: edge embeddings update -> node embeddings update
                res_e = edge_embs
                edge_embs = torch.cat([edge_embs, h[batch.edge_index.t()[:,0:1]].squeeze(), h[batch.edge_index.t()[:,1:2]].squeeze()], dim=-1)
                edge_embs = self.edge_mlp(edge_embs)
                edge_embs = self.activation_function(edge_embs)  # ADDED
                if self.add_residual_connections:
                    edge_embs = edge_embs + res_e
                if self.batch_norm_e is not None:
                    edge_embs = self.batch_norm_e(edge_embs, batch=batch.batch[batch.edge_index[0]])
                h = layer(x=h, edge_index=batch.edge_index, edge_attr=edge_embs, edge_weight=batch.weight if self.edge_weights else None)
            
            else:
                h = layer(x=h, edge_index=batch.edge_index, edge_weight=batch.weight if self.edge_weights else None)
            
            h = self.activation_function(h)
            if self.add_residual_connections:
                h = h + res_h
            if self.batch_norm_h is not None:
                h = self.batch_norm_h(h, batch=batch.batch)
        h = torch.stack(pyg.utils.unbatch(h, batch.batch))
    
        return h