import torch
import torch_sparse
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GINEConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import SparseTensor, torch_sparse
from torch_geometric.utils import (add_self_loops, is_torch_sparse_tensor,
                                   remove_self_loops, softmax)
from torch_geometric.utils.sparse import set_sparse_value


class GATWConv(GATConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lin = self.lin_src
    
    """GATConv with edge_weights"""
    def forward(self, x, edge_index, edge_weight=None, edge_attr=None, size=None, return_attention_weights=None):
        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"

            if self.lin is not None:
                x_src = x_dst = self.lin(x).view(-1, H, C)
            else:
                # If the module is initialized as bipartite, transform source
                # and destination node features separately:
                assert self.lin_src is not None and self.lin_dst is not None
                x_src = self.lin_src(x).view(-1, H, C)
                x_dst = self.lin_dst(x).view(-1, H, C)

        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"

            if self.lin is not None:
                # If the module is initialized as non-bipartite, we expect that
                # source and destination node features have the same shape and
                # that they their transformations are shared:
                x_src = self.lin(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin(x_dst).view(-1, H, C)
            else:
                assert self.lin_src is not None and self.lin_dst is not None

                x_src = self.lin_src(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
               
                prev_num_edges = edge_index.shape[1]  # save to add zeros to weight
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
                # add zeros to weight: IMPORTANT, add self lloops adds the edges to the end of edge_index!!!
                if edge_weight is not None:
                    edge_weight = torch.cat([edge_weight, torch.zeros(edge_index.shape[1]-prev_num_edges, dtype=edge_weight.dtype, device=edge_weight.device)])
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # ======================================================================
        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr,
                                size=size)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index,
                        edge_weight,
                        x_src.size(self.node_dim),
                        False, # improved
                        False, # add_self_loops
                        self.flow,
                        x_src.dtype)

        alpha = alpha * edge_weight.unsqueeze(-1)
        # ======================================================================

        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


class GATv2WConv(GATv2Conv):
    def forward(self, x, edge_index, edge_weight=None, edge_attr=None, return_attention_weights=None):
        H, C = self.heads, self.out_channels

        x_l = None
        x_r = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                _edge_index, _edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                _edge_index, edge_attr = add_self_loops(
                    _edge_index, _edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    _edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index,
                        edge_weight,
                        x_l.size(self.node_dim),
                        False, # improved
                        self.add_self_loops, # add_self_loops
                        self.flow,
                        x_l.dtype)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr, edge_weight=edge_weight,
                             size=None)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out
    
    def message(self, x_j, x_i, edge_attr, edge_weight, index, ptr, size_i):
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        edge_weight = edge_weight.unsqueeze(1) # add num head dimension for broadcasting
        
        return x_j * alpha.unsqueeze(-1) * edge_weight.unsqueeze(-1)
    

class GINEWConv(GINEConv):
    """GINEConv with edge weights"""
    def forward(
        self, x, edge_index, edge_attr=None, edge_weight=None, size=None):

        if isinstance(x, Tensor):
            x = (x, x)
            
        edge_index, edge_weight = gcn_norm(  # yapf: disable
                                           edge_index,
                                           edge_weight,
                                           x[0].size(self.node_dim),
                                           False, # improved
                                           False, # add_self_loops
                                           'source_to_target',
                                           x[0].dtype)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size, edge_weight=edge_weight)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)


    def message(self, x_j, edge_attr, edge_weight):
        if self.lin is None and edge_attr is not None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)
            xp_j = (x_j + edge_attr).relu()
        else:
            xp_j = x_j

        return edge_weight.unsqueeze(-1) * xp_j