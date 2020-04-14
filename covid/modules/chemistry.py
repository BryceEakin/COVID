from torch import nn
import torch as T

from chemprop.nn_utils import index_select_ND, get_activation_function
from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph

__ALL__ = ['MPNEncoder']

class MPNEncoder(nn.Module):
    def __init__(self, 
                 atom_fdim: int = 133,
                 bond_fdim: int = 147,
                 activation: str = 'ReLU',
                 hidden_size: int = 300,
                 bias: bool = False,
                 depth: int = 3,
                 dropout: float = 0.0,
                 layers_per_message: int = 1,
                 undirected: bool = False,
                 atom_messages: bool = False
                ):
        """
        Configures a message passing graph encoder
        
        Args:
            atom_fdim (int): feature dimensions to use for atoms, default 200
            bond_fdim (int): feature dimensions to use for bonds, default 200
            activation (str): 'ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', or 'ELU', default 'ReLU'
            hidden_size (int): dimension of messages, default 300
            bias (bool): include bias in internal linear layers, default False
            depth (int): number of message passing steps, default 3
            dropout (float): dropout rate on messages, default 0.0
            layers_per_message (int): linear layers included in message update function, default 1
            undirected (bool): propigate messages bidirectionally, default False
            atom_messages: pass messages from atoms to atoms along bonds, default False
        """
        super().__init__()
        
        self.dropout_layer = nn.Dropout(p=dropout)
        self.act_func = get_activation_function(activation)
        self.cached_zero_vector = T.zeros(hidden_size, requires_grad=False)
        
        # Input
        input_dim = atom_fdim if atom_messages else bond_fdim
        self.W_i = nn.Linear(input_dim, hidden_size, bias=bias)
        
        if atom_messages:
            w_h_input_size = hidden_size + bond_fdim
        else:
            w_h_input_size = hidden_size
            
        self.W_h = nn.Sequential(*([
            nn.Linear(w_h_input_size, hidden_size, bias=bias)
        ] + sum([
            [self.act_func, nn.Linear(hidden_size, hidden_size, bias=bias)] 
            for _ in range(layers_per_message-1)], []
        )))
        self.W_o = nn.Linear(atom_fdim + hidden_size, hidden_size)
        
        self.atom_messages = atom_messages
        self.depth = depth
        self.undirected = undirected
        
    def forward(self, mol_graph: BatchMolGraph, features_batch: T.Tensor):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()
        
        if self.atom_messages:
            a2a = mol_graph.get_a2a()
            input = self.W_i(f_atoms)
        else:
            input = self.W_i(f_bonds)
        message = self.act_func(input)
            
        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2
                
            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = T.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden
            
            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden
            
        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = T.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
        
        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens # (num_atoms, hidden_size)

                #mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vec = mol_vec.max(0).values
                mol_vecs.append(mol_vec)
                
        mol_vecs = T.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        
        return mol_vecs