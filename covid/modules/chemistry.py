from torch import nn
import torch as T

from chemprop.nn_utils import index_select_ND, get_activation_function
from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph

__ALL__ = [
    'MPNEncoder', 
    'ChemicalHeadModel', 
    'ChemicalMiddleModel', 
    'ChemicalTailModel',
    'create_chemical_model'
]

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
        self.W_i = nn.Linear(input_dim, hidden_size, bias=bias)\
        
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

                mol_vec = mol_vec.sum(dim=0) / a_size
                #mol_vec = mol_vec.mean(0).values
                mol_vecs.append(mol_vec)
                
        mol_vecs = T.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        
        return mol_vecs



class ChemicalHeadModel(nn.Module):
    def __init__(self, 
                 atom_fdim: int = 133,
                 bond_fdim: int = 147,
                 activation: str = 'ReLU',
                 hidden_size: int = 300,
                 bias: bool = False,
                 atom_messages: bool = False
                ):
        """
        Configures a message passing graph encoder
        
        Args:
            atom_fdim (int): feature dimensions to use for atoms, default 133
            bond_fdim (int): feature dimensions to use for bonds, default 147
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

        self.act_func = get_activation_function(activation)
        self.cached_zero_vector = T.zeros(hidden_size, requires_grad=False)

        # Input
        input_dim = atom_fdim if atom_messages else bond_fdim
        self.W_i = nn.Linear(input_dim, hidden_size, bias=bias)

        self.atom_messages = atom_messages


    def forward(self, mol_graph: BatchMolGraph):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()
        
        if self.atom_messages:
            a2a = mol_graph.get_a2a()
            input = self.W_i(f_atoms)
        else:
            a2a = None
            input = self.W_i(f_bonds)
        message = self.act_func(input)

        return (input, message, f_atoms, f_bonds, a2a, a2b, b2a, b2revb, a_scope)


def _build_expanded_context(context, a_scope):
    expanded_context = [
        T.zeros_like(context[0:1])
    ]
    for i, (a_start, a_size) in enumerate(a_scope):
        expanded_context.append(context[i:(i+1)].expand((a_size, -1)))

    return T.cat(expanded_context, 0)
    

class ChemicalMiddleModel(nn.Module):
    def __init__(self, 
                 atom_fdim: int = 133,
                 bond_fdim: int = 147,
                 activation: str = 'ReLU',
                 hidden_size: int = 300,
                 context_size: int = 300,
                 bias: bool = False,
                 dropout: float = 0.0,
                 layers_per_message: int = 1,
                 undirected: bool = False,
                 atom_messages: bool = False,
                 messages_per_pass: int = 2
                ):
        """
        Configures a message passing graph encoder
        
        Args:
            atom_fdim (int): feature dimensions to use for atoms, default 133
            bond_fdim (int): feature dimensions to use for bonds, default 147
            activation (str): 'ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', or 'ELU', default 'ReLU'
            hidden_size (int): dimension of messages, default 300
            bias (bool): include bias in internal linear layers, default False
            depth (int): number of message passing steps, default 3
            dropout (float): dropout rate on messages, default 0.0
            layers_per_message (int): linear layers included in message update function, default 1
            undirected (bool): propigate messages bidirectionally, default False
            atom_messages (bool): pass messages from atoms to atoms along bonds, default False
            messages_per_pass (int): messages passed between context updates
        """
        super().__init__()

        assert not (undirected and atom_messages), "Cannot have undirected atom messages -- sorry"

        self.dropout_layer = nn.Dropout(p=dropout)
        self.act_func = get_activation_function(activation)

        if atom_messages:
            w_h_input_size = hidden_size + bond_fdim + context_size
        else:
            w_h_input_size = hidden_size + context_size
            
        self.W_h = nn.Sequential(*([
            nn.Linear(w_h_input_size, hidden_size, bias=bias)
        ] + sum([
            [self.act_func, nn.Linear(hidden_size, hidden_size, bias=bias)] 
            for _ in range(layers_per_message-1)], []
        )))
        
        self.atom_messages = atom_messages
        self.undirected = undirected
        self.depth = messages_per_pass

    def forward(self, state, context):
        input, message, f_atoms, f_bonds, a2a, a2b, b2a, b2revb, a_scope = state

        expanded_context = _build_expanded_context(context, a_scope)
        if not self.atom_messages:
            expanded_context = expanded_context[b2a]

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2
                
            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = T.cat((
                    nei_a_message, 
                    nei_f_bonds
                ), dim=2)  # num_atoms x max_num_bonds x (hidden + bond_fdim)
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(T.cat((message, expanded_context), -1))
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        return (input, message, f_atoms, f_bonds, a2a, a2b, b2a, b2revb, a_scope)


class ChemicalTailModel(nn.Module):
    def __init__(self, 
                 atom_fdim:int = 133,
                 hidden_size: int = 300,
                 activation:str = 'ReLU', 
                 dropout:float = 0.0,
                 atom_messages:bool = False):
        super().__init__()

        self.W_o = nn.Linear(atom_fdim + hidden_size, hidden_size)

        self.dropout_layer = nn.Dropout(p=dropout)
        self.act_func = get_activation_function(activation)
        self.cached_zero_vector = T.zeros(hidden_size, requires_grad=False)

        self.atom_messages = atom_messages

    def forward(self, state):
        input, message, f_atoms, f_bonds, a2a, a2b, b2a, b2revb, a_scope = state

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

                mol_vec = mol_vec.sum(dim=0) / a_size
                #mol_vec = mol_vec.mean(0).values
                mol_vecs.append(mol_vec)
                
        mol_vecs = T.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        
        return mol_vecs
        

def create_chemical_models(atom_fdim: int = 133,
                           bond_fdim: int = 147,
                           activation: str = 'ReLU',
                           hidden_size: int = 300,
                           context_size: int = 256,
                           bias: bool = False,
                           depth: int = 3,
                           dropout: float = 0.0,
                           layers_per_message: int = 1,
                           undirected: bool = False,
                           atom_messages: bool = False,
                           messages_per_pass: int = 2):
    """
    Configures a multi-stage message passing graph encoder
    
    Args:
        atom_fdim (int): feature dimensions to use for atoms, default 133
        bond_fdim (int): feature dimensions to use for bonds, default 147
        activation (str): 'ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', or 'ELU', default 'ReLU'
        hidden_size (int): dimension of messages, default 300
        context_size (int): dimension of context to be processed
        bias (bool): include bias in internal linear layers, default False
        depth (int): number of message passing steps, default 3
        dropout (float): dropout rate on messages, default 0.0
        layers_per_message (int): linear layers included in message update function, default 1
        undirected (bool): propigate messages bidirectionally, default False
        atom_messages: pass messages from atoms to atoms along bonds, default False
        messages_per_pass (int): messages passed between context updates
    """

    head = ChemicalHeadModel(atom_fdim, bond_fdim, activation, hidden_size, bias, atom_messages)
    middle = ChemicalMiddleModel(
        atom_fdim,
        bond_fdim,
        activation,
        hidden_size,
        context_size,
        bias,
        dropout,
        layers_per_message,
        undirected,
        atom_messages,
        messages_per_pass
    )
    tail = ChemicalTailModel(atom_fdim, hidden_size, activation, dropout, atom_messages)

    return head, middle, tail