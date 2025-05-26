'''
this file contains the hybrid NN for the actor and the FFNN 
for the critic 

Total parameters: 60887
Trainable parameters: 60887
'''
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

class Attention_module_actor(torch.nn.Module):
    def __init__(self):
        super(Attention_module_actor, self).__init__()
        self.node_size  = 32
        self.lin_hid    = 100
        self.out_dim    = 100
        self.f_in       = 10            #features in
        self.N          = 14            #number of nodes
        self.n_heads    = 3
        self.out_ac1    = 9
        self.out_ac2    = 8
        
        # Projection
        self.proj_shape = (self.f_in, self.n_heads*self.node_size)
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)

        # Normalization
        self.node_shape = (self.n_heads, self.N, self.node_size)
        self.k_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)

        #Compatibility function
        self.k_lin = nn.Linear(self.node_size,self.N)
        self.q_lin = nn.Linear(self.node_size,self.N)
        self.a_lin = nn.Linear(self.N,self.N)

        self.linear1 = nn.Linear(self.n_heads * self.node_size, self.node_size)
        self.norm1 = nn.LayerNorm([self.N,self.node_size], elementwise_affine=False)
        self.linear2 = nn.Linear(self.node_size, self.out_dim)

        # Discrete Actor
        self.layer4 = nn.Linear(self.out_dim, self.lin_hid)
        self.layer5 = nn.Linear(self.lin_hid ,self.lin_hid)
        self.layer6 = nn.Linear(self.lin_hid ,self.out_ac1)
        # Continuous Actor
        self.layer7 = nn.Linear(self.out_dim, self.lin_hid)
        self.layer8 = nn.Linear(self.lin_hid ,self.lin_hid)
        self.layer9 = nn.Linear(self.lin_hid , self.out_ac2)

    def forward(self,x, mask_vec):
        if isinstance(mask_vec, np.ndarray):
            mask_vec = torch.tensor(mask_vec, dtype=torch.bool)      

        K = rearrange(self.k_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        K = self.k_norm(K) 
        Q = rearrange(self.q_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        Q = self.q_norm(Q) 
        V = rearrange(self.v_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        V = self.v_norm(V) 

        A = torch.nn.functional.elu(self.q_lin(Q) + self.k_lin(K)) #D
        A = self.a_lin(A)
        A = torch.nn.functional.softmax(A,dim=3) 
        #print(A)

        E = torch.einsum('bhfc,bhcd->bhfd',A,V) #F  contract the attention with V tensor+
        E = rearrange(E, 'b head n d -> b n (head d)')
        E = self.linear1(E)
        E = torch.relu(E)
        E = self.norm1(E)
        E = E.max(dim=1)[0]
        y = self.linear2(E)
        y = torch.nn.functional.elu(y) # final layer that output the q values
        
        # Discrete Actor
        activation3 = torch.tanh(self.layer4(y))
        activation4 = torch.tanh(self.layer5(activation3))
        activation5 = torch.tanh(self.layer6(activation4))
        logits = torch.where(mask_vec, activation5, torch.tensor(-1e+8))
        output1 = logits #F.softmax(logits,dim=0) 

        # Continuous Actor
        activation6 = torch.tanh(self.layer7(y))
        activation7 = torch.tanh(self.layer8(activation6))
        output2 = torch.sigmoid(self.layer9(activation7))
        return output1, output2
        
    
class Attention_module_critic(torch.nn.Module):
    def __init__(self):
        super(Attention_module_critic, self).__init__()
        self.node_size  = 32
        self.lin_hid    = 100
        self.out_dim    = 100
        self.f_in       = 10            #features in
        self.N          = 14            #number of nodes
        self.n_heads    = 3
        self.out_dim    = 1
        
        # Projection
        self.proj_shape = (self.f_in, self.n_heads*self.node_size)
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)

        # Normalization
        self.node_shape = (self.n_heads, self.N, self.node_size)
        self.k_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)

        #Compatibility function
        self.k_lin = nn.Linear(self.node_size,self.N)
        self.q_lin = nn.Linear(self.node_size,self.N)
        self.a_lin = nn.Linear(self.N,self.N)

        self.linear1 = nn.Linear(self.n_heads * self.node_size, self.node_size)
        self.norm1 = nn.LayerNorm([self.N,self.node_size], elementwise_affine=False)
        self.linear2 = nn.Linear(self.node_size, self.out_dim)

        # Critic
        self.layer1 = nn.Linear(self.out_dim,self.lin_hid)
        self.layer2 = nn.Linear(self.lin_hid,self.lin_hid)
        self.layer3 = nn.Linear(self.lin_hid, self.out_dim)

    def forward(self,x):
        K = rearrange(self.k_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        K = self.k_norm(K) 
        Q = rearrange(self.q_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        Q = self.q_norm(Q) 
        V = rearrange(self.v_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        V = self.v_norm(V) 

        A = torch.nn.functional.elu(self.q_lin(Q) + self.k_lin(K)) #D
        A = self.a_lin(A)
        A = torch.nn.functional.softmax(A,dim=3) 

        E = torch.einsum('bhfc,bhcd->bhfd',A,V) #F  contract the attention with V tensor+
        E = rearrange(E, 'b head n d -> b n (head d)')
        E = self.linear1(E)
        E = torch.relu(E)
        E = self.norm1(E)
        E = E.max(dim=1)[0]
        y = self.linear2(E)
        y = torch.nn.functional.elu(y) # final layer that output the q values
        
        # Critic
        activation6 = torch.tanh(self.layer1(y))
        activation7 = torch.tanh(self.layer2(activation6))
        output = torch.sigmoid(self.layer3(activation7))
        return output
        