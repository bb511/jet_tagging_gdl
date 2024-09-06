# Patrick Idea

# can be run using the following command:
# python3 learn_sym_weak.py
import torch
import torch.nn as nn
import torch.optim as optim

from import_data import select_data



class MLP_weak_loss_so3(nn.Module):
    def __init__(self,nbr_const:int, nbr_feat:int, hidden:int, output:int, num_layers:int, device: str = 'cuda:0'):
        super(MLP_weak_loss_so3, self).__init__()
        self.input = nbr_const * nbr_feat
        self.hidden = hidden
        self.output = output
        self.num_layers = num_layers
        
        self.device = device
        
        self.inp = nn.Linear(self.input, self.hidden)
        self.act = nn.ReLU()
        self.nn = nn.ModuleList([nn.Linear(self.hidden, self.hidden) for _ in range(self.num_layers-1)])
        self.out = nn.Linear(self.hidden, self.output)
        
        # Initialize learnable lambda parameters
        self.l_rot_x= nn.Parameter(torch.tensor(1.0))
        self.l_rot_y= nn.Parameter(torch.tensor(1.0))
        self.l_rot_z= nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        x = torch.flatten(x, start_dim = 1, end_dim= -1) 
        x = self.inp(x)
        x = self.act(x)
        for _, lin in enumerate(self.nn):
            x = self.act(lin(x))
        out = self.out(x)
        return out

class MLP_weak_loss_lorentz(nn.Module):
    def __init__(self,nbr_const:int, nbr_feat:int, hidden:int, output:int, num_layers:int, device: str = 'cuda:0'):
        super(MLP_weak_loss_lorentz, self).__init__()
        self.input = nbr_const * nbr_feat
        self.hidden = hidden
        self.output = output
        self.num_layers = num_layers
        
        self.device = device
        
        self.inp = nn.Linear(self.input, self.hidden)
        self.act = nn.ReLU()
        self.nn = nn.ModuleList([nn.Linear(self.hidden, self.hidden) for _ in range(self.num_layers-1)])
        self.out = nn.Linear(self.hidden, self.output)
        
        # Initialize learnable lambda parameters
        self.l_rot_x= nn.Parameter(torch.tensor(0.1))
        self.l_rot_y= nn.Parameter(torch.tensor(.1))
        self.l_rot_z= nn.Parameter(torch.tensor(.1))
        self.l_boost_x= nn.Parameter(torch.tensor(.1))
        self.l_boost_y= nn.Parameter(torch.tensor(.1))
        self.l_boost_z= nn.Parameter(torch.tensor(.1))
    
    def forward(self, x):
        x = torch.flatten(x, start_dim = 1, end_dim= -1) 
        x = self.inp(x)
        x = self.act(x)
        for _, lin in enumerate(self.nn):
            x = self.act(lin(x))
        out = self.out(x)
        return out
    
def sigmoid(x):
    return 1/(1 + torch.exp(-x))

def so3_rotation_matrix(theta):
    """Function to generate the SO(3) rotation matrix given the Euler angles theta"""
    #Basis for the Lie algebra so(3)
    Lx = torch.tensor([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    Ly = torch.tensor([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    Lz = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    
    #Compute the exponential map to get to SO(3)
    rot_x = torch.exp(theta[0] * Lx)
    rot_y = torch.exp(theta[1] * Ly)
    rot_z = torch.exp(theta[2] * Lz)
    
    return rot_x, rot_y, rot_z

def lorentz_rotation_matrix(beta):
    """Function to generate the Lorentz boost matrix given the velocity beta"""
    #Basis for the Lie algebra so(3,1)
    J1 = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]]) 
    J2 = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, -1, 0, 0]])
    J3 = torch.tensor([[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])   
    K1 = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    K2 = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    K3 = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
    
    #Compute the exponential map to get to SO(3,1)
    boost_x = torch.exp(beta[0] * J1)
    boost_y = torch.exp(beta[1] * J2)
    boost_z = torch.exp(beta[2] * J3)
    rot_x = torch.exp(beta[3] * K1)
    rot_y = torch.exp(beta[4] * K2)
    rot_z = torch.exp(beta[5] * K3)
    
    return boost_x, boost_y, boost_z, rot_x, rot_y, rot_z

# Custom loss function
def custom_loss(inputs, targets, model, lambda_param = 0.01):
    """Function to compute the custom loss function"""
    outputs = model(inputs)
    ce_loss = nn.CrossEntropyLoss()(outputs, targets)
    
    thetas = torch.rand((16,3))
    loss_rot_x = 0
    loss_rot_y = 0
    loss_rot_z = 0
    for theta in thetas:
        rot_x, rot_y, rot_z = so3_rotation_matrix(theta)
        outputs_rot_x = model(inputs @ rot_x)
        outputs_rot_y = model(inputs @ rot_y)
        outputs_rot_z = model(inputs @ rot_z)
        loss_rot_x += torch.norm(outputs_rot_x - outputs)
        loss_rot_y += torch.norm(outputs_rot_y - outputs)
        loss_rot_z += torch.norm(outputs_rot_z - outputs)
    mean_loss_rot_x = loss_rot_x / thetas.shape[0]
    mean_loss_rot_y = loss_rot_y / thetas.shape[0]
    mean_loss_rot_z = loss_rot_z / thetas.shape[0]
        
    return ce_loss + torch.abs(model.l_rot_x) * mean_loss_rot_x + torch.abs(model.l_rot_y) * mean_loss_rot_y + torch.abs(model.l_rot_z) * mean_loss_rot_z - lambda_param * (model.l_rot_x + model.l_rot_y + model.l_rot_z)

def custom_loss_lorentz(inputs, targets, model, lambda_param = 0.01):
    """Function to compute the Lorentz custom loss function"""
    outputs = model(inputs)
    ce_loss = nn.CrossEntropyLoss()(outputs, targets)
    
    betas = torch.rand((64,6))
    loss_boost_x = 0
    loss_boost_y = 0
    loss_boost_z = 0
    loss_rot_x = 0
    loss_rot_y = 0
    loss_rot_z = 0
    for beta in betas:
        boost_x, boost_y, boost_z, rot_x, rot_y, rot_z = lorentz_rotation_matrix(beta)
        outputs_boost_x = model(inputs @ boost_x)
        outputs_boost_y = model(inputs @ boost_y)
        outputs_boost_z = model(inputs @ boost_z)
        outputs_rot_x = model(inputs @ rot_x)
        outputs_rot_y = model(inputs @ rot_y)
        outputs_rot_z = model(inputs @ rot_z)
        loss_boost_x += torch.norm(outputs_boost_x - outputs)
        loss_boost_y += torch.norm(outputs_boost_y - outputs)
        loss_boost_z += torch.norm(outputs_boost_z - outputs)
        loss_rot_x += torch.norm(outputs_rot_x - outputs)
        loss_rot_y += torch.norm(outputs_rot_y - outputs)
        loss_rot_z += torch.norm(outputs_rot_z - outputs)
        
    mean_loss_boost_x = loss_boost_x / betas.shape[0]
    mean_loss_boost_y = loss_boost_y / betas.shape[0]
    mean_loss_boost_z = loss_boost_z / betas.shape[0]
    mean_loss_rot_x = loss_rot_x / betas.shape[0]
    mean_loss_rot_y = loss_rot_y / betas.shape[0]
    mean_loss_rot_z = loss_rot_z / betas.shape[0]
    
    return ce_loss + torch.abs(model.l_rot_x) * mean_loss_rot_x + torch.abs(model.l_rot_y) * mean_loss_rot_y + torch.abs(model.l_rot_z) * mean_loss_rot_z + torch.abs(model.l_boost_x)*mean_loss_boost_x +torch.abs(model.l_boost_y)*mean_loss_boost_y +torch.abs(model.l_boost_z)*mean_loss_boost_z - lambda_param * (sigmoid(model.l_rot_x + model.l_rot_y + model.l_rot_z + model.l_boost_x + model.l_boost_y + model.l_boost_z))

# Training loop
def train(model, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            if model.__class__.__name__ == 'MLP_weak_loss_so3':
                loss = custom_loss(data, target, model, 0.1)
            elif model.__class__.__name__ == 'MLP_weak_loss_lorentz':
                loss = custom_loss_lorentz(data, target, model, 0.1)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                if model.__class__.__name__ == 'MLP_weak_loss_so3':
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, L rot x: {torch.abs(model.l_rot_x).item():.4f}, L rot y: {torch.abs(model.l_rot_y).item():.4f}, L rot z: {torch.abs(model.l_rot_z).item():.4f}')
                elif model.__class__.__name__ == 'MLP_weak_loss_lorentz':
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, L rot x: {torch.abs(model.l_rot_x).item():.4f}, L rot y: {torch.abs(model.l_rot_y).item():.4f}, L rot z: {torch.abs(model.l_rot_z).item():.4f}, L boost x: {torch.abs(model.l_boost_x).item():.4f}, L boost y: {torch.abs(model.l_boost_y).item():.4f}, L boost z: {torch.abs(model.l_boost_z).item():.4f}')
    return loss, model.l_rot_x, model.l_rot_y, model.l_rot_z, model

def valid(model, valid_loader):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            outputs = model(data)
            loss += custom_loss_lorentz(data, target, model, 0.1)
            accuracy += (outputs.argmax(1) == target).float()
        print(f'Accuracy {accuracy/batch_idx:.4f}, Loss: {loss.item()/batch_idx:.4f}')
    return loss.item()/batch_idx, accuracy/batch_idx 

data_name = "px_py_pz_8"
train_data, valid_data = select_data("mlp_weak_loss", data_name)

nbr_const =  8
nbr_feat = 4


train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

model = MLP_weak_loss_lorentz(nbr_const = nbr_const, 
                    nbr_feat = nbr_feat, 
                    hidden = 64, 
                    output = 5, 
                    num_layers = 5, 
                    device = 'cuda:0'
                    )

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss, l_rot_x, l_rot_y, l_rot_z = train(model, train_loader, optimizer, 10)
print(f'Final loss: {loss.item():.4f}, L rot x: {torch.exp(l_rot_x).item():.4f}, L rot y: {torch.exp(l_rot_y).item():.4f}, L rot z: {torch.exp(l_rot_z).item():.4f}')


