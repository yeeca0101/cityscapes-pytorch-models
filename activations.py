import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# from activation.GLUs_v2 import *
# from util.utils import exclude_from_activations

def exclude_from_activations(cls):
    """
    Decorator to mark classes to be excluded from activation functions.
    """
    cls._exclude_from_activations = True  # Set an attribute to mark the class
    return cls



# final methods
class SwishT(nn.Module):
    def __init__(self, beta_init=1.0, alpha=0.1,requires_grad=True):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  # Learnable parameter
        self.alpha = alpha  # Could also be made learnable if desired

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x) + self.alpha * torch.tanh(x)

# variants of SwishT

class SwishT_A(nn.Module):
    def __init__(self, beta_init=1.0, alpha=0.1,requires_grad=True):
        super().__init__()
        # self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  
        # base not used beta : swish-1 + tanh
        self.alpha = alpha  

    def backward_(self,x):
        fx = self.forward(x)
        return torch.sigmoid(x)*(x+self.alpha+1-fx)

    def forward(self, x):
        # simplify by x*torch.sigmoid(x)+self.alpha*torch.tanh(x/2)
        return torch.sigmoid(x)*(x+2*self.alpha)-self.alpha

class SwishT_B(nn.Module):
    def __init__(self, beta_init=1.0, alpha=0.1,requires_grad=True):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  
        self.alpha = alpha  

    def backward_(self,x):
        fx = self.forward(x)
        return torch.sigmoid(self.beta*x)*(self.beta*(x+self.alpha-fx)+1)

    def forward(self, x):
        return torch.sigmoid(self.beta*x)*(x+2*self.alpha)-self.alpha

class SwishT_C(nn.Module):
    def __init__(self, beta_init=1.0, alpha=0.1,requires_grad=True):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  
        self.alpha = alpha  

    def backward_(self,x):
        fx = self.forward(x)
        return torch.sigmoid(self.beta*x)*(self.beta*x+self.alpha+1-self.beta*fx)

    def forward(self, x):
        return torch.sigmoid(self.beta*x)*(x+2*self.alpha/self.beta)-self.alpha/self.beta
    

# for comparsion
# [ACON_C, Pserf, ErfAct, SMU, GELU, SiLU, Mish, Swish]
class SMU(nn.Module):
    '''
    Implementation of SMU activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha: hyper parameter
    References:
        - See related paper:
        https://arxiv.org/abs/2111.04682
    '''
    def __init__(self, alpha = 0.25, mu = 1.0):
        '''
        Initialization.
        INPUT:
            - alpha: hyper parameter
            aplha is initialized with zero value by default
        '''
        super().__init__()
        self.alpha = alpha
        # initialize mu
        self.mu = nn.Parameter(torch.tensor([mu]),requires_grad=True)
       
    def forward(self, x):
        return ((1+self.alpha)*x + (1-self.alpha)*x*torch.erf(self.mu*(1-self.alpha)*x))/2

class GELU(nn.GELU):
    def __init__(self, approximate: str = 'none') -> None:
        super().__init__(approximate)
    @property
    def __name__(self):
        return self.__class__.__name__

class SiLU(nn.SiLU):
    '''
        x*sigmoid(x) as same as Swish-1
        for reinforcement learning
    '''
    def __init__(self, inplace: bool = False):
        super().__init__(inplace)

    @property
    def __name__(self):
        return self.__class__.__name__

class Mish(nn.Mish):
    '''
        x*Tanh(Softplus(x)) , Softplus = x*tanh(ln(1+exp(x))
    '''
    def __init__(self, inplace: bool = False):
        super().__init__(inplace)

    @property
    def __name__(self):
        return self.__class__.__name__
    
class Swish(nn.Module):
    '''
        x*sigmoid(b*x)  ,b = trainable parameter
    '''
    def __init__(self, beta_init=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=True)

    def forward(self,x):
        return x*torch.sigmoid(self.beta*x)
    

@exclude_from_activations
class ErfAct(nn.Module):
    def __init__(self, alpha=0.75, beta=0.75):
        super(ErfAct, self).__init__()
        self.alpha = torch.nn.Parameter(torch.tensor([alpha], dtype=torch.float32))
        self.beta = torch.nn.Parameter(torch.tensor([beta], dtype=torch.float32))
 
    def forward(self, x):
        # return ErfActFunction.apply(x, self.alpha, self.beta)
        return x * torch.erf(self.alpha * torch.exp(self.beta * x))
    
@exclude_from_activations
class Pserf(nn.Module):
    def __init__(self, gamma=1.25, delta=0.85):
        super(Pserf, self).__init__()
        self.gamma = nn.Parameter(torch.tensor([gamma]))
        self.delta = nn.Parameter(torch.tensor([delta]))

    def forward(self, x):
        # return PserfFunction.apply(input, self.gamma, self.delta)
        return x * torch.erf(self.gamma * torch.log1p(torch.exp(self.delta * x)))
    
@exclude_from_activations
class ACON_C(nn.Module):
    def __init__(self, p1=1.0, p2=0.0, beta=1.0):
        super(ACON_C, self).__init__()
        self.p1 = nn.Parameter(torch.tensor([p1]))
        self.p2 = nn.Parameter(torch.tensor([p2]))
        self.beta = nn.Parameter(torch.tensor([beta]))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * (self.p1 - self.p2) * x) + self.p2 * x


# for appendix
@exclude_from_activations
class SiLUT(SwishT):
    def __init__(self,):
        super().__init__(beta_init=1.0, alpha=0.1,requires_grad=False)

    @property
    def __name__(self):
        return self.__class__.__name__

@exclude_from_activations
class SliuT(nn.Module):
    '''scaled Swish using tanh'''
    def __init__(self, beta_init=1.0, alpha=0.1,requires_grad=True):
        super(SliuT, self).__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  # Learnable parameter
        self.alpha = alpha  # Could also be made learnable if desired

    def forward(self, x):
        return x*torch.sigmoid(x)+self.alpha*torch.tanh(self.beta*x)

@exclude_from_activations
class ASN(nn.Module):
    'Adaptive Squared Non-Linearity'
    def __init__(self, beta_init=1.0, alpha=0.1):
        super(ASN, self).__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=True)  # Learnable parameter
        self.alpha = alpha  # Could also be made learnable if desired

    def forward(self, x):
        sig_part = torch.sigmoid(self.beta * x)
        sqr_part = torch.pow(x, 2)
        y = x * sig_part + self.alpha * sqr_part
        y = torch.clamp(y, -5, 5)
        # x = x * torch.sigmoid(self.beta * x) + self.alpha * torch.pow(x, 2)
        # print(x)
        return y



def get_activations(return_type='dict'):
    # 전역 네임스페이스에서 nn.Module을 상속받는 클래스 찾기, 단 _exclude_from_activations 속성이 없는 클래스만
    module_subclasses = {name: cls for name, cls in globals().items()
                         if isinstance(cls, type) 
                         and (issubclass(cls, nn.Module) )
                         and cls is not nn.Module
                         and not getattr(cls, '_exclude_from_activations', False)}  # Check for the exclusion marker

    # 인스턴스 생성
    instances = {name: cls() for name, cls in module_subclasses.items()}

    # 반환 타입에 따라 딕셔너리 혹은 리스트 반환
    if return_type == 'dict':
        return instances
    else:
        return list(instances.values())

if __name__ == '__main__':
    auto_instances_dict = get_activations('dict')
    print(auto_instances_dict)
    act = auto_instances_dict[0]
    print(type(act))
    x = torch.linspace(-3,3,50)
    with torch.no_grad():
        print(act(x).shape)
