import torch
from nngeometry.generator import Jacobian
from nngeometry.object import FMatDense

class LinearizationProbe(object):
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.buffer = dict()

    def get_signs(self):
        handles = []
        signs = []
        def hook(m, input, output):
            signs.append(torch.sign(output).view(output.size(0), -1))
        for m in self.model.children():
            print(type(m))
            if type(m) == torch.nn.ReLU:
                handles.append(m.register_forward_hook(hook))
        with torch.no_grad():
            for x, *_ in iter(self.dataloader):
                self.model(x)
                # break # limit to 1 minibatch

        for h in handles:
            h.remove()
        return torch.cat(signs, dim=1).bool()

    def sign_similarity(self, signs1, signs2):
        return (signs1 == signs2).float().mean()

    def get_ntk(self):
        generator = Jacobian(model=self.model, n_output=1)
        K = FMatDense(generator, examples=self.dataloader)
        return K.get_dense_tensor()

    def kernel_alignment(self, K1, K2, centered=False):
        if centered:
            K1 = K1 - K1.mean(dim=(0, 1), keepdim=True)
            K1 = K1 - K1.mean(dim=(2, 3), keepdim=True)
            K2 = K2 - K2.mean(dim=(0, 1), keepdim=True)
            K2 = K2 - K2.mean(dim=(2, 3), keepdim=True)
        ka = (K1*K2).sum() / torch.norm(K1) / torch.norm(K2)
        return ka

    def get_last_layer_representation(self):
        # extract last linear layer
        for n_, m in self.model.named_modules():
            if type(m) == torch.nn.Linear:
                last_linear = m
        
        representation = []
        def hook(m, input, output):
            representation.append(input[0])
        handle = last_linear.register_forward_hook(hook)
        with torch.no_grad():
            for x, *_ in iter(self.dataloader):
                self.model(x)
                # break # limit to 1 minibatch
        handle.remove()
        return torch.cat(representation, dim=1)

    def representation_alignment(self, r1, r2, centered=False):
        k1 = torch.mm(r1.t(), r1)
        k2 = torch.mm(r2.t(), r2)
        return self.kernel_alignment(k1, k2, centered=centered)