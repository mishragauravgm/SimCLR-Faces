import torch
from simclr import SimCLR
from net import LResNet50E_IR, LResNet


device = torch.device("cpu")
encoder = get_resnet(args.resnet, pretrained=False)
model = SimCLR(encoder, args.projection_dim, n_features)

model = model.load_state_dict(torch.load('save/checkpoint_100.tar', map_location=device))
model.to(device)
model.eval()

print(model)