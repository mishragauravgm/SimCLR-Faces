import os
import numpy as np
import torch
import torchvision
import argparse
import pandas as pd

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

from simclr.modules.tfiwDataset import TFIWDataset
from net import LResNet50E_IR, LResNet

from model import load_optimizer, save_model
from utils import yaml_config_hook
from PIL import Image

#from torchsummary import summary

torch.cuda.empty_cache()

def choose_another_member(args,fids): #take in FID return tensor equivalent of another family member
    dir = '/home/mishra.g/thesis/data/T-1/all_images/train_list.csv'
    filelist = pd.read_csv(dir)
    size = (args.batch_size, 3, args.image_size[0], args.image_size[1])
    x_j = torch.empty(size)

    convert_tensor = torchvision.transforms.ToTensor()
    resize_tensor = torchvision.transforms.Resize(size=(args.image_size[0],args.image_size[1]))
    x_j = torch.empty(size)

    for index, i in enumerate(fids):
        image = convert_tensor(resize_tensor(Image.open(filelist[filelist.iloc[:,1]==int(i)].sample().iloc[0,0])))
        x_j[index,:,:,:] = image
    return x_j

def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, ((x_i, x_j), fid) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)

        x_j = choose_another_member(args, fid)
        x_j = x_j.cuda(non_blocking=True)
        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)
        loss = criterion(z_i, z_j)

        loss.backward()
        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch


def inference(args, model):
    model.load_state_dict(torch.load(args.saved_model_path, map_location=args.device))
    model.eval()
    img_dir = '/home/mishra.g/thesis/data/T-1/all_images/test'
    img_names = os.listdir(img_dir)
    embeddings_df = pd.DataFrame(columns = ['embeddings','family_id']);
    labels = []
    embeddings = []
    for index,i in enumerate(img_names):
        if((index+1)%1000==0):
            print(f"Processed {index+1}/{len(img_names)} images")

        if(i[-3:]=='jpg'):
            image = Image.open(os.path.join(img_dir, i))
            #image = image.resize(1,3,108,124)
            labels.extend([int(i[1:5])])
            if type(image)!=None:
                transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize(size=args.image_size),
                        torchvision.transforms.ToTensor()
                    ])
                        
                    #)
                image = transform(image)
            image = image[None,:,:,:]
            embedding = model.encoder(image)
    
            embeddings_df.loc[index]= [embedding[0].tolist(), int(i[1:5])]
    
    print(f"Number of embeddings is {len(embeddings_df)}");
    
    embeddings_df.to_csv('embeddings_test.csv',  index = False)
    
def main(gpu, args):
    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )

    elif args.dataset == "TFIW":
        train_dataset = TFIWDataset(
            args.dataset_dir,
            transform = TransformsSimCLR(size=args.image_size),
        )

    else:
        raise NotImplementedError

    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer
    
    # initialize model
    model = SimCLR(encoder, args.projection_dim, n_features)
    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
        )
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)

    if args.task == "train":
        print("Starting training")
        # optimizer / loss
        optimizer, scheduler = load_optimizer(args, model)
        criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)

        # DDP / DP
        if args.dataparallel:
            model = convert_model(model)
            model = DataParallel(model)
        else:
            if args.nodes > 1:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = DDP(model, device_ids=[gpu])

        model = model.to(args.device)

        writer = None
        if args.nr == 0:
            writer = SummaryWriter()

        args.global_step = 0
        args.current_epoch = 0
        for epoch in range(args.start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            lr = optimizer.param_groups[0]["lr"]
            loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)

            if args.nr == 0 and scheduler:
                scheduler.step()

            if args.nr == 0 and epoch % 10 == 0:
                save_model(args, model, optimizer)

            if args.nr == 0:
                writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
                writer.add_scalar("Misc/learning_rate", lr, epoch)
                print(
                    f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
                )
                args.current_epoch += 1
        ## end training
        save_model(args, model, optimizer)

    else:
        print("Starting inference") #inference
        inference(args, model);

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    if args.nodes > 1:
        print(
            f"Training with {args.nodes} nodes, waiting until all nodes join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)
