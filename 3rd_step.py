# loading and importing necessary module
import glob
import random
import os
import scipy
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import itertools
from ignite.engine import Engine, Events
import ignite.distributed as idist
from torchvision.utils import save_image
import ignite
import pickle
import logging
from ignite.metrics import FID, InceptionScore
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar
import numpy as np


# defining necessary variables used for implementation of Cycle GAN.
epoch = 10
tot_epochs = 200
batchSize = 8
dataroot = 'datasets/real2live'
lr = 0.0002
decay_epoch = 25
size = 256
in_channels = 3
out_channels = 3
cuda = True
num_gpu = 4
num_cpu = 8
resume = True

#importing inceptio v3 for fid scores
device='cuda'
from torchvision.models import inception_v3
inception_model = inception_v3(pretrained=True)
# inception_model.load_state_dict(torch.load("inception_v3_google-1a9a5a14.pth"))
inception_model.to(device)
inception_model = inception_model.eval() # Evaluation mode\
inception_model.fc = nn.Identity()




# This is the matrix square root function you will be using
def matrix_sqrt(x):
    '''
    Function that takes in a matrix and returns the square root of that matrix.
    For an input matrix A, the output matrix B would be such that B @ B is the matrix A.
    Parameters:
        x: a matrix
    '''
    y = x.cpu().detach().numpy()
    y = scipy.linalg.sqrtm(y)
    return torch.Tensor(y.real, device=x.device)
#preprocess required for fid computation
def preprocess(img):
    img = torch.nn.functional.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
    return img

def get_covariance(features):
    return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))


def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):
    '''
    Function for returning the Fréchet distance between multivariate Gaussians,
    parameterized by their means and covariance matrices.
    Parameters:
        mu_x: the mean of the first Gaussian, (n_features)
        mu_y: the mean of the second Gaussian, (n_features) 
        sigma_x: the covariance matrix of the first Gaussian, (n_features, n_features)
        sigma_y: the covariance matrix of the second Gaussian, (n_features, n_features)
    '''
    fid=(mu_x-mu_y).dot(mu_x-mu_y)+torch.trace(sigma_x)+torch.trace(sigma_y)-2*torch.trace(matrix_sqrt(sigma_x@sigma_y))
    return fid







# verifying whether cuda is enable or not.
if torch.cuda.is_available() and not cuda:
    print("Integrate cuda for faster processing")

# setting up ignite loggers to log the progress
ignite.utils.manual_seed(999)
ignite.utils.setup_logger(name="ignite.distributed.auto.auto_dataloader", level=logging.WARNING)
ignite.utils.setup_logger(name="ignite.distributed.launcher.Parallel", level=logging.WARNING)


# below section of code consists of code implementation for dataset preparation and loading.

class ImageDataset(Dataset):
    def __init__(self, root, transforms=None, notaligned=False, mode='train'):
        self.transform_image = transforms.Compose(transforms)
        self.notaligned = notaligned

        self.image_list_a = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.image_list_b = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        image_a = self.transform_image(Image.open(self.image_list_a[index % len(self.image_list_a)]).convert('RGB'))

        if self.notaligned:
            image_b = self.transform_image(Image.open(self.image_list_b[random.randint(0, len(self.image_list_b) - 1)]).convert('RGB'))
        else:
            image_b = self.transform_image(Image.open(self.image_list_b[index % len(self.image_list_b)]).convert('RGB'))

        return {'A': image_a, 'B': image_b}

    def __len__(self):
        return max(len(self.image_list_a), len(self.image_list_b))


# below section of code consists of code implementation of utility helper functions for Cycle GAN

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class Lambda_LR:
    def __init__(self, tot_epochs, offset, decay_epoch):
        assert ((tot_epochs - decay_epoch) > 0), "Decay must start before the training ends!"
        self.tot_epochs = tot_epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch) / (self.tot_epochs - self.decay_epoch)


def initial_weights(m):
    cls_name = m.__class__.__name__
    if cls_name.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif cls_name.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

# below section of code consists of code implementation of Generator and Discriminator architecture


class Res_Block(nn.Module):
    """
    This class consists of code that implements the required residual blocks
    """
    def __init__(self, in_features):
        super(Res_Block, self).__init__()
        layers = [nn.ReflectionPad2d(1),
                  nn.Conv2d(in_features, in_features, 3),
                  nn.InstanceNorm2d(in_features),
                  nn.ReLU(inplace=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(in_features, in_features, 3),
                  nn.InstanceNorm2d(in_features)]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    This class consists of code that implements the required generator of Cycle GAN.
    """
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        # initial convolution block
        blocks = [nn.ReflectionPad2d(3),
                  nn.Conv2d(in_channels, 64, 7),
                  nn.InstanceNorm2d(64),
                  nn.ReLU(inplace=True)]

        # down sampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            blocks += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                       nn.InstanceNorm2d(out_features),
                       nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2

        # residual blocks
        for _ in range(9):
            blocks += [Res_Block(in_features)]

        # up sampling
        out_features = in_features//2
        for _ in range(2):
            blocks += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                       nn.InstanceNorm2d(out_features),
                       nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features//2

        # final layer
        blocks += [nn.ReflectionPad2d(3),
                   nn.Conv2d(64, out_channels, 7),
                   nn.Tanh()]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """
    This class consists of code that implements the required discriminator of Cycle GAN.
    """
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # layer of convolution blocks
        blocks = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, inplace=True)]

        blocks += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                   nn.InstanceNorm2d(128),
                   nn.LeakyReLU(0.2, inplace=True)]

        blocks += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                   nn.InstanceNorm2d(256),
                   nn.LeakyReLU(0.2, inplace=True)]

        blocks += [nn.Conv2d(256, 512, 4, padding=1),
                   nn.InstanceNorm2d(512),
                   nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        blocks += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0],-1)

# below section of code consists of code implementation of training of the Cycle GAN


def define_generators():
    global gen_A2B, gen_B2A
    gen_A2B = idist.auto_model(Generator(in_channels, out_channels))
    gen_B2A = idist.auto_model(Generator(out_channels, in_channels))


def define_discriminators():
    global disc_A, disc_B
    disc_A = idist.auto_model(Discriminator(in_channels))
    disc_B = idist.auto_model(Discriminator(out_channels))


def define_criterion():
    global cri_GAN, cri_cycle, cri_identity
    cri_GAN = torch.nn.MSELoss()
    cri_cycle = torch.nn.L1Loss()
    cri_identity = torch.nn.L1Loss()


def define_optimizers():
    global optim_G, optim_D_A, optim_D_B
    optim_G = idist.auto_optim(
        torch.optim.Adam(itertools.chain(gen_A2B.parameters(), gen_B2A.parameters()), lr=lr, betas=(0.5, 0.999)))
    optim_D_A = idist.auto_optim(torch.optim.Adam(disc_A.parameters(), lr=lr, betas=(0.5, 0.999)))
    optim_D_B = idist.auto_optim(torch.optim.Adam(disc_B.parameters(), lr=lr, betas=(0.5, 0.999)))


def define_learning_rates():
    global lr_sch_G, lr_sch_D_A, lr_sch_D_B
    lr_sch_G = idist.auto_optim(
        torch.optim.lr_scheduler.LambdaLR(optim_G, lr_lambda=Lambda_LR(tot_epochs, epoch,
                                                                       decay_epoch).step))
    lr_sch_D_A = idist.auto_optim(
        torch.optim.lr_scheduler.LambdaLR(optim_D_A, lr_lambda=Lambda_LR(tot_epochs, epoch,
                                                                         decay_epoch).step))
    lr_sch_D_B = idist.auto_optim(
        torch.optim.lr_scheduler.LambdaLR(optim_D_B, lr_lambda=Lambda_LR(tot_epochs, epoch,
                                                                         decay_epoch).step))


def define_fake_buffer():
    global fake_A_buffer, fake_B_buffer
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()


def define_dataloaders():
    global dataloader, dataloader_test
    dataloader = idist.auto_dataloader(ImageDataset(dataroot, transforms=transforms_, notaligned=True),
                                       batch_size=batchSize, shuffle=True, num_workers=num_cpu, drop_last=True)
    dataloader_test = idist.auto_dataloader(ImageDataset(dataroot, transforms=transforms_, mode='test'),
                                            batch_size=batchSize, shuffle=False, num_workers=num_cpu, drop_last=True)


define_generators()
define_discriminators()
define_criterion()
define_optimizers()
define_learning_rates()

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

input_A = Tensor(batchSize, in_channels, size, size)
input_B = Tensor(batchSize, out_channels, size, size)
target_real = Variable(Tensor(batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batchSize).fill_(0.0), requires_grad=False)

define_fake_buffer()

transforms_ = [transforms.Resize(int(size*1.12), transforms.InterpolationMode.BICUBIC),
               transforms.RandomCrop(size),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

define_dataloaders()

real_features_list_A=[]
real_features_list_B=[]
fake_features_list_A=[]
fake_features_list_B=[]
def train(engine, batch):
    
    gen_A2B.train()
    gen_B2A.train()

    disc_A.train()
    disc_B.train()

    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))

    optim_G.zero_grad()

    same_B = gen_A2B(real_B)
    loss_identity_B = cri_identity(same_B, real_B) * 5.0

    same_A = gen_B2A(real_A)
    loss_identity_A = cri_identity(same_A, real_A) * 5.0

    fake_B = gen_A2B(real_A)
    pred_fake = disc_B(fake_B)
    loss_GAN_A2B = cri_GAN(torch.squeeze(pred_fake, 1), target_real)

    fake_A = gen_B2A(real_B)
    pred_fake = disc_A(fake_A)
    loss_GAN_B2A = cri_GAN(torch.squeeze(pred_fake, 1), target_real)

    recovered_A = gen_B2A(fake_B)
    loss_cycle_ABA = cri_cycle(recovered_A, real_A) * 10.0

    recovered_B = gen_A2B(fake_A)
    loss_cycle_BAB = cri_cycle(recovered_B, real_B) * 10.0

    loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
    loss_G.backward()

    optim_G.step()

    optim_D_A.zero_grad()

    pred_real = disc_A(real_A)
    loss_D_real = cri_GAN(torch.squeeze(pred_real, 1), target_real)

    fake_A = fake_A_buffer.push_and_pop(fake_A)
    pred_fake = disc_A(fake_A.detach())
    loss_D_fake = cri_GAN(torch.squeeze(pred_fake, 1), target_fake)

    loss_D_A = (loss_D_real + loss_D_fake)*0.5
    loss_D_A.backward()

    optim_D_A.step()

    optim_D_B.zero_grad()

    pred_real = disc_B(real_B)
    loss_D_real = cri_GAN(torch.squeeze(pred_real, 1), target_real)

    fake_B = fake_B_buffer.push_and_pop(fake_B)
    pred_fake = disc_B(fake_B.detach())
    loss_D_fake = cri_GAN(torch.squeeze(pred_fake, 1), target_fake)

    loss_D_B = (loss_D_real + loss_D_fake)*0.5
    loss_D_B.backward()

    optim_D_B.step()


    #fidA2Bvalues
    real_samples_A= real_A
    real_features_A= inception_model(real_samples_A.to(device)).detach().to('cpu') # Move features to CPU
    real_features_list_A.append(real_features_A)
    fake_samples_A= preprocess(gen_A2B(real_samples_A))
    fake_features_A = inception_model(fake_samples_A.to(device)).detach().to('cpu')
    fake_features_list_A.append(fake_features_A)
    fake_features_all_A = torch.cat(fake_features_list_A)
    real_features_all_A = torch.cat(real_features_list_A)
    mu_fake_A = fake_features_all_A.mean(0)
    mu_real_A= real_features_all_A.mean(0)
    sigma_fake_A = get_covariance(fake_features_all_A)
    sigma_real_A = get_covariance(real_features_all_A)

    #fidB2Avalues
    real_samples_B= real_B
    real_features_B= inception_model(real_samples_B.to(device)).detach().to('cpu') # Move features to CPU
    real_features_list_B.append(real_features_B)
    fake_samples_B= preprocess(gen_B2A(real_samples_B))
    fake_features_B = inception_model(fake_samples_B.to(device)).detach().to('cpu')
    fake_features_list_B.append(fake_features_B)
    fake_features_all_B = torch.cat(fake_features_list_B)
    real_features_all_B = torch.cat(real_features_list_B)
    mu_fake_B = fake_features_all_B.mean(0)
    mu_real_B= real_features_all_B.mean(0)
    sigma_fake_B = get_covariance(fake_features_all_B)
    sigma_real_B = get_covariance(real_features_all_B)

    #fidA2B
    fid_A2B=frechet_distance(mu_real_B, mu_fake_A, sigma_real_B, sigma_fake_A).item()
    #fidB2A
    fid_B2A=frechet_distance(mu_real_A, mu_fake_B, sigma_real_A, sigma_fake_B).item()

    return {
        'loss_G': loss_G.item(),
        'loss_G_identity' : loss_identity_A.item() + loss_identity_B.item(),
        'loss_G_GAN': loss_GAN_A2B.item() + loss_GAN_B2A.item(),
        'loss_G_cycle': loss_cycle_ABA.item() + loss_cycle_BAB.item(),
        'loss_D': loss_D_A.item() + loss_D_A.item(),
        'real_A': real_A,
        'real_B': real_B,
        'fake_A': fake_A,
        'fake_B': fake_B,
        'recovered_A': recovered_A,
        'recovered_B': recovered_B,
        'fid_A2B':fid_A2B,
        'fid_B2A':fid_B2A
    }


trainer = Engine(train)


@trainer.on(Events.STARTED)
def init_weights():
    if not resume:
        gen_A2B.apply(initial_weights)
        gen_B2A.apply(initial_weights)
        disc_A.apply(initial_weights)
        disc_B.apply(initial_weights)
    else:
        x = torch.load(f'result/{epoch}.pth')
        gen_A2B.load_state_dict(x['netG_A2B'])
        gen_B2A.load_state_dict(x['netG_B2A'])
        disc_A.load_state_dict(x['netD_A'])
        disc_B.load_state_dict(x['netD_B'])


G_losses = []
G_losses_identity = []
G_losses_GAN = []
G_losses_cycle = []
D_losses = []
fid_A2B=[]
fid_B2A=[]


@trainer.on(Events.ITERATION_COMPLETED)
def store_losses(engine):
    o = engine.state.output
    G_losses.append(o["loss_G"])
    D_losses.append(o["loss_D"])
    G_losses_identity.append(o["loss_G_identity"])
    G_losses_GAN.append(o["loss_G_GAN"])
    G_losses_cycle.append(o["loss_G_cycle"])
    fid_A2B.append(o["fid_A2B"])
    fid_B2A.append(o["fid_B2A"])

    
    with open('G_losses.pkl', 'wb') as f:
        pickle.dump(G_losses, f)

    with open('D_losses.pkl', 'wb') as f:
        pickle.dump(D_losses, f)
    
    with open('G_losses_identity.pkl', 'wb') as f:
        pickle.dump(G_losses_identity, f)

    with open('G_losses_GAN.pkl', 'wb') as f:
        pickle.dump(G_losses_GAN, f)
    
    with open('G_losses_cycle.pkl', 'wb') as f:
        pickle.dump(G_losses_cycle, f)

    with open('fid_A2B', 'wb') as f:
        pickle.dump(fid_A2B, f)
    
    with open('fid_B2A','wb') as f:
        pickle.dump(fid_A2B,f)



fid_metric = FID(device=idist.device())
is_metric = InceptionScore(device=idist.device(), output_transform=lambda x: x[0])

fid_metric_1 = FID(device=idist.device())
is_metric_1 = InceptionScore(device=idist.device(), output_transform=lambda x: x[0])

def eval(engine, batch):
    with torch.no_grad():
        gen_B2A.eval()
        input_A = Tensor(batchSize, in_channels, size, size)
        input_B = Tensor(batchSize, out_channels, size, size)
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        fake_A = 0.5*(gen_B2A(real_B).data + 1.0)
        recovered_B = gen_A2B(fake_A)
        A_output_imgs = []
        A_output_imgs.extend(real_B[:5])
        A_output_imgs.extend(fake_A[:5])
        A_output_imgs.extend(recovered_B[:5])
        save_image(A_output_imgs, 'result/A/%04d.png' % (engine.state.iteration+1))
        return fake_A, real_A

def eval_step_1(engine, batch):
    with torch.no_grad():
        gen_A2B.eval()
        input_A = Tensor(batchSize, in_channels, size, size)
        input_B = Tensor(batchSize, out_channels, size, size)
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        fake_B = 0.5*(gen_A2B(real_A).data + 1.0)
        recovered_A = gen_B2A(fake_B)
        B_output_imgs = []
        B_output_imgs.extend(real_A[:5])
        B_output_imgs.extend(fake_B[:5])
        B_output_imgs.extend(recovered_A[:5])
        save_image(B_output_imgs, 'result/B/%04d.png' % (engine.state.iteration+1))
        return fake_B, real_B


evaluator = Engine(eval)
evaluator1 = Engine(eval_step_1)

fid_metric.attach(evaluator, "fid")
is_metric.attach(evaluator, "is")

fid_metric_1.attach(evaluator1, "fid")
is_metric_1.attach(evaluator1, "is")


fid_values = []
is_values = []

fid_values_1 = []
is_values_1 = []


@trainer.on(Events.EPOCH_COMPLETED)
def logging_results(engine):
    lr_sch_G.step()
    lr_sch_D_A.step()
    lr_sch_D_B.step()
    evaluator.run(dataloader_test,max_epochs=1)
    metrics = evaluator.state.metrics
    fid_score = metrics['fid']
    is_score = metrics['is']
    fid_values.append(fid_score)
    is_values.append(is_score)
    
    print(f"Epoch [{engine.state.epoch}] Metric Scores")
    print(f"*   FID : {fid_score:4f}")
    print(f"*    IS : {is_score:4f}")
    
    evaluator1.run(dataloader_test,max_epochs=1)
    metrics = evaluator1.state.metrics
    fid_score = metrics['fid']
    is_score = metrics['is']
    fid_values_1.append(fid_score)
    is_values_1.append(is_score)
    
    print(f"Epoch [{engine.state.epoch}] Metric Scores")
    print(f"*   FID : {fid_score:4f}")
    print(f"*    IS : {is_score:4f}")
    
    with open('fid.pkl', 'wb') as f:
        pickle.dump(fid_values, f)

    with open('is.pkl', 'wb') as f:
        pickle.dump(is_values, f)
    
    with open('fid_1.pkl', 'wb') as f:
        pickle.dump(fid_values_1, f)

    with open('is_1.pkl', 'wb') as f:
        pickle.dump(is_values_1, f)
    
    if engine.state.epoch % 5 == 0:
            weights = {}
            weights['netG_A2B'] = gen_A2B.state_dict()
            weights['netG_B2A'] = gen_B2A.state_dict()
            weights['netD_A'] = disc_A.state_dict()
            weights['netD_B'] = disc_B.state_dict()
            weights['epoch'] = engine.state.epoch
            torch.save(weights, f'result/{engine.state.epoch}.pth')
    
    o = engine.state.output    
    print(f"fid_A2B is :{o[fid_A2B]}")    
    print(f"fid_B2A is :{o[fid_B2A]}")
    A_output_imgs = []
    A_output_imgs.extend(o['real_A'])
    A_output_imgs.extend(o['fake_B'])
    A_output_imgs.extend(o['recovered_A'])
    B_output_imgs = []
    B_output_imgs.extend(o['real_B'])
    B_output_imgs.extend(o['fake_A'])
    B_output_imgs.extend(o['recovered_B'])
    save_image(A_output_imgs, f'images_A_{engine.state.epoch}.png')
    save_image(B_output_imgs, f'images_B_{engine.state.epoch}.png')


# defining running average
RunningAverage(output_transform=lambda x: x["loss_G"]).attach(trainer, 'loss_G')
RunningAverage(output_transform=lambda x: x["loss_D"]).attach(trainer, 'loss_D')

# defining the progress bar
ProgressBar().attach(trainer, metric_names=['loss_G','loss_D'])
ProgressBar().attach(evaluator)
ProgressBar().attach(evaluator1)

# def training(*args):
trainer.run(dataloader, max_epochs=50)