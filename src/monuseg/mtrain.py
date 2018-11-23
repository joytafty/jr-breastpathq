import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as vis_tf
from torch.autograd import Variable
from .mdatasets import *
from .mmodel import *
from .mutils import visualize_learning
import tqdm
import numpy as np
import yaml
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def weights_init(model):
    if isinstance(model, nn.Conv2d):
        torch.nn.init.xavier_uniform(model.weight, gain=np.sqrt(2.0))
        torch.nn.init.constant(model.bias, 0.1)


def train_unet(model, epoch, train_dataloader, optimizer, criterion):
    model.train()
    for batch_idx, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        data, target = Variable(data["image"]), Variable(data["mask"])
        optimizer.zero_grad()
        output = model.forward(data.float())
        loss = criterion(output.float(), target.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_dataloader.dataset),
                       100. * batch_idx / len(train_dataloader), loss.data[0]))


def test_unet(model, val_dataloader, criterion):
    model.eval()
    test_loss = 0
    for data in tqdm(val_dataloader):
        data, target = Variable(data["image"], volatile=True), Variable(data["mask"])
        output = model(data.float())
        # print(output.data[0])
        test_loss += criterion(output.float(), target.float()).data[0] # sum up batch loss
    test_loss /= len(val_dataloader.dataset)
    print("Average Loss: ", test_loss)


with open('./mconfig.yml', 'r') as f:
    config = yaml.load(f)

## Setting up data loader
data_config = config['data']
crop_size = data_config['crop_size']
crop_padding =  data_config['crop_padding']

transformations = vis_tf.Compose([
    vis_tf.RandomHorizontalFlip(),
    vis_tf.RandomVerticalFlip(),
    vis_tf.RandomCrop(size=crop_size, padding=crop_padding),
    vis_tf.ToTensor()
])

image_path = config['data']['image_path']
mask_path = config['data']['mask_path']
train_ds = NuclearMaskedDataSet(
    image_path=os.path.join(image_path, "train"),
    mask_path=os.path.join(mask_path, "train"),
    transforms=transformations
)

val_ds = NuclearMaskedDataSet(
    image_path=os.path.join(image_path, "validation"),
    mask_path=os.path.join(mask_path, "validation"),
    transforms=None
)

test_ds = NuclearMaskedDataSet(
    image_path=os.path.join(image_path, "test"),
    mask_path=os.path.join(mask_path, "test"),
    transforms=None
)

## Setting up model
model_config = config['model']
n_down = model_config['n_down']
n_up = model_config['n_up']
model_instance = UNet(n_down, n_up)

train_config = config['training']
batch_size = train_config['batch_size']
lr = train_config['lr']
epoches = train_config['epoches']
checkpoint_interval = train_config['checkpoint_interval']
optimizer = optim.Adam(model_instance.parameters(), lr=lr)
criterion = nn.BCELoss()

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4)


trained_model_path = train_config['model_output_path']
if not os.path.isdir(trained_model_path):
    os.makedirs(trained_model_path)

for epoch in tqdm(epoches):
    if np.mod(epoch, checkpoint_interval) == 0:
        torch.save(
            model_instance.state_dict(),
            os.path.join(trained_model_path, f"monuseg_unet33_{epoch+base_epoch}ep.pth")
        )
        test_unet(model=model_instance, val_dataloader=val_loader)
        visualize_learning(model=model_instance, dataloader=val_loader)
    train_unet(model=model_instance, epoch=epoch, train_dataloader=train_loader)



