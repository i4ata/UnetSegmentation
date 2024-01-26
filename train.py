"""Script to train a Unet"""

import torch
import albumentations as A

from unet import Unet
from custom_unet import CustomUnet
from dataset import SegmentationDataset

import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image_height',   type=int,   default=320,                    help='Input image height')
    parser.add_argument('--image_width',    type=int,   default=320,                    help='Input image weight')
    parser.add_argument('--image_channels', type=int,   default=3,                      help='Input image channels')
    parser.add_argument('--lr',             type=float, default=1e-3,                   help='Optimizer learning rate')
    parser.add_argument('--encoder_depth',  type=int,   default=5,                      help='Depth of the Unet')
    parser.add_argument('--augmentation',   type=bool,  default=True,                   help='Include data augmentation for training')
    parser.add_argument('--epochs',         type=int,   default=25,                     help='Number of training epochs')
    parser.add_argument('--train_size',     type=float, default=.8,                     help='Proportion of data used for training')
    parser.add_argument('--batch_size',     type=int,   default=32,                     help='Model batch size')
    parser.add_argument('--model_name',     type=str,   default='unet',                 help='Name of the model')
    parser.add_argument('--log_dir',        type=str,   default='runs',                 help='Where to load the model training')
    parser.add_argument('--device',         type=str,   default='cuda',                 help='Device on which to train the model', choices=('cpu', 'cuda'))
    parser.add_argument('--patience',       type=int,   default=2,                      help='Early stopping patience')
    
    parser.add_argument('--use_custom',     type=bool,  default=False,                  help='Whether to use the custom implementation or the pretrained one')

    # Custom Unet args
    parser.add_argument('--start_channels', type=int,   default=16,                     help='Number of starting channels for the custom implementation') 

    # The pretrained implementation args
    parser.add_argument('--encoder_name',   type=str,   default='timm-efficientnet-b0', help='Unet encoder')
    parser.add_argument('--pretrained',     type=bool,  default=True,                   help='Whether to use a pretrained model')
    
    return parser.parse_args()

if __name__ == '__main__':

    random_state = 0
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)

    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('Device set to cuda but no cuda found. Defaulting to cpu')
        args.device = 'cpu'
    image_size = (args.image_height, args.image_width)
    
    # Training data augmentation
    TRAIN_TRANSFORM = A.Compose(
        transforms=[
            A.Resize(*image_size),
            A.HorizontalFlip(p=.5),
            A.VerticalFlip(p=.5)
        ], 
        is_check_shapes=False
    )

    # Testing data augmentation
    TEST_TRANSFORM = A.Compose(
        transforms=[
            A.Resize(*image_size) 
        ], 
        is_check_shapes=False
    )

    dataset = SegmentationDataset()
    dataset.split(train_size=args.train_size, 
                  train_transform=TRAIN_TRANSFORM if args.augmentation else TEST_TRANSFORM,
                  test_transform=TEST_TRANSFORM,
                  random_state=random_state)
    dataset.get_dataloaders(batch_size=args.batch_size)

    if args.use_custom:
        model = CustomUnet(name=args.model_name,
                           image_size=image_size,
                           in_channels=args.image_channels,
                           start_channels=args.start_channels,
                           encoder_depth=args.encoder_depth,
                           device=args.device)
    else:
        model = Unet(name=args.model_name, 
                    image_size=image_size, 
                    encoder_name=args.encoder_name,
                    pretrained=args.pretrained,
                    in_channels=args.image_channels,
                    encoder_depth=args.encoder_depth,
                    device=args.device)
        
    model.configure_optimizers(lr=args.lr, patience=args.patience)
    
    model.train_model(dataset.train_dataloader, 
                      dataset.test_dataloader, 
                      epochs=args.epochs, 
                      log_dir=args.log_dir)