from resnet18 import ResNet18
from PIL import Image
from torchvision import transforms
from utils import labels_to_fen, fen_to_labels
from comet_ml import Experiment
import numpy as np
import torch
import os

ood_data_path = 'dataset/ood'

def load_model(save_path: str):
    """
    Load a ResNet model from a saved state dictionary.
        
    Args:
        save_path (str): Path to the saved state_dict
    """
    model = ResNet18(num_classes=13)
    model.load_state_dict(torch.load(save_path,map_location='cpu'))
    return model


def get_example(file_name: str, has_labels: bool=True):
    """
    Function that takes an image from the ood data folder and returns the image
    data as a tensor. If the image has the FEN notationfor file name, the labels
    will be returned in addition to the image data. 
        
    Args:
        file_name (str): Name of the file to open
        has_labels (bool): If the file name is the FEN notation
    """

    image = Image.open(os.path.join(ood_data_path,file_name))
    image = image.resize((400,400), Image.Resampling.LANCZOS)
    if has_labels:
        labels = fen_to_labels(file_name.split('.')[0])
    convert_tensor = transforms.ToTensor()
    features = convert_tensor(image)[:3,:,:]
    if not has_labels:
        return features
    return features, labels


def split_example(features: torch.Tensor, labels=None):
    """
     Splits the (3, 400, 400) inputed board image tensor in a (64, 3, 50, 50) tensor representing the 
     64 individual squares on the board. If the labels are passed they will also be split and returned
     as a (64,) tensor.
        
    Args:
        features (tensor): Image data
        labels (None or tensor): Label for each square if labels are given
    """

    split_features = torch.empty(64,3,50,50)
    split_labels = torch.empty(64,)
    for square in range(64):
        square_i = square // 8
        square_j = square % 8
        split_features[square,:,:,:] = features[:,50*square_i:50*(square_i+1),50*square_j:50*(square_j+1)]
        if type(labels) != type(None):
            split_labels[square] = labels[square_i,square_j]
    if type(labels) == type(None):
        return split_features
    return split_features, split_labels
        

if __name__ == '__main__':
    # Simple model prediction test on ood examples
    file_name = 'tester.png'
    features = get_example(file_name, has_labels=False)
    split_features = split_example(features)
    
    model = load_model('./models/resnet18.pth')
    logits = model(split_features)
    preds = logits.argmax(dim=1)
    fen_pred = labels_to_fen(preds.reshape(8,8))

    print(f'Predicted FEN: {fen_pred}')

    # Save model to comet workspace
    # my_key = os.environ.get("COMET_API_KEY")
    # experiment = Experiment(api_key=my_key,workspace='vincentlongpre',project_name='fen-generator')
    # experiment.log_model('ResNet18', './models/resnet18.pth')