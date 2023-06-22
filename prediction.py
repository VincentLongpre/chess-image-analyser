from resnet18 import ResNet18
from PIL import Image
from torchvision import transforms
from utils import labels_to_fen, fen_to_labels
from comet_ml import Experiment
import numpy as np
import torch
import os

test_data_path = 'dataset/ood'

def load_model(save_path):
    model = ResNet18(num_classes=13)
    model.load_state_dict(torch.load(save_path,map_location='cpu'))
    return model

def get_example(file_name, has_labels=True):
    image = Image.open(os.path.join(test_data_path,file_name))
    image = image.resize((400,400), Image.Resampling.LANCZOS)
    if has_labels:
        labels = fen_to_labels(file_name.split('.')[0])
    convert_tensor = transforms.ToTensor()
    features = convert_tensor(image)[:3,:,:]
    if not has_labels:
        return features
    return features, labels

def split_example(features, labels=None):
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
    features, labels = get_example('2b5-p2NBp1p-1bp1nPPr-3P4-2pRnr1P-1k1B1Ppp-1P1P1pQP-Rq1N3K.jpeg')
    split_features, split_labels = split_example(features, labels)
    convert_tensor = transforms.ToPILImage()
    image = convert_tensor(split_features[57,:,:,:])
    
    model = load_model('./models/resnet18.pth')
    logits = model(split_features)
    preds = logits.argmax(dim=1)
    fen_pred = labels_to_fen(preds.reshape(8,8))

    print(f'Predicted FEN: {fen_pred}')

    # Save model to comet workspace
    # my_key = os.environ.get("COMET_API_KEY")
    # experiment = Experiment(api_key=my_key,workspace='vincentlongpre',project_name='fen-generator')
    # experiment.log_model('Augmented ResNet18', './models/resnet18_v2.1.pth')