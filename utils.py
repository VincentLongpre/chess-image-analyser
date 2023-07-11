import numpy as np
import torch

pieces_conv_list = ['p','n','b','r','q','k','P','N','B','R','Q','K']

def fen_to_labels(fen: str):
    """ 
    Convert FEN notation to a tensor containing an integer label representing
    the piece (or absence of piece) on each square of the board.

    Args:
        fen (str): FEN notation of the position
    """

    labels = torch.empty(8,8)
    rows = fen.split('-')
    i, j = 0, 0
    for row in rows:
        while row:
            if row[0] in pieces_conv_list:
                labels[i,j] = 1 + pieces_conv_list.index(row[0])
                j += 1
            elif row[0].isdigit():
                gap = int(row[0])
                labels[i,j:j+gap] = 0
                j += gap
            row = row[1:]
        j = 0
        i += 1
    return labels

def labels_to_fen(labels: torch.Tensor):
    """ 
    Convert a tensor of labels to the FEN notation of the position.

    Args:
        labels (tensor): Tensor containing an integer representing the piece on each square
    """

    fen = ''
    for i in range(8):
        row = ''
        j = 0
        gap = 0
        while j < 8:
            label = int(labels[i,j].item())
            if label in range(1,13):
                if gap:
                    row += str(gap)
                    gap = 0
                row = row + pieces_conv_list[label-1]
            elif label == 0:
                gap += 1
            j += 1
        if gap: row += str(gap)
        fen += row + '-'
    return fen[:-1]

def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor):
    """ 
    Return the mean loss for this batch

    Args:
        logits (tensor): [batch_size, num_class]
        labels (tensor): [batch_size] 
    """

    labels = torch.nn.functional.one_hot(labels,num_classes=logits.shape[1])
    preds = torch.nn.functional.log_softmax(logits,dim=1)
    return -(preds * labels).sum(dim=1).mean()

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    """ 
    Compute the accuracy of the batch 

    Args:
        logits (tensor): [batch_size, num_class]
        labels (tensor): [batch_size] 
    """

    acc = (logits.argmax(dim=1) == labels).float().mean()
    return acc

def to_device(tensors, device: str):
    """
    Transfer a tensor, list of tensor or tensor dictionary to the selected device.

    Args:
        tensors (tensor, tensor list or tensor dict): Tensor object to send to device
        device (str): Device name
    """
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, dict):
        return dict(
            (key, to_device(tensor, device)) for (key, tensor) in tensors.items()
        )
    elif isinstance(tensors, list):
        return list(
            (to_device(tensors[0], device), to_device(tensors[1], device)))
    else:
        raise NotImplementedError("Unknown type {0}".format(type(tensors)))
        
def seed_experiment(seed: int):
    """Seed the pseudorandom number generator, for repeatability.

    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True