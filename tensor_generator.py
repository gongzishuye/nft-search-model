import torch
import glob
import pickle
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from eval import MiniModelInfer, resize_pad


model_path = '/home/ec2-user/workspace/facenet/log/useable-20220510/last_checkpoint.pth'
model_infer = MiniModelInfer(model_path)


class InferenceDataset(Dataset):

    def __init__(self, nfts, transform=None, noise=None):
        # notrain_base_dir = '/home/ec2-user/workspace/data/notrain/*/*'
        # train_base_dir = '/home/ec2-user/workspace/data/train/*/*'
        # notrain_nfts = glob.glob(notrain_base_dir)
        # train_nfts = glob.glob(train_base_dir)
        # nfts = notrain_nfts + train_nfts
        print(f'total nft amount {len(nfts)}')
        self.nfts = nfts
        self.transform = transform

    def __getitem__(self, idx):
        nft = self.nfts[idx]
        try:
            img = Image.open(nft).convert('RGB')
        except Exception:
            print(f'error {nft}')
            default_img = '/home/ec2-user/workspace/data/twitter_avt.jpeg'
            img = Image.open(default_img).convert('RGB')
        img = resize_pad(img)
        img_tensor = self.transform(img)
        return img_tensor, nft

    def __len__(self):
        return len(self.nfts)


def get_nfts():
    train_base_dir = '/home/ec2-user/workspace/data/train/azuki/*'
    train_nfts = glob.glob(train_base_dir)
    return train_nfts


trms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
nfts = get_nfts()
dataset = InferenceDataset(nfts, trms)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=16,
        shuffle=False, num_workers=8)


def get_logits(tensors):
    with torch.no_grad():
        tensors = model_infer(tensors)
        tensors = tensors.cpu().numpy()
        return tensors


def get_dataset_tensors():
    idx_path_kv = dict()
    logits_lst = []

    g_idx = 0
    for idx, (tensors, nfts) in enumerate(data_loader):
        if idx % 100 == 0:
            print(idx)
        
        for nft in nfts:
            idx_path_kv[g_idx] = nft
            g_idx += 1

        output_logits = get_logits(tensors)
        for logits in output_logits:
            logits_lst.append(logits.squeeze())

    logits_np = np.array(logits_lst)
    return logits_np, idx_path_kv


def load_tensor():
    logits_np = np.load('log/database_collections-v1.npy')
    with open('log/idx_path_kv-v1.pickle', 'rb') as handle:
        idx_path_kv = pickle.load(handle)
    return logits_np, idx_path_kv


def save_tensors(logits_np, idx_path_kv):
    np.save('log/database_collections-v1.1', logits_np)
    with open('log/idx_path_kv-v1.1.pickle', 'wb') as handle:
        pickle.dump(idx_path_kv, handle)


'''
with open('filename.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('filename.pickle', 'rb') as handle:
    b = pickle.load(handle)
'''


def main():
    old_logits_np, old_idx_path_kv = load_tensor()
    logits_np, idx_path_kv = get_dataset_tensors()
    new_logits_np = np.concatenate((old_logits_np, logits_np), axis=0)
    g_idx = len(old_logits_np)
    
    for idx in range(len(idx_path_kv)):
        nft = idx_path_kv[idx]
        old_idx_path_kv[g_idx] = nft
        g_idx += 1
    
    save_tensors(new_logits_np, old_idx_path_kv)


main()
# load_tensor()
