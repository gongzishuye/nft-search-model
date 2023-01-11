import faiss                   # make faiss available
import numpy as np
import pickle
import torch

from eval import ModelInfer


d = 128

postfix = 'v1.1'
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
xb = np.load(f'log/database_collections-{postfix}.npy')
print(xb.shape)
index.add(xb)                  # add vectors to the index
print(index.ntotal)

with open(f'log/idx_path_kv-{postfix}.pickle', 'rb') as handle:
    idx_path_kv = pickle.load(handle)

model_path = '/home/ec2-user/workspace/facenet/log/useable-20220510/last_checkpoint.pth'
model_infer = ModelInfer(model_path)


def get_logits(path_img):
    with torch.no_grad():
        tensors = model_infer([path_img])
        tensor = tensors.cpu().numpy()
        return tensor


def search_nft(img_path):
    logits = get_logits(img_path)
    print(logits.shape)
    k = 10                          # we want to see 10 nearest neighbors
    D, I = index.search(logits, k)     # actual search
    print(I)
    print(D)
    score = D[0][0]
    print(idx_path_kv[I[0][0]])
    if score > 10.0:
        return None
    return idx_path_kv[I[0][0]]


if __name__ == '__main__':
    search_nft('meebits5.png')
