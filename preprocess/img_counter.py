import os
import glob


# collections = glob.glob('/home/ec2-user/workspace/projects/facenet/data/train/*')
# for collection in collections:
#     print(f'running {collection}')
#     nfts = os.listdir(collection)
#     nfts.remove('image_data')
#     print(f'{len(nfts)} nfts')

def dec_zeros(s):
    for idx in range(len(s)):
        if s[idx] != '0':
            break
    return int(s[idx:])

lines = open('collection_num.txt').readlines()
ctr_num_kv = dict()
for line in lines:
    tokens = line.strip().split(' ')
    ctr_num_kv[tokens[0]] = int(tokens[1])

nfts = os.listdir('/home/ec2-user/workspace/projects/facenet/data/train-officail/alienfrensnft')
try:
    nfts.remove('image_data')
except Exception:
    pass
nfts = [dec_zeros(nft.split('.')[0]) for nft in nfts]
nfts = sorted(nfts)

print(len(nfts))
for idx in range(0, 10000):
    try:
        nfts.index(idx)
    except Exception:
        print(f'do not find {idx}')
