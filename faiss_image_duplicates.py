import copy

import numpy as np
import faiss
import requests
import torch
from PIL import Image
from tqdm import tqdm
from img2vec_pytorch import Img2Vec
import os
from clint.textui import progress
import zipfile



def getCocoImages():
    if not os.path.exists('cocoImages'):
        os.makedirs('cocoImages')
    r = requests.get('http://images.cocodataset.org/zips/train2017.zip', stream=True)
    path = os.path.join(os.getcwd(), 'cocoImages', 'train2017.zip')

    if not os.path.exists(path) or not os.path.isfile(path):
        with open(path, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
                if chunk:
                    f.write(chunk)
                    f.flush()

        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(os.getcwd(), 'cocoImages'))

    file_list = []

    for (root, dirs, files) in os.walk(os.path.join(os.getcwd(), 'cocoImages')):
        for file in files:
            file_name, file_ext = os.path.splitext(file)
            if file_ext == '.jpg':
                file_list.append(os.path.join(root, file))

    return file_list

if __name__ == "__main__":
    img2vec = Img2Vec(cuda=False)

    file_list = getCocoImages()
    #print(file_list)
    batch_size = 1000
    batch_images = [file_list[i:i + batch_size] for i in range(0, len(file_list), batch_size)]

    image_vectors = []
    for idx in tqdm(range(len(batch_images))):
        images = batch_images[idx]
        pil_images = [Image.open(i).convert('RGB') for i in images]

        vectors = img2vec.get_vec(pil_images, tensor=True)
        vectors = [t.reshape(512) for t in vectors]
        image_vectors.extend(vectors)

    v_list = []
    for idx in tqdm(range(len(image_vectors))):
        a = image_vectors[idx].detach().numpy()
        v_list.append(a)

    arr = np.array(v_list).reshape(len(v_list), 512)

    dim = 512
    nlist = 100
    quantizer = faiss.IndexFlatL2(dim)
    cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

    assert not cpu_index.is_trained
    cpu_index.train(arr)
    assert cpu_index.is_trained
    cpu_index.add(arr)

    D, I = cpu_index.search(arr, 5)

    print(D)

    print('Image duplicate check(Faiss)')
