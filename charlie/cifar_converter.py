import pickle
import numpy as np
import os

def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']  # shape (10000, 3072)
    labels = np.array(batch[b'labels'])
    return data, labels

def load_cifar10(root):
    xs, ys = [], []
    for i in range(1, 6):
        data, labels = load_cifar_batch(os.path.join(root, f"data_batch_{i}"))
        xs.append(data)
        ys.append(labels)
    x_train = np.concatenate(xs)   # shape (50000, 3072)
    y_train = np.concatenate(ys)   # shape (50000,)

    x_test, y_test = load_cifar_batch(os.path.join(root, "test_batch"))
    return (x_train, y_train), (x_test, y_test)

def one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels]

def save_cifar_as_npy(root, outdir):
    (x_train, y_train), (x_test, y_test) = load_cifar10(root)

    # flatten is already (N, 3072), so no reshape needed
    y_train = one_hot(y_train, 10)
    y_test = one_hot(y_test, 10)
   
    os.makedirs(outdir, exist_ok=True)

    np.save(os.path.join(outdir, "train_images.npy"), x_train.astype(np.float32) / 255.0)
    np.save(os.path.join(outdir, "train_labels.npy"), y_train.astype(np.float32) / 255.0)
    np.save(os.path.join(outdir, "test_images.npy"), x_test.astype(np.float32))
    np.save(os.path.join(outdir, "test_labels.npy"), y_test.astype(np.float32))


save_cifar_as_npy('../', 'completed/')

