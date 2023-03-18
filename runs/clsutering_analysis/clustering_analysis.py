import argparse
import datetime
import os
from typing import Any
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from inference import BufferClassifier, BMM
from scipy.io import savemat
from functions import relabel, stats
from multiprocessing import Pool
from utils import setup_logger


parser = argparse.ArgumentParser()
parser.add_argument("--n_classes", default=3, type=int)
parser.add_argument("--p", default=0.07, type=float)
parser.add_argument("--n_training", default=150, type=int)
parser.add_argument("--train_with_errors", default=1, type=int)
parser.add_argument("--shuffle", default=1, type=int)
parser.add_argument("--n_runs", default=10, type=int)
parser.add_argument("--n_jobs", default=0, type=int)
args = parser.parse_args()

n_classes = args.n_classes
p = args.p
n_training = args.n_training
train_with_errors = bool(args.train_with_errors)
shuffle = bool(args.shuffle)
number_of_runs = args.n_runs
n_jobs = args.n_jobs
logger = setup_logger(name=__file__, log_file=os.path.join("results/", 'log.log'))


def analyze_mine(buffers, n_buffers):
    global n_classes, n_training
    classifier = BufferClassifier(n_training, n_classes, classify_dist="LL", merge_dist="Hellinger",
                                  weight_scheme="linear")
    labels = np.empty(n_buffers * n_classes, dtype=np.uint8)
    for idx, buffer in enumerate(buffers):
        labels[idx] = classifier.classify(buffer)
    return labels[n_training:]


def analyze(run_idx: int):
    global n_classes, p, n_training, train_with_errors, shuffle, logger
    rng = np.random.default_rng()
    if n_classes == 2:
        n = 1296
        r = 1/2
    elif n_classes == 3:
        n = 648
        r = 3 / 4
    else:
        raise ValueError("n_classes must be 2 or 3")

    k = int(n*r)
    tx = np.genfromtxt("telemetry_hc_tx_info_bits_2023.csv", delimiter=",", dtype=np.uint8)
    n_buffers, real_info_bits_per_buffer = tx.shape
    no_errors = int(k*p)

    # break down to buffers
    buffers = np.empty((n_buffers*n_classes, k), dtype=np.bool_)
    actual_classes = np.empty(n_buffers*n_classes, dtype=np.uint8)
    pad_len = k - real_info_bits_per_buffer//n_classes
    for tx_idx, tx in enumerate(tx):
        for class_idx in range(n_classes):
            buffers[tx_idx*n_classes + class_idx] = np.hstack(
                    (tx[class_idx*real_info_bits_per_buffer//n_classes:(class_idx+1)*real_info_bits_per_buffer//n_classes],
                        rng.integers(low=0, high=2, size=pad_len))
                )  # pad with random bits
            if no_errors > 0 and (train_with_errors or tx_idx >= n_training):
                errors = rng.choice(k//n_classes, size=no_errors//n_classes, replace=False)
                errors_bool = np.zeros(k, dtype=np.bool_)
                errors_bool[errors] = True
                buffers[tx_idx * n_classes + class_idx] = np.bitwise_xor(buffers[tx_idx * n_classes + class_idx], errors_bool)
            actual_classes[tx_idx*n_classes + class_idx] = class_idx

    if shuffle:
        X_train, X_test, y_train, y_test = train_test_split(
            buffers, actual_classes, train_size=n_training, shuffle=True, stratify=actual_classes
        )
        buffers = np.vstack((X_train, X_test))
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            buffers, actual_classes, train_size=n_training, shuffle=False
        )
        buffers = np.vstack((X_train, X_test))
    clustering_labels = analyze_mine(buffers, n_buffers)
    clustering_labels, l_map = relabel(clustering_labels, n_classes, y_test)
    clustering_stats = stats(y_test, clustering_labels)
    logger.info(f"Run {run_idx}, clustering done, accuracy {clustering_stats[-1]:.3f}")
    # compare with BMM
    bmm = BMM(n_classes, 1000)
    bmm.fit(X_train)
    logger.info(f"Run {run_idx}, BMM training done")
    bmm_predictions = bmm.predict(X_test)
    relabeled_bmm_predictions, l_map = relabel(bmm_predictions, n_classes,y_test)
    bmm_stats = stats(y_test, relabeled_bmm_predictions)
    logger.info(f"Run {run_idx}, BMM done, accuracy {bmm_stats[-1]:.3f}")
    return clustering_stats, bmm_stats

if __name__ == "__main__":
    try:
        timestamp = f'{str(datetime.date.today())}_{str(datetime.datetime.now().hour)}_{str(datetime.datetime.now().minute)}_' \
                    f'{str(datetime.datetime.now().second)}'

        path = os.path.join("results/", timestamp)
        os.mkdir(path)
        logger.info(f"Starting {n_classes} classes, {p} p, {n_training} training, train with errors {train_with_errors}, "
                    f"shuffle {shuffle}, {number_of_runs} runs")
        cmd = f"python {__file__} --n_classes {n_classes} --p {p} --n_training {n_training} --train_with_errors " \
              f"{train_with_errors} --shuffle {shuffle} --n_runs {number_of_runs}"
        with open(os.path.join(path, "cmd.txt"), "w") as f:
            f.write(cmd)
        logger.info(cmd)
        if n_jobs == 1:
            results: list[Any] = [analyze(idx) for idx in range(number_of_runs)]
        else:
            with Pool(processes=n_jobs if n_jobs>0 else None) as pool:
                results = pool.map(analyze, range(number_of_runs))
        clustering_results = [t[0] for t in results]
        clustering_cm = np.round(np.array([t[0] for t in clustering_results]).mean(axis=0)).astype(np.int_)
        clustering_cm, clustering_recall, clustering_precision, clustering_f1, clustering_accuracy = stats(cm=clustering_cm)

        bmm_results = [t[1] for t in results]
        bmm_cm = np.round(np.array([t[0] for t in bmm_results]).mean(axis=0)).astype(np.int_)
        bmm_cm, bmm_recall, bmm_precision, bmm_f1, bmm_accuracy = stats(cm=bmm_cm)

        savemat(os.path.join(path, f'{timestamp}_{n_classes}classes_{p}p_{n_training}training_train_with_'
                                   f'errors{int(train_with_errors)}_shuffle{shuffle}_clustering_results.mat'), {
            "classified_buffers": np.sum(clustering_cm),
            "p": p,
            "n_training": n_training,
            "n_classes": n_classes,
            "train_with_errors": train_with_errors,
            "clustering_cm": clustering_cm,
            "clustering_recall": clustering_recall,
            "clustering_precision": clustering_precision,
            "clustering_f1": clustering_f1,
            "clustering_accuracy": clustering_accuracy,
            "bmm_cm": bmm_cm,
            "bmm_recall": bmm_recall,
            "bmm_precision": bmm_precision,
            "bmm_f1": bmm_f1,
            "bmm_accuracy": bmm_accuracy
        })
        logger.info("saved results to mat file")
        shutil.move("results/log.log", os.path.join(path, "log.log"))
    except Exception as e:
        logger.exception(e)
        shutil.move("results/log.log", os.path.join(path, "log.log"))
        raise e
