import pickle
from bitstring import Bits, BitArray
import numpy as np
from ldpc.decoder import bsc_llr, LogSpaDecoder
from ldpc.utils import AList
from decoders import ClassifyingEntropyDecoder
from utils.bit_operations import hamming_distance
from typing import Any
import matplotlib.pyplot as plt
import argparse
import datetime
import os
from multiprocessing import Pool
from scipy.io import savemat
import lzma
import pandas as pd
from utils import setup_logger
import shutil


parser = argparse.ArgumentParser(description='Run decoding on simulated data using multiprocessing.')
parser.add_argument("--N", default=0, help="max number of transmissions to consider", type=int)
parser.add_argument("--minflip", default=2*1e-2, help="minimal bit flip probability to consider", type=float)
parser.add_argument("--maxflip", default=40*1e-2, help="maximal bit flip probability to consider", type=float)
parser.add_argument("--nflips", default=20, help="number of bit flips to consider", type=int)
parser.add_argument("--ldpciterations", default=60, help="number of iterations of  LDPC decoder", type=int)
parser.add_argument("--ent_threshold", default=0.36, help="entropy threshold to be used in entropy decoder", type=float)
parser.add_argument("--window_len", default=50, help="number of previous samples to use, if 0 all are used", type=int)
parser.add_argument("--clipping_factor", default=2, help="dictates maximal and minimal llr", type=int)
parser.add_argument("--multiply_data", default=0, help="multiplies amount of buffers by 2 to power of arg", type=int)
parser.add_argument("--processes", default=0, help="number of processes to spawn", type=int)
parser.add_argument("--conf_center", default=40, help="center of model sigmoid", type=int)
parser.add_argument("--conf_slope", default=0.35, help="slope of model sigmoid", type=float)
parser.add_argument("--dec_type", default="BP", help="scheme for determining confidence", type=str)
parser.add_argument("--model_length", default="all", help="model length", type=str)

args = parser.parse_args()

ldpc_iterations = args.ldpciterations
thr = args.ent_threshold
clipping_factor = args.clipping_factor
processes = args.processes if args.processes > 0 else None

hc_tx = np.genfromtxt('data/non_repeated_tx.csv', dtype=np.uint8, delimiter=',')

encoded = [Bits(auto=tx) for tx in hc_tx]
n = args.N if args.N > 0 else len(encoded)
encoded = encoded[:n]
window_len = args.window_len if args.window_len > 0 else None

# corrupt data
rng = np.random.default_rng()
bit_flip_p = np.linspace(args.minflip, args.maxflip, num=args.nflips)

for _ in range(args.multiply_data):  # generate more buffers for statistical reproducibility
    encoded.extend(encoded)

k = 984  # info bits
word_len = 4098
model_length = k if args.model_length == 'info' else word_len
n = len(encoded)  # redfine n

logger = setup_logger(name=__file__, log_file=os.path.join("results/", 'log.log'))


cmd = f'python {__file__} --minflip {args.minflip} --maxflip {args.maxflip} --nflips {args.nflips} --ldpciterations ' \
      f'{ldpc_iterations} --ent_threshold {thr} --clipping_factor {clipping_factor} --conf_center {args.conf_center} ' \
      f'--conf_slope {args.conf_slope} --multiply_data {args.multiply_data} --dec_type {args.dec_type} ' \
      f'--model_length {args.model_length}'

if window_len is not None:
    cmd += f' --window_len {window_len}'
else:
    cmd += ' --window_len 0'
if args.N > 0:
    cmd += f' --N {n}'
if processes is not None:
    cmd += f' --processes {processes}'

h = AList.from_file("spec/4098_3095_non_sys_h.alist")

def simulation_step(p: float) -> dict[str, Any]:
    global ldpc_iterations
    global model_length
    global thr
    global clipping_factor
    global args
    global window_len
    global n
    global k
    global word_len
    global h
    global logger
    channel = bsc_llr(p=p)
    ldpc_decoder = LogSpaDecoder(h=h.to_array(), max_iter=ldpc_iterations, decoder_type=args.dec_type,
                             info_idx=np.array([True] * k + [False] * (word_len - k)))
    entropy_decoder = ClassifyingEntropyDecoder(
        LogSpaDecoder(h=h.to_array(), max_iter=ldpc_iterations, decoder_type=args.dec_type, info_idx=np.array(
            [True] * k + [False] * (word_len - k))),
        model_length=model_length, entropy_threshold=thr, clipping_factor=clipping_factor,
        classifier_training=0, n_clusters=1, window_length=window_len, conf_center=args.conf_center,
        conf_slope=args.conf_slope, bit_flip=p, cluster=False)
    no_errors = int(word_len * p)
    rx = []
    decoded_ldpc = []
    decoded_entropy = []
    errors = np.vstack(
        tuple(rng.choice(word_len, size=no_errors, replace=False)
              for _ in range(n))
    )
    step_results: dict[str, Any] = {}
    for tx_idx in range(n):
        corrupted = BitArray(encoded[tx_idx])
        for idx in errors[tx_idx]:
            corrupted[idx] = not corrupted[idx]
        rx.append(corrupted)
        channel_llr = channel(np.array(corrupted, dtype=np.int_))
        d = ldpc_decoder.decode(channel_llr)
        decoded_ldpc.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        d = entropy_decoder.decode_buffer(channel_llr)
        decoded_entropy.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        logger.info(f"p= {p}, tx id: {tx_idx}")
    logger.info(f"successful pure decoding for bit flip p= {p}, is: {sum(int(r[-1] == 0) for r in decoded_ldpc)}/{n}")
    logger.info(f"successful entropy decoding for bit flip p= {p}, is: {sum(int(r[-1] == 0) for r in decoded_entropy)}/{n}")
    # log data
    info_errors = np.sum(errors < k, axis=1)
    parity_errors = np.sum(errors >= k, axis=1)
    zipped = [[np.array(en, dtype=np.int_), np.array(r, dtype=np.int_), er, inf, pa] for en, r, er, inf, pa in
              zip(encoded, rx, errors, info_errors, parity_errors)]
    step_results["data"] = pd.DataFrame(zipped, columns=["encoded", "corrupted", "error_idx", "info_errors",
                                                         "parity_errors"])

    # params
    step_results["raw_ber"] = no_errors / word_len
    step_results["buffer_len"] = word_len
    step_results["number_of_buffers"] = n
    step_results["max_ldpc_iterations"] = ldpc_iterations

    # decoding
    decoded_entropy_df = pd.DataFrame(decoded_entropy,
                                      columns=["estimate", "llr", "decode_success", "iterations", "syndrome",
                                               "vnode_validity", "dist", "structural_idx", "cluster_label", "hamming"])
    step_results["decoded_entropy"] = decoded_entropy_df
    decoded_ldpc_df = pd.DataFrame(decoded_ldpc,
                                   columns=["estimate", "llr", "decode_success", "iterations", "syndrome",
                                            "vnode_validity", "hamming"])
    step_results['decoded_ldpc'] = decoded_ldpc_df
    # performance
    step_results["ldpc_buffer_success_rate"] = sum(int(res[-1] == 0) for res in decoded_ldpc) / float(n)
    step_results["ldpc_decoder_ber"] = sum(res[-1] for res in decoded_ldpc) / float(n * len(encoded[0]))
    step_results["entropy_buffer_success_rate"] = sum(int(res[-1] == 0) for res in decoded_entropy) / float(n)
    step_results["entropy_decoder_ber"] = sum(res[-1] for res in decoded_entropy) / float(n * len(encoded[0]))

    timestamp = f'{str(datetime.date.today())}_{str(datetime.datetime.now().hour)}_{str(datetime.datetime.now().minute)}_' \
                f'{str(datetime.datetime.now().second)}'
    with open(f'{timestamp}_{p}_simulation_classifying_entropy.pickle', 'wb') as f:
        pickle.dump(step_results, f)
    return step_results


if __name__ == '__main__':

    timestamp = f'{str(datetime.date.today())}_{str(datetime.datetime.now().hour)}_{str(datetime.datetime.now().minute)}_' \
                f'{str(datetime.datetime.now().second)}'

    path = os.path.join("results/", timestamp)
    os.mkdir(path)

    logger.info(__file__)
    logger.info(f"number of buffers to process: {n}")
    logger.info(f"smallest bit flip probability: {args.minflip}")
    logger.info(f"largest bit flip probability: {args.maxflip}")
    logger.info(f"number of bit flips: {args.nflips}")
    logger.info(f"number of ldpc decoder iterations: {ldpc_iterations}")
    logger.info(f"entropy threshold used in entropy decoder: {thr}")
    logger.info(f"entropy decoder window length: {window_len}")
    logger.info(f"clipping factor: {clipping_factor}")
    logger.info(f"model center: {args.conf_center}")
    logger.info(f"model slope: {args.conf_slope}")
    logger.info(f"processes: {args.processes}")
    logger.info(f"multiply data: {args.multiply_data}")
    logger.info(f"decoder type: {args.dec_type}")
    logger.info(f"model_length: {args.model_length}")

    with open(os.path.join(path, "cmd.txt"), 'w') as f:
        f.write(cmd)
    logger.info(cmd)

    with Pool(processes=processes) as pool:
        results: list[dict[str, Any]] = pool.map(simulation_step, bit_flip_p)
    # results: list[dict[str, Any]] = list(map(simulation_step, bit_flip_p))

    try:
        os.mkdir(os.path.join(path, f'{timestamp}_simulation_entropy_vs_pure_LDPC_weighted_model_{args.dec_type}_decoder'))
        with lzma.open(
                os.path.join(path, f'{timestamp}_simulation_rafael_code.xz'),
                "wb") as f:
            pickle.dump(results, f)
        logger.info("saved compressed results file")

        raw_ber = np.array([p['raw_ber'] for p in results])
        ldpc_ber = np.array([p['ldpc_decoder_ber'] for p in results])
        entropy_ber = np.array([p['entropy_decoder_ber'] for p in results])
        fig = plt.figure()
        plt.plot(raw_ber, ldpc_ber, 'bo', raw_ber, raw_ber, 'g^', raw_ber, entropy_ber, 'r*')
        plt.xlabel("BSC bit flip probability p")
        plt.ylabel("post decoding BER")
        fig.savefig(os.path.join(path, "ber_vs_error_p.eps"), dpi=150)

        figure = plt.figure()
        ldpc_buffer_success_rate = np.array([p['ldpc_buffer_success_rate'] for p in results])
        entropy_buffer_success_rate = np.array([p['entropy_buffer_success_rate'] for p in results])
        plt.plot(raw_ber, ldpc_buffer_success_rate, 'bo', raw_ber, entropy_buffer_success_rate, 'r*')
        plt.xlabel("BSC bit flip probability p")
        plt.ylabel("Decode success rate")
        figure.savefig(os.path.join(path, "buffer_success_rate_vs_error_p.eps"), dpi=150)

        logger.info("saved figures")
        summary = {"args": args, "raw_ber": raw_ber, "ldpc_ber": ldpc_ber, "entropy_ber": entropy_ber,
                   "ldpc_buffer_success_rate": ldpc_buffer_success_rate,
                   "entropy_buffer_success_rate": entropy_buffer_success_rate}
        with open(os.path.join(path, f'{timestamp}_summary_simulation_rafael_code.pickle'), 'wb') as f:
            pickle.dump(summary, f)

        savemat(os.path.join(path, f'{timestamp}_summary_simulation_rafael_code.mat'),
                summary)
        logger.info("saved summary")
        for step in results:
            step['data'] = step['data'].to_dict("list")
            step['decoded_entropy'] = step['decoded_entropy'].to_dict("list")
            step['decoded_ldpc'] = step['decoded_ldpc'].to_dict("list")

        summary["results"] = results
        savemat(os.path.join(path, f'{timestamp}_simulation_rafael_code.mat'),
                summary, do_compression=True)
        logger.info("saved results to mat file")
        shutil.move("results/log.log",os.path.join(path, "log.log"))
    except Exception as e:
        logger.exception(e)
        shutil.move("results/log.log", os.path.join(path, "log.log"))
        raise e
