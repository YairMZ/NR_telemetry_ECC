import pickle
from bitstring import Bits, BitArray
import numpy as np
from ldpc.encoder import EncoderWiFi
from ldpc.wifi_spec_codes import WiFiSpecCode
from ldpc.decoder import bsc_llr, DecoderWiFi
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
import json


parser = argparse.ArgumentParser(description='Run decoding on simulated data using multiprocessing.')
parser.add_argument("--N", default=0, help="max number of transmissions to consider", type=int)
parser.add_argument("--minflip", default=33*1e-3, help="minimal bit flip probability to consider", type=float)
parser.add_argument("--maxflip", default=70*1e-3, help="maximal bit flip probability to consider", type=float)
parser.add_argument("--nflips", default=20, help="number of bit flips to consider", type=int)
parser.add_argument("--ldpciterations", default=20, help="number of iterations of  LDPC decoder", type=int)
parser.add_argument("--ent_threshold", default=0.36, help="entropy threshold to be used in entropy decoder", type=float)
parser.add_argument("--window_len", default=50, help="number of previous samples to use, if 0 all are used", type=int)
parser.add_argument("--clipping_factor", default=2, help="dictates maximal and minimal llr", type=int)
parser.add_argument("--multiply_data", default=0, help="multiplies amount of buffers by 2 to power of arg", type=int)
parser.add_argument("--processes", default=0, help="number of processes to spawn", type=int)
parser.add_argument("--conf_center", default=40, help="center of model sigmoid", type=int)
parser.add_argument("--conf_slope", default=0.35, help="slope of model sigmoid", type=float)
parser.add_argument("--dec_type", default="BP", help="scheme for determining confidence", type=str)
parser.add_argument("--msg_delay", default="50000", help="sampling delay", type=str)
parser.add_argument("--model_length", default="info", help="model length", type=str)

args = parser.parse_args()

ldpc_iterations = args.ldpciterations
thr = args.ent_threshold
clipping_factor = args.clipping_factor
processes = args.processes if args.processes > 0 else None

with open('data/hc_to_ship.pickle', 'rb') as f:
    hc_tx = pickle.load(f)

hc_bin_data = [Bits(auto=tx.get("bin")) for tx in hc_tx.get(args.msg_delay)]
n = args.N if args.N > 0 else len(hc_bin_data)
window_len = args.window_len if args.window_len > 0 else None

# corrupt data
rng = np.random.default_rng()
bit_flip_p = np.linspace(args.minflip, args.maxflip, num=args.nflips)

spec = WiFiSpecCode.N1944_R23
encoder = EncoderWiFi(spec=spec)
encoded = []
for binary_data in hc_bin_data[:n]:
    pad_len = encoder.k - len(binary_data)
    padded = binary_data + Bits(auto=rng.integers(low=0, high=2, size=pad_len))
    encoded.append(encoder.encode(padded))

for _ in range(args.multiply_data):  # generate more buffers for statistical reproducibility
    encoded.extend(encoded)

model_length = encoder.k if args.model_length == 'info' else encoder.n
n = len(encoded)  # redfine n
with open('data/encoded_data_data_model_1_clusters.json', 'r') as f:
    data_model = np.array(json.load(f))

# encoded structure: {starting byte index in buffer: msg_id}
# {0: 33, 52: 234, 72: 30, 108: 212, 135: 218}
# last 9 bytes (72 bits) are padding and not messages. Thus last message ends at byte index 152
# bit indices:
# {0: 33, 416: 234, 576: 30, 864: 212, 1080: 218}

logger = setup_logger(name=__file__, log_file=os.path.join("results/", 'log.log'))


def simulation_step(p: float) -> dict[str, Any]:
    global ldpc_iterations
    global model_length
    global args
    global window_len
    global n
    global spec
    global encoder
    global logger
    global rng
    global thr
    global clipping_factor
    global data_model

    channel = bsc_llr(p=p)
    ldpc_decoder = DecoderWiFi(spec=spec, max_iter=ldpc_iterations, decoder_type=args.dec_type)
    entropy_decoder = ClassifyingEntropyDecoder(DecoderWiFi(spec=spec, max_iter=ldpc_iterations,
                                                            decoder_type=args.dec_type),
                                                model_length=model_length, entropy_threshold=thr,
                                                clipping_factor=clipping_factor, classifier_training=0,
                                                n_clusters=1, window_length=window_len,
                                                conf_center=args.conf_center,
                                                conf_slope=args.conf_slope, bit_flip=p, cluster=0)
    pre_trained_decoder = ClassifyingEntropyDecoder(DecoderWiFi(spec=spec, max_iter=ldpc_iterations,
                                                                decoder_type=args.dec_type),
                                                    model_length=model_length, entropy_threshold=thr,
                                                    clipping_factor=clipping_factor, classifier_training=0,
                                                    n_clusters=1, window_length=window_len,
                                                    bit_flip=p, cluster=0,
                                                    data_model=data_model)
    no_errors = int(encoder.n * p)
    rx = np.zeros((n, encoder.n), dtype=np.bool_)
    decoded_ldpc = []
    decoded_entropy = []
    decoded_pre_trained = []
    errors = np.vstack(
        tuple(rng.choice(encoder.n, size=no_errors, replace=False)
              for _ in range(n))
    )
    step_results: dict[str, Any] = {'data': hc_bin_data[:n]}
    for tx_idx in range(n):
        corrupted = BitArray(encoded[tx_idx])
        corrupted.invert(errors[tx_idx])
        rx[tx_idx] = np.array(corrupted, dtype=np.bool_)
        channel_llr = channel(np.array(corrupted, dtype=np.int_))
        d = ldpc_decoder.decode(channel_llr)
        decoded_ldpc.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        d = entropy_decoder.decode_buffer(channel_llr)
        decoded_entropy.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        d = pre_trained_decoder.decode_buffer(channel_llr)
        decoded_pre_trained.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        logger.info(f"p= {p}, tx id: {tx_idx}")
    pure_success = sum(int(r[-1] == 0) for r in decoded_ldpc)
    entropy_success = sum(int(r[-1] == 0) for r in decoded_entropy)
    pre_trained_success = sum(int(r[-1] == 0) for r in decoded_pre_trained)
    logger.info(f"successful pure decoding for bit flip p= {p}, is: {pure_success}/{n}")
    logger.info(f"successful entropy decoding for bit flip p= {p}, is: {entropy_success}/{n}")
    logger.info(f"successful pre-trained decoding for bit flip p= {p}, is: {pre_trained_success}/{n}")
    if n-pure_success > 0:
        logger.info(f"NR recovery rate for bit flip p= {p}, is: {(entropy_success-pure_success)/(n-pure_success)}")
    if n-pre_trained_success > 0:
        logger.info(f"pre-trained recovery rate for bit flip p= {p}, is: "
                    f"{(pre_trained_success-pure_success)/(n-pure_success)}")
    # log data
    info_errors = np.sum(errors < encoder.k, axis=1)
    parity_errors = np.sum(errors >= encoder.k, axis=1)
    zipped = [[np.array(en, dtype=np.int_), r, er, inf, pa] for en, r, er, inf, pa in
              zip(encoded, rx, errors, info_errors, parity_errors)]
    step_results["data"] = pd.DataFrame(zipped, columns=["encoded", "corrupted", "error_idx", "info_errors",
                                                         "parity_errors"])

    # params
    step_results["raw_ber"] = no_errors / encoder.n
    step_results["buffer_len"] = len(encoded[0])
    step_results["number_of_buffers"] = n
    step_results["max_ldpc_iterations"] = ldpc_iterations

    # decoding
    decoded_entropy_df = pd.DataFrame(decoded_entropy,
                                      columns=["estimate", "llr", "decode_success", "iterations", "syndrome",
                                               "vnode_validity", "dist", "structural_idx", "cluster_label", "hamming"])
    decoded_entropy_df.drop('structural_idx', axis=1, inplace=True)
    step_results["decoded_entropy"] = decoded_entropy_df
    decoded_ldpc_df = pd.DataFrame(decoded_ldpc,
                                   columns=["estimate", "llr", "decode_success", "iterations", "syndrome",
                                            "vnode_validity", "hamming"])
    step_results['decoded_ldpc'] = decoded_ldpc_df
    decoded_pre_trained_df = pd.DataFrame(decoded_pre_trained,
                                          columns=["estimate", "llr", "decode_success", "iterations", "syndrome",
                                                   "vnode_validity", "dist", "structural_idx", "cluster_label", "hamming"])
    decoded_pre_trained_df.drop('structural_idx', axis=1, inplace=True)
    step_results["decoded_pre_trained"] = decoded_pre_trained_df
    # performance
    step_results["ldpc_buffer_success_rate"] = sum(int(res[-1] == 0) for res in decoded_ldpc) / float(n)
    step_results["ldpc_decoder_ber"] = sum(res[-1] for res in decoded_ldpc) / float(n * len(encoded[0]))
    step_results["entropy_buffer_success_rate"] = sum(int(res[-1] == 0) for res in decoded_entropy) / float(n)
    step_results["entropy_decoder_ber"] = sum(res[-1] for res in decoded_entropy) / float(n * len(encoded[0]))
    step_results["pre_trained_buffer_success_rate"] = sum(int(res[-1] == 0) for res in decoded_pre_trained) / float(n)
    step_results["pre_trained_decoder_ber"] = sum(res[-1] for res in decoded_pre_trained) / float(n * len(encoded[0]))

    timestamp = f'{str(datetime.date.today())}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}_' \
                f'{datetime.datetime.now().second}'
    with open(f'{timestamp}_{p}_simulation_pretrained_entropy_model.pickle', 'wb') as f:
        pickle.dump(step_results, f)
    return step_results


if __name__ == '__main__':
    timestamp = f'{str(datetime.date.today())}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}_' \
                f'{datetime.datetime.now().second}'

    path = os.path.join("results/", timestamp)

    logger.info(__file__)
    logger.info(f"number of buffers to process: {n}")
    logger.info(f"smallest bit flip probability: {args.minflip}")
    logger.info(f"largest bit flip probability: {args.maxflip}")
    logger.info(f"number of bit flips: {args.nflips}")
    logger.info(f"number of ldpc decoder iterations: {ldpc_iterations}")
    logger.info(f"processes: {args.processes}")
    logger.info(f"multiply data: {args.multiply_data}")
    logger.info(f"msg_delay: {args.msg_delay} ")
    logger.info(f"decoder type: {args.dec_type}")
    logger.info(f"decoder window length: {window_len}")
    logger.info(f"confidence center: {args.conf_center}")
    logger.info(f"confidence slope: {args.conf_slope}")
    logger.info(f"entropy threshold: {thr}")
    logger.info(f"clipping factor: {clipping_factor}")
    logger.info(f"model length: {args.model_length}")

    cmd = f'python {__file__} --minflip {args.minflip} --maxflip {args.maxflip} --nflips {args.nflips}  ' \
          f'--ldpciterations {ldpc_iterations} --multiply_data {args.multiply_data} --msg_delay {args.msg_delay} ' \
          f'--dec_type {args.dec_type} --conf_center {args.conf_center} --model_length {args.model_length} ' \
          f'--conf_slope {args.conf_slope} --entropy_threshold {thr} --clipping_factor {clipping_factor}'
    if args.N > 0:
        cmd += f' --N {n}'
    if window_len is not None:
        cmd += f' --window_len {window_len}'
    else:
        cmd += ' --window_len 0'
    if processes is not None:
        cmd += f' --processes {processes}'
    os.mkdir(path)
    with open(os.path.join(path, "cmd.txt"), 'w') as f:
        f.write(cmd)
    logger.info(cmd)

    try:
        with Pool(processes=processes) as pool:
            results: list[dict[str, Any]] = pool.map(simulation_step, bit_flip_p)
        # results: list[dict[str, Any]] = list(map(simulation_step, bit_flip_p))
        with lzma.open(
                os.path.join(path, f'{timestamp}_simulation_pretrained_entropy_model_{args.dec_type}_decoder.xz'),
                "wb") as f:
            pickle.dump(results, f)

        raw_ber = np.array([p['raw_ber'] for p in results])
        ldpc_ber = np.array([p['ldpc_decoder_ber'] for p in results])
        entropy_ber = np.array([p['entropy_decoder_ber'] for p in results])
        pre_trained_ber = np.array([p['pre_trained_decoder_ber'] for p in results])
        fig = plt.figure()
        plt.plot(raw_ber, ldpc_ber, 'bo', raw_ber, raw_ber, 'g^', raw_ber, entropy_ber, 'r*', raw_ber, pre_trained_ber, 'k+')
        plt.xlabel("BSC bit flip probability p")
        plt.ylabel("post decoding BER")
        fig.savefig(os.path.join(path, "ber_vs_error_p.eps"), dpi=150)

        figure = plt.figure()
        ldpc_buffer_success_rate = np.array([p['ldpc_buffer_success_rate'] for p in results])
        entropy_buffer_success_rate = np.array([p['entropy_buffer_success_rate'] for p in results])
        pre_trained_buffer_success_rate = np.array([p['pre_trained_buffer_success_rate'] for p in results])
        plt.plot(raw_ber, ldpc_buffer_success_rate, 'bo', raw_ber, entropy_buffer_success_rate, 'r*', raw_ber,
                 pre_trained_buffer_success_rate, 'k+')
        plt.xlabel("BSC bit flip probability p")
        plt.ylabel("Decode success rate")
        figure.savefig(os.path.join(path, "buffer_success_rate_vs_error_p.eps"), dpi=150)

        summary = {"args": args, "raw_ber": raw_ber, "ldpc_ber": ldpc_ber, "entropy_ber": entropy_ber,
                   "pre_trained_ber": pre_trained_ber, "pre_trained_buffer_success_rate": pre_trained_buffer_success_rate,
                   "ldpc_buffer_success_rate": ldpc_buffer_success_rate,
                   "entropy_buffer_success_rate": entropy_buffer_success_rate}
        with open(os.path.join(path, f'{timestamp}_summary_pretrained_entropy_model_{args.dec_type}_decoder.pickle'),
                  'wb') as f:
            pickle.dump(summary, f)

        savemat(os.path.join(path, f'{timestamp}_summary_pretrained_entropy_model_{args.dec_type}_decoder.mat'), summary)

        for step in results:
            step['data'] = step['data'].to_dict("list")
            step['decoded_entropy'] = step['decoded_entropy'].to_dict("list")
            step['decoded_ldpc'] = step['decoded_ldpc'].to_dict("list")
            step['decoded_pre_trained'] = step['decoded_pre_trained'].to_dict("list")

        summary["results"] = results
        savemat(os.path.join(path, f'{timestamp}_simulation_pretrained_entropy_model_{args.dec_type}_decoder.mat'),
                summary, do_compression=True)
        logger.info("saved results to mat file")
        shutil.move("results/log.log", os.path.join(path, "log.log"))
    except Exception as e:
        logger.exception(e)
        shutil.move("results/log.log", os.path.join(path, "log.log"))
        raise e
