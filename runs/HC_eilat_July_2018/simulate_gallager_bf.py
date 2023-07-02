import pickle
from bitstring import Bits
import numpy as np
from ldpc.encoder import EncoderWiFi
from ldpc.wifi_spec_codes import WiFiSpecCode
from ldpc.decoder import GalBfDecoder, bsc_llr
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
parser.add_argument("--minflip", default=0.001, help="minimal bit flip probability to consider", type=float)
parser.add_argument("--maxflip", default=0.02, help="maximal bit flip probability to consider", type=float)
parser.add_argument("--nflips", default=15, help="number of bit flips to consider", type=int)
parser.add_argument("--ldpciterations", default=300, help="number of iterations of BF LDPC decoder", type=int)
parser.add_argument("--ent_threshold", default=0.36, help="entropy threshold to be used in entropy decoder", type=float)
parser.add_argument("--window_len", default=100, help="number of previous samples to use, if 0 all are used", type=int)
parser.add_argument("--multiply_data", default=2, help="multiplies amount of buffers by 2 to power of arg", type=int)
parser.add_argument("--processes", default=0, help="number of processes to spawn", type=int)
parser.add_argument("--conf_center", default=40, help="center of model sigmoid", type=int)
parser.add_argument("--conf_slope", default=0.35, help="slope of model sigmoid", type=float)
parser.add_argument("--model_length", default="info", help="model length", type=str)

args = parser.parse_args()

ldpc_iterations = args.ldpciterations
thr = args.ent_threshold
processes = args.processes if args.processes > 0 else None

with open('data/hc_to_ship.pickle', 'rb') as f:
    hc_tx = pickle.load(f)

hc_bin_data = [Bits(auto=tx.get("bin")) for tx in hc_tx.get("50000")]
n = args.N if args.N > 0 else len(hc_bin_data)
window_len = args.window_len if args.window_len > 0 else None

# corrupt data
rng = np.random.default_rng()

spec = WiFiSpecCode.N1944_R23
encoder = EncoderWiFi(spec=spec)
h = encoder.h
encoded = []
for binary_data in hc_bin_data[:n]:
    pad_len = encoder.k - len(binary_data)
    padded = np.array(binary_data + Bits(auto=rng.integers(low=0, high=2, size=pad_len)), dtype=np.int_)
    encoded.append(encoder.encode(padded))

for _ in range(args.multiply_data):  # generate more buffers for statistical reproducibility
    encoded.extend(encoded)
model_length = encoder.k if args.model_length == 'info' else encoder.n
n = len(encoded)  # redfine n

# encoded structure: {starting byte index in buffer: msg_id}
# {0: 33, 52: 234, 72: 30, 108: 212, 135: 218}
# last 9 bytes (72 bits) are padding and not messages. Thus last message ends at byte index 152
# bit indices:
# {0: 33, 416: 234, 576: 30, 864: 212, 1080: 218}

logger = setup_logger(name=__file__, log_file=os.path.join("results/", 'log.log'))


def simulation_step(p: float) -> dict[str, Any]:
    global ldpc_iterations
    global model_length
    global thr
    global args
    global window_len
    global n
    global spec
    global encoded
    global h

    channel = bsc_llr(p=p)
    no_errors = int(encoder.n * p)
    errors = np.vstack(
        tuple(rng.choice(encoder.n, size=no_errors, replace=False)
              for _ in range(n))
    )
    m, nn = h.shape
    k = nn - m
    galbf_decoder = GalBfDecoder(h=h, max_iter=ldpc_iterations, info_idx=np.array([True] * k + [False] * m))
    galbf_entropy_decoder = ClassifyingEntropyDecoder(GalBfDecoder(
        h=h, max_iter=ldpc_iterations, info_idx=np.array([True] * k + [False] * m)),
        model_length=model_length, entropy_threshold=thr, clipping_factor=0, classifier_training=0, n_clusters=1,
        window_length=window_len, conf_center=args.conf_center, conf_slope=args.conf_slope, bit_flip=p, cluster=0,
        reliability_method=0)
    galbf_entropy_decoder_forcing = ClassifyingEntropyDecoder(GalBfDecoder(
        h=h, max_iter=ldpc_iterations, info_idx=np.array([True] * k + [False] * m)),
        model_length=model_length, entropy_threshold=thr/5, clipping_factor=0, classifier_training=0, n_clusters=1,
        window_length=window_len, conf_center=args.conf_center, conf_slope=args.conf_slope, bit_flip=p, cluster=0,
        reliability_method=0)

    decoded_galbf = []
    decoded_entropy_galbf = []
    decoded_entropy_galbf_forcing = []
    rx = np.zeros((n, encoder.n), dtype=np.bool_)

    step_results: dict[str, Any] = {'data': hc_bin_data[:n]}
    for tx_idx in range(n):
        corrupted = np.array(encoded[tx_idx])
        corrupted[errors[tx_idx]] = 1 - corrupted[errors[tx_idx]]
        channel_llr = channel(corrupted)
        rx[tx_idx] = corrupted
        d = galbf_decoder.decode(channel_llr)
        decoded_galbf.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        d = galbf_entropy_decoder.decode_buffer(channel_llr)
        decoded_entropy_galbf.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        d = galbf_entropy_decoder_forcing.decode_buffer(channel_llr)
        decoded_entropy_galbf_forcing.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        logger.info(f"p= {p}, tx id: {tx_idx}")
    logger.info(f"successful GalBF decoding for p= {p}, is: {sum(int(res[-1] == 0) for res in decoded_galbf)}/{n}")
    logger.info(f"successful GalBF-entropy decoding for p= {p}, is: "
                f"{sum(int(res[-1] == 0) for res in decoded_entropy_galbf)}/{n}")
    logger.info(f"successful GalBF-entropy-forcing decoding for p= {p}, is: "
                f"{sum(int(res[-1] == 0) for res in decoded_entropy_galbf_forcing)}/{n}")
    # log data
    info_errors = np.sum(errors < encoder.k, axis=1)
    parity_errors = np.sum(errors >= encoder.k, axis=1)
    step_results["raw_ber"] = no_errors / encoder.n
    zipped = [[np.array(en, dtype=np.int_), np.array(r, dtype=np.int_), er, inf, pa] for en, r, er, inf, pa in
              zip(encoded, rx, errors, info_errors, parity_errors)]
    step_results["data"] = pd.DataFrame(zipped, columns=["encoded", "corrupted", "error_idx", "info_errors",
                                                         "parity_errors"])
    # params
    step_results["buffer_len"] = len(encoded[0])
    step_results["number_of_buffers"] = n
    step_results["max_ldpc_iterations"] = ldpc_iterations

    # decoding
    decoded_galbf_df = pd.DataFrame(decoded_galbf, columns=["estimate", "decode_success", "iterations", "syndrome",
                                                            "vnode_validity", "hamming"])
    step_results['decoded_galbf'] = decoded_galbf_df
    decoded_entropy_galbf_df = pd.DataFrame(decoded_entropy_galbf, columns=["estimate", "llr", "decode_success", "iterations", "syndrome",
                                                            "vnode_validity", "dist", "structural_idx", "cluster_label", "hamming"])
    step_results['decoded_entropy_galbf'] = decoded_entropy_galbf_df
    decoded_entropy_galbf_forcing_df = pd.DataFrame(decoded_entropy_galbf_forcing, columns=["estimate", "llr", "decode_success", "iterations", "syndrome",
                                                            "vnode_validity", "dist", "structural_idx", "cluster_label", "hamming"])
    step_results['decoded_entropy_galbf_forcing'] = decoded_entropy_galbf_forcing_df

    # performance
    step_results["galbf_success_rate"] = sum(int(res[-1] == 0) for res in decoded_galbf) / float(n)
    step_results["galbf_decoder_ber"] = sum(res[-1] for res in decoded_galbf) / float(n * len(encoded[0]))
    step_results["galbf_entropy_success_rate"] = sum(int(res[-1] == 0) for res in decoded_entropy_galbf) / float(n)
    step_results["galbf_entropy_decoder_ber"] = sum(res[-1] for res in decoded_entropy_galbf) / float(n * len(encoded[0]))
    step_results["entropy_forcing_success_rate"] = sum(int(res[-1] == 0) for res in decoded_entropy_galbf_forcing) / float(n)
    step_results["entropy_forcing_decoder_ber"] = sum(res[-1] for res in decoded_entropy_galbf_forcing) / float(n * len(encoded[0]))

    timestamp = f'{str(datetime.date.today())}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}_' \
                f'{datetime.datetime.now().second}'
    with open(os.path.join("results/", f'{timestamp}_{p}_simulation_gallager_entropy.pickle'), 'wb') as f:
        pickle.dump(step_results, f)
    return step_results


if __name__ == '__main__':
    timestamp = f'{str(datetime.date.today())}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}_' \
                f'{datetime.datetime.now().second}'

    path = os.path.join("results/", timestamp)

    logger.info(__file__)
    logger.info(f"number of buffers to process: {n}")
    logger.info(f"number of processes: {args.processes}")
    logger.info(f"number of bit flips: {args.nflips}")
    logger.info(f"smallest f: {args.minflip}")
    logger.info(f"largest f: {args.maxflip}")
    logger.info(f"number of BF decoder iterations: {ldpc_iterations}")
    logger.info(f"entropy threshold: {thr}")
    logger.info(f"model center: {args.conf_center}")
    logger.info(f"model slope: {args.conf_slope}")
    logger.info(f"multiply data: {args.multiply_data}")
    logger.info(f"model length: {args.model_length}")
    logger.info(f"window length: {window_len}")

    cmd = f'python {__file__} --minflip {args.minflip} --maxflip {args.maxflip} --nflips {args.nflips} ' \
          f'--ldpciterations {ldpc_iterations} --ent_threshold {thr} --conf_center {args.conf_center} ' \
          f'--conf_slope {args.conf_slope} --multiply_data {args.multiply_data}  ' \
          f'--model_length {args.model_length}'

    if window_len is not None:
        cmd += f' --window_len {window_len}'
    else:
        cmd += ' --window_len 0'
    if args.N > 0:
        cmd += f' --N {n}'
    if processes is not None:
        cmd += f' --processes {processes}'

    os.mkdir(path)
    with open(os.path.join(path, "cmd.txt"), 'w') as f:
        f.write(cmd)
    logger.info(cmd)

    try:
        bit_flip_p = np.linspace(args.minflip, args.maxflip, num=args.nflips)

        # with Pool(processes=processes) as pool:
        #     results: list[dict[str, Any]] = pool.map(simulation_step, bit_flip_p)
        results: list[dict[str, Any]] = list(map(simulation_step, bit_flip_p))

        with lzma.open(
                os.path.join(path, f'{timestamp}_simulation_gallger_entropy_decoder.xz'),
                "wb") as f:
            pickle.dump(results, f)
        logger.info("saved compressed results file")

        raw_ber = np.array([p['raw_ber'] for p in results])
        galbf_decoder_ber = np.array([p['galbf_decoder_ber'] for p in results])
        galbf_entropy_decoder_ber = np.array([p['galbf_entropy_decoder_ber'] for p in results])
        galbf_entropy_forcing_decoder_ber = np.array([p['entropy_forcing_decoder_ber'] for p in results])
        fig = plt.figure()
        plt.plot(raw_ber, galbf_decoder_ber, 'bo', raw_ber, galbf_entropy_decoder_ber, 'g^', raw_ber, galbf_entropy_forcing_decoder_ber, 'r*')
        plt.xlabel("BSC bit flip probability p")
        plt.ylabel("post decoding BER")
        fig.savefig(os.path.join(path, "ber_vs_error_p.eps"), dpi=150)

        figure = plt.figure()
        galbf_buffer_success_rate = np.array([p['galbf_success_rate'] for p in results])
        galbf_entropy_buffer_success_rate = np.array([p['galbf_entropy_success_rate'] for p in results])
        galbf_entropy_forcing_buffer_success_rate = np.array([p['entropy_forcing_success_rate'] for p in results])
        plt.plot(raw_ber, galbf_buffer_success_rate, 'bo', raw_ber, galbf_entropy_buffer_success_rate, 'g^', raw_ber, galbf_entropy_forcing_buffer_success_rate, 'r*')
        plt.xlabel("BSC bit flip probability p")
        plt.ylabel("Decode success rate")
        figure.savefig(os.path.join(path, "success_rate_vs_error_p.eps"), dpi=150)
        logger.info("saved figures")

        summary = {"args": args, "raw_ber": raw_ber, "galbf_decoder_ber": galbf_decoder_ber,
                   "galbf_entropy_decoder_ber": galbf_entropy_decoder_ber,
                   "entropy_forcing_decoder_ber": galbf_entropy_forcing_decoder_ber,
                   "galbf_buffer_success_rate": galbf_buffer_success_rate,
                   "entropy_buffer_success_rate": galbf_entropy_buffer_success_rate,
                   "entropy_forcing_buffer_success": galbf_entropy_forcing_buffer_success_rate}

        with open(os.path.join(path, f'{timestamp}_summary_gal_entropy_decoder.pickle'), 'wb') as f:
            pickle.dump(summary, f)
        savemat(os.path.join(path, f'{timestamp}_summary_gal_entropy_decoder.mat'), summary)
        logger.info("saved summary")

        for step in results:
            step['data'] = step['data'].to_dict("list")
            step['decoded_galbf'] = step['decoded_galbf'].to_dict("list")
            step['decoded_entropy_galbf'] = step['decoded_entropy_galbf'].to_dict("list")
            step['decoded_entropy_galbf_forcing'] = step['decoded_entropy_galbf_forcing'].to_dict("list")

        summary["results"] = results
        savemat(os.path.join(path, f'{timestamp}_simulation_gal_entropy_decoder.mat'),
                summary, do_compression=True)
        logger.info("saved results to mat file")
        shutil.move("results/log.log", os.path.join(path, "log.log"))
    except Exception as e:
        logger.exception(e)
        shutil.move("results/log.log", os.path.join(path, "log.log"))
        raise e
