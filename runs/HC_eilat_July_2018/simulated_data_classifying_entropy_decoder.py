import pickle
from bitstring import Bits
import numpy as np
from ldpc.encoder import EncoderWiFi
from ldpc.wifi_spec_codes import WiFiSpecCode
from ldpc.decoder import bsc_llr, DecoderWiFi, WbfDecoder, WbfVariant, awgn_llr
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
from scipy.special import erfc

parser = argparse.ArgumentParser(description='Run decoding on simulated data using multiprocessing.')
parser.add_argument("--N", default=0, help="max number of transmissions to consider", type=int)
parser.add_argument("--minflip", default=33 * 1e-3, help="minimal bit flip probability to consider", type=float)
parser.add_argument("--maxflip", default=70 * 1e-3, help="maximal bit flip probability to consider", type=float)
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
parser.add_argument("--classifier_train", default=100, help="number of buffers to use for classifier training", type=int)
parser.add_argument("--n_clusters", default=1, help="number of clusters", type=int)
parser.add_argument("--msg_delay", default="50000", help="sampling delay", type=str)
parser.add_argument("--cluster", default=1, help="enable or disable clustering", type=int)
parser.add_argument("--model_length", default="info", help="model length", type=str)
parser.add_argument("--wbf_variant", default=0, help="which wbf variant to use", type=int)
parser.add_argument("--channel_type", default="bsc", help="which chanel model", type=str)

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

if args.n_clusters == 1:
    spec = WiFiSpecCode.N1944_R23
    args.classifier_train = 0
elif args.n_clusters == 2:
    spec = WiFiSpecCode.N1296_R12
elif args.n_clusters == 3:
    spec = WiFiSpecCode.N648_R34
encoder = EncoderWiFi(spec=spec)
h = encoder.h
encoded = []
for binary_data in hc_bin_data[:n]:
    if args.n_clusters == 1:
        pad_len = encoder.k - len(binary_data)
        padded = np.array(binary_data + Bits(auto=rng.integers(low=0, high=2, size=pad_len)), dtype=np.int_)
        encoded.append(encoder.encode(padded))
    elif args.n_clusters == 2:
        padded = np.array(binary_data[:576] + Bits(auto=rng.integers(low=0, high=2, size=encoder.k - 576)), dtype=np.int_)
        encoded.extend(
            (
                encoder.encode(padded),
                encoder.encode(np.array(binary_data[576:], dtype=np.int_)),
            )
        )
    elif args.n_clusters == 3:
        padded = np.array(binary_data[:416] + Bits(auto=rng.integers(low=0, high=2, size=encoder.k - 416)), dtype=np.int_)
        encoded.append(encoder.encode(padded))
        padded = np.array(binary_data[416:864] + Bits(auto=rng.integers(low=0, high=2, size=encoder.k - (864 - 416))),
                          dtype=np.int_)
        encoded.append(encoder.encode(padded))
        padded = np.array(binary_data[864:] + Bits(auto=rng.integers(low=0, high=2, size=encoder.k - (1224 - 864))),
                          dtype=np.int_)
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
    global clipping_factor
    global args
    global window_len
    global n
    global spec
    global encoded
    global h

    if args.channel_type == "bsc":
        channel = bsc_llr(p=p)
        no_errors = int(encoder.n * p)
        errors = np.vstack(
            tuple(rng.choice(encoder.n, size=no_errors, replace=False)
                  for _ in range(n))
        )
    else:
        snr_linear = p
        noise_power = 1 / snr_linear
        sigma = np.sqrt(noise_power / 2)
        channel = awgn_llr(sigma=sigma)
        p = 0.5 * erfc(np.sqrt(snr_linear))
        errors = np.zeros((n, encoder.n), dtype=np.bool_)

    if args.dec_type in {"BP", "MS"}:
        ldpc_decoder = DecoderWiFi(spec=spec, max_iter=ldpc_iterations, decoder_type=args.dec_type)
        entropy_decoder = ClassifyingEntropyDecoder(DecoderWiFi(spec=spec, max_iter=ldpc_iterations,
                                                                decoder_type=args.dec_type),
                                                    model_length=model_length, entropy_threshold=thr,
                                                    clipping_factor=clipping_factor, classifier_training=args.classifier_train,
                                                    n_clusters=args.n_clusters, window_length=window_len,
                                                    conf_center=args.conf_center,
                                                    conf_slope=args.conf_slope, bit_flip=p, cluster=args.cluster)
    elif args.dec_type == "BF":
        m, n = h.shape
        k = n - m
        ldpc_decoder = WbfDecoder(h=h, max_iter=ldpc_iterations, decoder_variant=WbfVariant(args.wbf_variant),
                                  info_idx=np.array([True] * k + [False] * m))
        entropy_decoder = ClassifyingEntropyDecoder(WbfDecoder(h=h, max_iter=ldpc_iterations, decoder_variant=WbfVariant(
            args.wbf_variant), info_idx=np.array([True] * k + [False] * m)),
                                                    model_length=model_length, entropy_threshold=thr,
                                                    clipping_factor=clipping_factor, classifier_training=args.classifier_train,
                                                    n_clusters=args.n_clusters, window_length=window_len,
                                                    conf_center=args.conf_center, conf_slope=args.conf_slope, bit_flip=p,
                                                    cluster=args.cluster, sigma=sigma)
    else:
        raise ValueError('Incorrect decoder type')
    decoded_ldpc = []
    decoded_entropy = []
    rx = np.zeros((n, encoder.n), dtype=np.bool_)

    step_results: dict[str, Any] = {'data': hc_bin_data[:n]}
    for tx_idx in range(n):
        if args.channel_type == "bsc":
            corrupted = encoded[tx_idx]
            corrupted[errors[tx_idx]] = 1 - corrupted[errors[tx_idx]]
            channel_llr = channel(corrupted)
        else:
            # baseband modulate: x_i = 1 - 2 * c_i, i.e. map 0 to 1 and 1 to -1
            baseband = 1 - 2 * np.array(encoded[tx_idx], dtype=np.int_)
            # Generate noise
            noise = sigma * rng.normal(size=len(baseband))
            # channel: y_i = x_i + n_i, i.e. add noise
            noisy = baseband + noise
            channel_llr = channel(noisy)
            corrupted = noisy < 0
            errors[tx_idx] = encoded[tx_idx] ^ corrupted
        rx[tx_idx] = corrupted
        d = ldpc_decoder.decode(channel_llr)
        decoded_ldpc.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        d = entropy_decoder.decode_buffer(channel_llr)
        decoded_entropy.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        logger.info(f"p= {p}, tx id: {tx_idx}")
    logger.info(f"successful BP decoding for p= {p}, is: {sum(int(res[-1] == 0) for res in decoded_ldpc)}/{n}")
    logger.info(f"successful entropy decoding for p= {p}, is: {sum(int(res[-1] == 0) for res in decoded_entropy)}/{n}")
    # log data
    if args.channel_type == "bsc":
        info_errors = np.sum(errors < encoder.k, axis=1)
        parity_errors = np.sum(errors >= encoder.k, axis=1)
        step_results["raw_ber"] = no_errors / encoder.n
    else:
        info_errors = np.sum(errors[:, :encoder.k], axis=1)
        parity_errors = np.sum(errors[:, encoder.k:], axis=1)
        step_results["raw_ber"] = np.sum(errors) / encoder.n / n
    zipped = [[np.array(en, dtype=np.int_), np.array(r, dtype=np.int_), er, inf, pa] for en, r, er, inf, pa in
              zip(encoded, rx, errors, info_errors, parity_errors)]
    step_results["data"] = pd.DataFrame(zipped, columns=["encoded", "corrupted", "error_idx", "info_errors",
                                                         "parity_errors"])
    # params
    step_results["buffer_len"] = len(encoded[0])
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

    timestamp = f'{str(datetime.date.today())}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}_' \
                f'{datetime.datetime.now().second}'
    with open(os.path.join("results/", f'{timestamp}_{p}_simulation_classifying_entropy.pickle'), 'wb') as f:
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
    logger.info(f"smallest bit flip probability: {args.minflip}")
    logger.info(f"largest bit flip probability: {args.maxflip}")
    logger.info(f"number of ldpc decoder iterations: {ldpc_iterations}")
    logger.info(f"entropy threshold: {thr}")
    logger.info(f"clipping factor: {clipping_factor}")
    logger.info(f"model center: {args.conf_center}")
    logger.info(f"model slope: {args.conf_slope}")
    logger.info(f"multiply data: {args.multiply_data}")
    logger.info(f"decoder type: {args.dec_type}")
    logger.info(f"classifier train: {args.classifier_train}")
    logger.info(f"number of clusters: {args.n_clusters}")
    logger.info(f"message delay: {args.msg_delay}")
    logger.info(f"cluster: {args.cluster}")
    logger.info(f"model length: {args.model_length}")
    logger.info(f"window length: {window_len}")
    logger.info(f"wbf variant: {args.wbf_variant}")
    logger.info(f"channel_type: {args.channel_type}")

    cmd = f'python {__file__} --minflip {args.minflip} --maxflip {args.maxflip} --nflips {args.nflips} --ldpciterations ' \
          f'{ldpc_iterations} --ent_threshold {thr} --clipping_factor {clipping_factor} --conf_center {args.conf_center} ' \
          f'--conf_slope {args.conf_slope} --multiply_data {args.multiply_data} --dec_type {args.dec_type}  ' \
          f'--classifier_train {args.classifier_train} --n_clusters {args.n_clusters} --msg_delay {args.msg_delay} ' \
          f'--cluster {args.cluster} --model_length {args.model_length} --wbf_variant {args.wbf_variant} ' \
          f'--channel_type {args.channel_type}'

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
        if args.channel_type == "bsc":
            bit_flip_p = np.linspace(args.minflip, args.maxflip, num=args.nflips)
            map_arg = bit_flip_p
        else:
            snr_db = np.linspace(args.min_ebno, args.max_ebno, num=args.n_points)
            snr_linear = np.array([10 ** (snr / 10) for snr in snr_db])
            map_arg = snr_linear

        # with Pool(processes=processes) as pool:
        #     results: list[dict[str, Any]] = pool.map(simulation_step, map_arg)
        results: list[dict[str, Any]] = list(map(simulation_step, map_arg))

        with lzma.open(
                os.path.join(path, f'{timestamp}_simulation_classifying_entropy_{args.dec_type}_decoder.xz'),
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
                   "entropy_buffer_success_rate": entropy_buffer_success_rate, "channel_type": args.channel_type}
        if args.channel_type != "bsc":
            summary['snr_linear'] = snr_linear
            summary['snr_db'] = snr_db
        with open(os.path.join(path, f'{timestamp}_summary_classifying_entropy__{args.dec_type}_decoder.pickle'), 'wb') as f:
            pickle.dump(summary, f)
        savemat(os.path.join(path, f'{timestamp}_summary_classifying_entropy__{args.dec_type}_decoder.mat'),
                summary)
        logger.info("saved summary")

        for step in results:
            step['data'] = step['data'].to_dict("list")
            step['decoded_entropy'] = step['decoded_entropy'].to_dict("list")
            step['decoded_ldpc'] = step['decoded_ldpc'].to_dict("list")

        summary["results"] = results
        savemat(os.path.join(path, f'{timestamp}_simulation_classifying_entropy_{args.dec_type}_decoder.mat'),
                summary, do_compression=True)
        logger.info("saved results to mat file")
        shutil.move("results/log.log", os.path.join(path, "log.log"))
    except Exception as e:
        logger.exception(e)
        shutil.move("results/log.log", os.path.join(path, "log.log"))
        raise e
