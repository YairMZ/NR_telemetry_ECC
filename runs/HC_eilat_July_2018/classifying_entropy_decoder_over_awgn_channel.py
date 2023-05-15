import pickle
from bitstring import Bits
import numpy as np
from ldpc.encoder import EncoderWiFi
from ldpc.wifi_spec_codes import WiFiSpecCode
from ldpc.decoder import DecoderWiFi
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
from scipy.special import erfc  # erfc/Q function
import shutil


def awgn_llr(sigma: float):
    """
    awgn llr is defined as:
        x_i = 1 - 2 * c_i
        y_i = x_i + n_i
        L(y_i) = 2 * y_i / sigma^2
    :param float sigma: the llr is parameterized by the standard deviation of the noise sigma.
    :returns: return a callable which accepts a single argument - y_i noisy channel symbol, and returns its llr
    """
    return lambda y: 2 * y / np.power(sigma, 2)  # type: ignore


parser = argparse.ArgumentParser(description='Run decoding on simulated data using multiprocessing.')
parser.add_argument("--N", default=0, help="max number of transmissions to consider", type=int)
parser.add_argument("--min_ebno", default=-2, help="minimal snr to consider", type=float)
parser.add_argument("--max_ebno", default=3, help="maximal snr to consider", type=float)
parser.add_argument("--snr_scale", default="db", help="scale of snr", type=str)
parser.add_argument("--delta", default=1.0, help="scaling of p", type=float)
parser.add_argument("--n_points", default=20, help="number of bit flips to consider", type=int)
parser.add_argument("--ldpciterations", default=20, help="number of iterations of LDPC decoder", type=int)
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
parser.add_argument("--model_length", default="all", help="model length", type=str)

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

# init rng
rng = np.random.default_rng()

if args.n_clusters == 1:
    spec = WiFiSpecCode.N1944_R23
    args.classifier_train = 0
elif args.n_clusters == 2:
    spec = WiFiSpecCode.N1296_R12
elif args.n_clusters == 3:
    spec = WiFiSpecCode.N648_R34
else:
    raise ValueError("Invalid number of clusters")

encoder = EncoderWiFi(spec=spec)
encoded = []
for binary_data in hc_bin_data[:n]:
    if args.n_clusters == 1:
        pad_len = encoder.k - len(binary_data)
        padded = binary_data + Bits(auto=rng.integers(low=0, high=2, size=pad_len))
        encoded.append(encoder.encode(padded))
    elif args.n_clusters == 2:
        padded = binary_data[:576] + Bits(auto=rng.integers(low=0, high=2, size=encoder.k-576))
        encoded.extend((encoder.encode(padded), encoder.encode(binary_data[576:])))
    elif args.n_clusters == 3:
        padded = binary_data[:416] + Bits(auto=rng.integers(low=0, high=2, size=encoder.k-416))
        encoded.append(encoder.encode(padded))
        padded = binary_data[416:864] + Bits(auto=rng.integers(low=0, high=2, size=encoder.k - (864-416)))
        encoded.append(encoder.encode(padded))
        padded = binary_data[864:] + Bits(auto=rng.integers(low=0, high=2, size=encoder.k - (1224-864)))
        encoded.append(encoder.encode(padded))

for _ in range(args.multiply_data):  # generate more buffers for statistical reproducibility
    encoded.extend(encoded)

model_length = encoder.k if args.model_length == 'info' else encoder.n
n = len(encoded)  # redefine n

# encoded structure: {starting byte index in buffer: msg_id}
# {0: 33, 52: 234, 72: 30, 108: 212, 135: 218}
# last 9 bytes (72 bits) are padding and not messages. Thus last message ends at byte index 152
# bit indices:
# {0: 33, 416: 234, 576: 30, 864: 212, 1080: 218}

logger = setup_logger(name=__file__, log_file=os.path.join("results/", 'log.log'))


def simulation_step(snr: float) -> dict[str, Any]:
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

    if args.snr_scale == 'db':
        snr_linear = 10 ** (snr / 10)
    elif args.snr_scale == 'linear':
        snr_linear = snr
    else:
        raise ValueError("Invalid snr scale")
    noise_power = 1 / snr_linear
    sigma = np.sqrt(noise_power/2)
    channel = awgn_llr(sigma=sigma)
    # delta = args.delta if snr > 0 else 1
    # p = 0.5 * erfc(np.sqrt(10 ** (delta * snr / 10)))
    delta = args.delta
    p = 0.5 * erfc(np.sqrt(delta * snr_linear))
    ldpc_decoder = DecoderWiFi(spec=spec, max_iter=ldpc_iterations, decoder_type=args.dec_type)
    entropy_decoder = ClassifyingEntropyDecoder(DecoderWiFi(spec=spec, max_iter=ldpc_iterations, decoder_type=args.dec_type),
                                                model_length=model_length, entropy_threshold=thr,
                                                clipping_factor=clipping_factor, classifier_training=args.classifier_train,
                                                n_clusters=args.n_clusters, window_length=window_len,
                                                conf_center=args.conf_center,
                                                conf_slope=args.conf_slope, bit_flip=p, cluster=args.cluster)
    rx = np.zeros((n, encoder.n), dtype=np.bool_)
    errors = np.zeros((n, encoder.n), dtype=np.bool_)
    decoded_ldpc = []
    decoded_entropy = []
    step_results: dict[str, Any] = {'data': hc_bin_data[:n]}
    for tx_idx in range(n):
        # baseband modulate: x_i = 1 - 2 * c_i, i.e. map 0 to 1 and 1 to -1
        baseband = 1 - 2*np.array(encoded[tx_idx], dtype=np.int_)
        # Generate noise
        noise = sigma * rng.normal(size=len(baseband))
        # channel: y_i = x_i + n_i, i.e. add noise
        noisy = baseband + noise
        channel_llr = channel(noisy)
        corrupted = noisy < 0
        rx[tx_idx] = corrupted
        tx_bool = np.array(encoded[tx_idx], dtype=np.bool_)
        errors[tx_idx] = tx_bool ^ corrupted
        d = ldpc_decoder.decode(channel_llr)
        decoded_ldpc.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        d = entropy_decoder.decode_buffer(channel_llr)
        decoded_entropy.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        logger.info(f"snr= {snr}, tx id: {tx_idx}")
    bp_success = sum(int(res[-1] == 0) for res in decoded_ldpc)
    entropy_success = sum(int(r[-1] == 0) for r in decoded_entropy)
    logger.info(f"successful BP decoding for snr= {snr}, is: {bp_success}/{n}")
    logger.info(f"successful entropy decoding for snr= {snr}, is: {entropy_success}/{n}")
    if n-bp_success > 0:
        logger.info(f"NR recovery rate for snr= {snr}, is: {(entropy_success-bp_success)/(n-bp_success)}")
    # log data
    info_errors = np.sum(errors < encoder.k, axis=1)
    parity_errors = np.sum(errors >= encoder.k, axis=1)
    zipped = [[np.array(en, dtype=np.int_), r, er, inf, pa] for en, r, er, inf, pa in
              zip(encoded, rx, errors, info_errors, parity_errors)]
    step_results["data"] = pd.DataFrame(zipped, columns=["encoded", "corrupted", "error_idx", "info_errors",
                                                         "parity_errors"])

    # params
    step_results["mean_raw_ber"] = np.sum(errors) / encoder.n / n
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
    # performance
    step_results["ldpc_buffer_success_rate"] = sum(int(res[-1] == 0) for res in decoded_ldpc) / float(n)
    step_results["ldpc_decoder_ber"] = sum(res[-1] for res in decoded_ldpc) / float(n * len(encoded[0]))
    step_results["entropy_buffer_success_rate"] = sum(int(res[-1] == 0) for res in decoded_entropy) / float(n)
    step_results["entropy_decoder_ber"] = sum(res[-1] for res in decoded_entropy) / float(n * len(encoded[0]))

    timestamp = f'{str(datetime.date.today())}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}_' \
                f'{datetime.datetime.now().second}'
    with open(os.path.join("results/", f'{timestamp}_{snr}_simulation_awgn_classifying_entropy.pickle'), 'wb') as f:
        pickle.dump(step_results, f)
    return step_results


if __name__ == '__main__':

    timestamp = f'{str(datetime.date.today())}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}_' \
                f'{datetime.datetime.now().second}'

    path = os.path.join("results/", timestamp)

    logger.info(__file__)
    logger.info(f"number of buffers to process: {n}")
    logger.info(f"smallest EbN0: {args.min_ebno}")
    logger.info(f"largest EbN0: {args.max_ebno}")
    logger.info(f"snr scale: {args.snr_scale}")
    logger.info(f"delta: {args.delta}")
    logger.info(f"number of points: {args.n_points}")
    logger.info(f"number of ldpc decoder iterations: {ldpc_iterations}")
    logger.info(f"processes: {args.processes}")
    logger.info(f"multiply data: {args.multiply_data}")
    logger.info(f"msg_delay: {args.msg_delay} ")
    logger.info(f"decoder type: {args.dec_type}")
    logger.info(f"classifier_train: {args.classifier_train}")
    logger.info(f"n_clusters: {args.n_clusters}")
    logger.info(f"cluster: {args.cluster}")
    logger.info(f"decoder window length: {window_len}")
    logger.info(f"confidence center: {args.conf_center}")
    logger.info(f"confidence slope: {args.conf_slope}")
    logger.info(f"entropy threshold: {thr}")
    logger.info(f"clipping factor: {clipping_factor}")
    logger.info(f"model length: {args.model_length}")

    cmd = f'python {__file__} --min_ebno {args.min_ebno} --max_ebno {args.max_ebno} --n_points {args.n_points}  ' \
          f'--snr_scale {args.snr_scale} --ldpciterations {ldpc_iterations} --multiply_data {args.multiply_data} ' \
          f'--msg_delay {args.msg_delay} --dec_type {args.dec_type} --classifier_train {args.classifier_train} ' \
          f'--n_clusters {args.n_clusters} --cluster {args.cluster} --delta {args.delta} ' \
          f'--conf_center {args.conf_center} --model_length {args.model_length} ' \
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
        snr_list = np.linspace(args.min_ebno, args.max_ebno, num=args.n_points)
        with Pool(processes=processes) as pool:
            results: list[dict[str, Any]] = pool.map(simulation_step, snr_list)
        # results: list[dict[str, Any]] = list(map(simulation_step, snr_list))

        with lzma.open(
                os.path.join(path, f'{timestamp}_simulation_AWGN_classifying_entropy_{args.dec_type}_decoder.xz'),
                "wb") as f:
            pickle.dump(results, f)
        logger.info("saved compressed results file")

        x_label = "SNR (Eb/N0) [dB]" if args.snr_scale == "db" else "SNR (Eb/N0) - linear scale"
        ldpc_ber = np.array([p['ldpc_decoder_ber'] for p in results])
        entropy_ber = np.array([p['entropy_decoder_ber'] for p in results])
        fig = plt.figure()
        plt.plot(snr_list, ldpc_ber, 'bo', snr_list, entropy_ber, 'r*')
        plt.xlabel(x_label)
        plt.ylabel("post decoding BER")
        fig.savefig(os.path.join(path, "ber_vs_snr.eps"), dpi=150)

        figure = plt.figure()
        ldpc_buffer_success_rate = np.array([p['ldpc_buffer_success_rate'] for p in results])
        entropy_buffer_success_rate = np.array([p['entropy_buffer_success_rate'] for p in results])
        plt.plot(snr_list, ldpc_buffer_success_rate, 'bo', snr_list, entropy_buffer_success_rate, 'r*')
        plt.xlabel(x_label)
        plt.ylabel("Decode success rate")
        figure.savefig(os.path.join(path, "buffer_success_rate_vs_snr.eps"), dpi=150)
        logger.info("saved figures")

        summary = {"args": args, "snr": snr_list, "ldpc_ber": ldpc_ber, "entropy_ber": entropy_ber,
                   "ldpc_buffer_success_rate": ldpc_buffer_success_rate,
                   "entropy_buffer_success_rate": entropy_buffer_success_rate}
        with open(os.path.join(path, f'{timestamp}_summary_classifying_entropy__{args.dec_type}_decoder.pickle'), 'wb') as f:
            pickle.dump(summary, f)

        savemat(os.path.join(path, f'{timestamp}_summary_AWGN_classifying_entropy__{args.dec_type}_decoder.mat'),
                summary)
        logger.info("saved summary")

        for step in results:
            step['data'] = step['data'].to_dict("list")
            step['decoded_entropy'] = step['decoded_entropy'].to_dict("list")
            step['decoded_ldpc'] = step['decoded_ldpc'].to_dict("list")
        # for i in range(len(results)):
            # del results[i]['data']
            # del results[i]['decoded_entropy']
        summary["results"] = results
        savemat(os.path.join(path, f'{timestamp}_simulation_AWGN_classifying_entropy_{args.dec_type}_decoder.mat'),
                summary, do_compression=True)
        logger.info("saved results to mat file")
        shutil.move("results/log.log", os.path.join(path, "log.log"))
    except Exception as e:
        logger.exception(e)
        shutil.move("results/log.log", os.path.join(path, "log.log"))
        raise e
