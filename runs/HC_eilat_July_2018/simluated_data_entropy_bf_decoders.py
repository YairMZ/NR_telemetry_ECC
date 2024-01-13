import pickle
from bitstring import Bits
import numpy as np
from ldpc.encoder import EncoderWiFi
from ldpc.wifi_spec_codes import WiFiSpecCode
from ldpc.decoder import DecoderWiFi, WbfDecoder, WbfVariant, awgn_llr
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
parser.add_argument("--ldpciterations", default=300, help="number of iterations of BF LDPC decoder", type=int)
parser.add_argument("--ent_threshold", default=0.36, help="entropy threshold to be used in entropy decoder", type=float)
parser.add_argument("--window_len", default=50, help="number of previous samples to use, if 0 all are used", type=int)
parser.add_argument("--multiply_data", default=0, help="multiplies amount of buffers by 2 to power of arg", type=int)
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

    snr_linear = p
    noise_power = 1 / snr_linear
    sigma = np.sqrt(noise_power / 2)
    channel = awgn_llr(sigma=sigma)
    p = 0.5 * erfc(np.sqrt(snr_linear))
    errors = np.zeros((n, encoder.n), dtype=np.bool_)

    ms_decoder = DecoderWiFi(spec=spec, max_iter=20, decoder_type="MS")
    m, nn = h.shape
    k = nn - m
    ldpc_iterations = max(int(3*p*nn), ldpc_iterations)
    wbf_decoder = WbfDecoder(h=h, max_iter=ldpc_iterations, decoder_variant=WbfVariant.WBF,
                              info_idx=np.array([True] * k + [False] * m))
    wbf_entropy_decoder = ClassifyingEntropyDecoder(
        WbfDecoder(h=h,max_iter=ldpc_iterations, decoder_variant=WbfVariant.WBF,
                   info_idx=np.array([True] * k + [False] * m)),
        model_length=model_length, entropy_threshold=thr, clipping_factor=0, classifier_training=0, n_clusters=1,
        window_length=window_len, conf_center=args.conf_center, conf_slope=args.conf_slope, bit_flip=p, cluster=0, reliability_method=0)
    mwbf_decoder = WbfDecoder(h=h, max_iter=ldpc_iterations, decoder_variant=WbfVariant.MWBF,
                              info_idx=np.array([True] * k + [False] * m))
    mwbf_entropy_decoder = ClassifyingEntropyDecoder(
        WbfDecoder(h=h, max_iter=ldpc_iterations, decoder_variant=WbfVariant.MWBF,
                   info_idx=np.array([True] * k + [False] * m)),
        model_length=model_length, entropy_threshold=thr, clipping_factor=0, classifier_training=0, n_clusters=1,
        window_length=window_len, conf_center=args.conf_center, conf_slope=args.conf_slope, bit_flip=p, cluster=0, reliability_method=0)
    mwbf_decoder_no_loop = WbfDecoder(h=h, max_iter=ldpc_iterations, decoder_variant=WbfVariant.MWBF_NO_LOOPS,
                                      info_idx=np.array([True] * k + [False] * m))
    mwbf_entropy_decoder_no_loop = ClassifyingEntropyDecoder(
        WbfDecoder(h=h, max_iter=ldpc_iterations, decoder_variant=WbfVariant.MWBF_NO_LOOPS,
                   info_idx=np.array([True] * k + [False] * m)),
        model_length=model_length, entropy_threshold=thr, clipping_factor=0, classifier_training=0, n_clusters=1,
        window_length=window_len, conf_center=args.conf_center, conf_slope=args.conf_slope, bit_flip=p, cluster=0, reliability_method=0)

    decoded_ms = []
    decoded_wbf = []
    decoded_entropy_wbf = []
    decoded_mwbf = []
    decoded_entropy_mwbf = []
    decoded_mwbf_no_loop = []
    decoded_entropy_mwbf_no_loop = []

    rx = np.zeros((n, encoder.n), dtype=np.bool_)

    step_results: dict[str, Any] = {'data': hc_bin_data[:n]}
    for tx_idx in range(n):
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
        d = ms_decoder.decode(channel_llr)
        decoded_ms.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        d = wbf_decoder.decode(noisy)
        decoded_wbf.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        d = wbf_entropy_decoder.decode_buffer(noisy)
        decoded_entropy_wbf.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        d = mwbf_decoder.decode(noisy)
        decoded_mwbf.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        d = mwbf_entropy_decoder.decode_buffer(noisy)
        decoded_entropy_mwbf.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        d = mwbf_decoder_no_loop.decode(noisy)
        decoded_mwbf_no_loop.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        d = mwbf_entropy_decoder_no_loop.decode_buffer(noisy)
        decoded_entropy_mwbf_no_loop.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        logger.info(f"p= {p}, tx id: {tx_idx}")
    logger.info(f"successful MS decoding for p= {p}, is: {sum(int(res[-1] == 0) for res in decoded_ms)}/{n}")
    logger.info(f"successful WBF decoding for p= {p}, is: {sum(int(res[-1] == 0) for res in decoded_wbf)}/{n}")
    logger.info(
        f"successful WBF-entropy decoding for p= {p}, is: {sum(int(res[-1] == 0) for res in decoded_entropy_wbf)}/{n}")
    logger.info(f"successful MWBF decoding for p= {p}, is: {sum(int(res[-1] == 0) for res in decoded_mwbf)}/{n}")
    logger.info(f"successful MWBF-entropy decoding for p= {p}, is: "
                f"{sum(int(res[-1] == 0) for res in decoded_entropy_mwbf)}/{n}")
    logger.info(f"successful MWBF No Loops decoding for p= {p}, is: "
                f"{sum(int(res[-1] == 0) for res in decoded_mwbf_no_loop)}/{n}")
    logger.info(f"successful MWBF No Loop-entropy decoding for p= {p}, is: "
                f"{sum(int(res[-1] == 0) for res in decoded_entropy_mwbf_no_loop)}/{n}")
    # log data
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
    decoded_ms_df = pd.DataFrame(decoded_ms,
                                 columns=["estimate", "llr", "decode_success", "iterations", "syndrome",
                                          "vnode_validity", "hamming"])

    step_results['decoded_ms'] = decoded_ms_df


    decoded_wbf_df = pd.DataFrame(decoded_wbf, columns=["estimate", "decode_success", "iterations", "syndrome",
                                          "vnode_validity", "hamming"])
    step_results['decoded_wbf'] = decoded_wbf_df
    decoded_entropy_wbf_df = pd.DataFrame(decoded_entropy_wbf, columns=["estimate", "llr", "decode_success", "iterations", "syndrome",
                                               "vnode_validity", "dist", "structural_idx", "cluster_label", "hamming"])
    step_results['decoded_entropy_wbf'] = decoded_entropy_wbf_df


    decoded_mwbf_df = pd.DataFrame(decoded_mwbf, columns=["estimate", "decode_success", "iterations", "syndrome",
                                                        "vnode_validity", "hamming"])
    step_results['decoded_mwbf'] = decoded_mwbf_df
    decoded_entropy_mwbf_df = pd.DataFrame(decoded_entropy_mwbf,
                                          columns=["estimate", "llr", "decode_success", "iterations", "syndrome",
                                                   "vnode_validity", "dist", "structural_idx", "cluster_label",
                                                   "hamming"])
    step_results['decoded_entropy_mwbf'] = decoded_entropy_mwbf_df

    decoded_mwbf_no_loop_df = pd.DataFrame(decoded_mwbf_no_loop, columns=["estimate", "decode_success", "iterations", "syndrome",
                                                          "vnode_validity", "hamming"])
    step_results['decoded_no_loop'] = decoded_mwbf_no_loop_df
    decoded_entropy_mwbf_no_loop_df = pd.DataFrame(decoded_entropy_mwbf_no_loop,
                                           columns=["estimate", "llr", "decode_success", "iterations", "syndrome",
                                                    "vnode_validity", "dist", "structural_idx", "cluster_label",
                                                    "hamming"])
    step_results['decoded_entropy_no_loop'] = decoded_entropy_mwbf_no_loop_df

    # performance
    step_results["ms_success_rate"] = sum(int(res[-1] == 0) for res in decoded_ms) / float(n)
    step_results["ms_decoder_ber"] = sum(res[-1] for res in decoded_ms) / float(n * len(encoded[0]))
    step_results["wbf_success_rate"] = sum(int(res[-1] == 0) for res in decoded_wbf) / float(n)
    step_results["wbf_decoder_ber"] = sum(res[-1] for res in decoded_wbf) / float(n * len(encoded[0]))
    step_results["mwbf_success_rate"] = sum(int(res[-1] == 0) for res in decoded_mwbf) / float(n)
    step_results["mwbf_decoder_ber"] = sum(res[-1] for res in decoded_mwbf) / float(n * len(encoded[0]))
    step_results["no_loop_success_rate"] = sum(int(res[-1] == 0) for res in decoded_mwbf_no_loop) / float(n)
    step_results["no_loop_decoder_ber"] = sum(res[-1] for res in decoded_mwbf_no_loop) / float(n * len(encoded[0]))
    step_results["entropy_wbf_success_rate"] = sum(int(res[-1] == 0) for res in decoded_entropy_wbf) / float(n)
    step_results["entropy_wbf_decoder_ber"] = sum(res[-1] for res in decoded_entropy_wbf) / float(n * len(encoded[0]))
    step_results["entropy_mwbf_success_rate"] = sum(int(res[-1] == 0) for res in decoded_entropy_mwbf) / float(n)
    step_results["entropy_mwbf_decoder_ber"] = sum(res[-1] for res in decoded_entropy_mwbf) / float(n * len(encoded[0]))
    step_results["entropy_no_loop_success_rate"] = sum(int(res[-1] == 0) for res in decoded_entropy_mwbf_no_loop) / float(n)
    step_results["entropy_no_loop_decoder_ber"] = sum(res[-1] for res in decoded_entropy_mwbf_no_loop) / float(n * len(encoded[0]))


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
    logger.info(f"smallest EbN0: {args.minflip}")
    logger.info(f"largest EbN0: {args.maxflip}")
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
        snr_db = np.linspace(args.minflip, args.maxflip, num=args.nflips)
        snr_linear = np.array([10 ** (snr / 10) for snr in snr_db])

        with Pool(processes=processes) as pool:
            results: list[dict[str, Any]] = pool.map(simulation_step, snr_linear)
        # results: list[dict[str, Any]] = list(map(simulation_step, snr_linear))

        with lzma.open(
                os.path.join(path, f'{timestamp}_simulation_BF_comparison_entropy_decoder.xz'),
                "wb") as f:
            pickle.dump(results, f)
        logger.info("saved compressed results file")

        raw_ber = np.array([p['raw_ber'] for p in results])
        ms_ber = np.array([p['ms_decoder_ber'] for p in results])
        wbf_decoder_ber = np.array([p['wbf_decoder_ber'] for p in results])
        mwbf_decoder_ber = np.array([p['mwbf_decoder_ber'] for p in results])
        mwbf_no_loop_decoder_ber = np.array([p['no_loop_decoder_ber'] for p in results])
        entropy_wbf_decoder_ber = np.array([p['entropy_wbf_decoder_ber'] for p in results])
        entropy_mwbf_decoder_ber = np.array([p['entropy_mwbf_decoder_ber'] for p in results])
        entropy_mwbf_no_loop_decoder_ber = np.array([p['entropy_no_loop_decoder_ber'] for p in results])
        fig = plt.figure()
        plt.plot(raw_ber, ms_ber, 'bo', raw_ber, raw_ber, 'g^', raw_ber, entropy_mwbf_no_loop_decoder_ber, 'r*')
        plt.xlabel("AWGN bit flip probability p")
        plt.ylabel("post decoding BER")
        fig.savefig(os.path.join(path, "ber_vs_error_p.eps"), dpi=150)

        figure = plt.figure()
        ms_buffer_success_rate = np.array([p['ms_success_rate'] for p in results])
        wbf_buffer_success_rate = np.array([p['wbf_success_rate'] for p in results])
        mwbf_buffer_success_rate = np.array([p['mwbf_success_rate'] for p in results])
        mwbf_no_loop_buffer_success_rate = np.array([p['no_loop_success_rate'] for p in results])
        entropy_wbf_buffer_success_rate = np.array([p['entropy_wbf_success_rate'] for p in results])
        entropy_mwbf_buffer_success_rate = np.array([p['entropy_mwbf_success_rate'] for p in results])
        entropy_mwbf_no_loop_buffer_success_rate = np.array([p['entropy_no_loop_success_rate'] for p in results])
        plt.plot(raw_ber, ms_buffer_success_rate, 'bo', raw_ber, entropy_mwbf_no_loop_buffer_success_rate, 'r*')
        plt.xlabel("AWGN bit flip probability p")
        plt.ylabel("Decode success rate")
        figure.savefig(os.path.join(path, "buffer_success_rate_vs_error_p.eps"), dpi=150)
        logger.info("saved figures")

        summary = {"args": args, "raw_ber": raw_ber, "ms_ber": ms_ber, "wbf_decoder_ber": wbf_decoder_ber,
                   "mwbf_decoder_ber": mwbf_decoder_ber, "no_loop_decoder_ber": mwbf_no_loop_decoder_ber,
                   "entropy_wbf_decoder_ber": entropy_wbf_decoder_ber,
                   "entropy_mwbf_decoder_ber": entropy_mwbf_decoder_ber,
                   "entropy_no_loop_decoder_ber": entropy_mwbf_no_loop_decoder_ber,
                   "ms_buffer_success_rate": ms_buffer_success_rate,
                   "wbf_buffer_success_rate": wbf_buffer_success_rate,
                   "mwbf_buffer_success_rate": mwbf_buffer_success_rate,
                   "no_loop_buffer_success_rate":  mwbf_no_loop_buffer_success_rate,
                   "entropy_wbf_buffer_success_rate": entropy_wbf_buffer_success_rate,
                   "entropy_mwbf_buffer_success_rate": entropy_mwbf_buffer_success_rate,
                   "entropy_no_loop_buffer_success_rate": entropy_mwbf_no_loop_buffer_success_rate,
                   "snr_linear":  snr_linear, "snr_db": snr_db}
        with open(os.path.join(path, f'{timestamp}_summary_BF_comparison_entropy_decoder.pickle'), 'wb') as f:
            pickle.dump(summary, f)
        savemat(os.path.join(path, f'{timestamp}_summary_BF_comparison_entropy_decoder.mat'), summary)
        logger.info("saved summary")

        for step in results:
            step['data'] = step['data'].to_dict("list")
            step['decoded_ms'] = step['decoded_ms'].to_dict("list")
            step['wbf'] = step['decoded_wbf'].to_dict("list")
            step['entropy_wbf'] = step['decoded_entropy_wbf'].to_dict("list")
            step['mwbf'] = step['decoded_mwbf'].to_dict("list")
            step['entropy_mwbf'] = step['decoded_entropy_mwbf'].to_dict("list")
            step['no_loop'] = step['decoded_no_loop'].to_dict("list")
            step['entropy_no_loop'] = step['decoded_entropy_no_loop'].to_dict("list")

        summary["results"] = results
        savemat(os.path.join(path, f'{timestamp}_simulation_BF_comparison_entropy_decoder.mat'),
                summary, do_compression=True)
        logger.info("saved results to mat file")
        shutil.move("results/log.log", os.path.join(path, "log.log"))
    except Exception as e:
        logger.exception(e)
        shutil.move("results/log.log", os.path.join(path, "log.log"))
        raise e
