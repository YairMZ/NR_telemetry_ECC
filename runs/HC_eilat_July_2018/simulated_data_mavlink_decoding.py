import pickle
from bitstring import Bits, BitArray
import numpy as np
from ldpc.encoder import EncoderWiFi
from ldpc.wifi_spec_codes import WiFiSpecCode
from ldpc.decoder import bsc_llr, DecoderWiFi
from numpy.typing import NDArray
from decoders import MavlinkRectifyingDecoder
from inference import BufferSegmentation, BufferModel
from protocol_meta import dialect_meta as meta
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
parser.add_argument("--minflip", default=33 * 1e-3, help="minimal bit flip probability to consider", type=float)
parser.add_argument("--maxflip", default=70 * 1e-3, help="maximal bit flip probability to consider", type=float)
parser.add_argument("--nflips", default=20, help="number of bit flips to consider", type=int)
parser.add_argument("--ldpciterations", default=20, help="number of iterations of  LDPC decoder", type=int)
parser.add_argument("--segiterations", default=1, help="number of exchanges between LDPC and CB decoder", type=int)
parser.add_argument("--multiply_data", default=0, help="multiplies amount of buffers by 2 to power of arg", type=int)
parser.add_argument("--processes", default=0, help="number of processes to spawn", type=int)
parser.add_argument("--dec_type", default="BP", help="scheme for determining confidence", type=str)
parser.add_argument("--classifier_train", default=100, help="number of buffers used for classifier training", type=int)
parser.add_argument("--n_clusters", default=1, help="number of clusters", type=int)
parser.add_argument("--msg_delay", default="50000", help="sampling delay", type=str)
parser.add_argument("--cluster", default=1, help="enable or disable clustering", type=int)
parser.add_argument("--valid_factor", default=2.0, help="valid factor", type=float)
parser.add_argument("--invalid_factor", default=0.7, help="invalid factor", type=float)
parser.add_argument("--valid_threshold", default=0.03,
                    help="valid_threshold  as outlier probability for classifying a field as valid", type=float)
parser.add_argument("--invalid_threshold", default=0.08,
                    help="invalid_threshold  as outlier probability for classifying a field as invalid", type=float)
parser.add_argument("--window_len", default=50,
                    help="number of previous samples to use for training the model, if 0 all are used", type=int)
parser.add_argument("--learn", default=0, help="0 predefined model is used. 1 model is learned online", type=int)
parser.add_argument("--conf_center", default=40, help="center of model sigmoid", type=int)
parser.add_argument("--conf_slope", default=0.35, help="slope of model sigmoid", type=float)

args = parser.parse_args()

ldpc_iterations = args.ldpciterations
processes = args.processes if args.processes > 0 else None

with open('data/hc_to_ship.pickle', 'rb') as f:
    hc_tx = pickle.load(f)

hc_bin_data = [Bits(auto=tx.get("bin")) for tx in hc_tx.get(args.msg_delay)]
n = args.N if args.N > 0 else len(hc_bin_data)
window_len = args.window_len if args.window_len > 0 else None

rng = np.random.default_rng()
bit_flip_p = np.linspace(args.minflip, args.maxflip, num=args.nflips)

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
        padded = binary_data[:576] + Bits(auto=rng.integers(low=0, high=2, size=encoder.k - 576))
        encoded.extend((encoder.encode(padded), encoder.encode(binary_data[576:])))
    elif args.n_clusters == 3:
        padded = binary_data[:416] + Bits(auto=rng.integers(low=0, high=2, size=encoder.k - 416))
        encoded.append(encoder.encode(padded))
        padded = binary_data[416:864] + Bits(auto=rng.integers(low=0, high=2, size=encoder.k - (864 - 416)))
        encoded.append(encoder.encode(padded))
        padded = binary_data[864:] + Bits(auto=rng.integers(low=0, high=2, size=encoder.k - (1224 - 864)))
        encoded.append(encoder.encode(padded))

for _ in range(args.multiply_data):  # generate more buffers for statistical reproducibility
    encoded.extend(encoded)
n = len(encoded)
#{0: 33, 52: 234, 72: 30, 108: 212, 135: 218}
if not bool(args.cluster):
    data_model = None
    bs = BufferSegmentation(meta.protocol_parser)
    if args.n_clusters == 1:
        _, _, buffer_structure = bs.segment_buffer(binary_data.tobytes())
        buffer_structures = [buffer_structure]
    elif args.n_clusters == 2:
        _, _, buffer_structure = bs.segment_buffer(binary_data[:576].tobytes())
        buffer_structures = [buffer_structure]
        _, _, buffer_structure = bs.segment_buffer(binary_data[576:].tobytes())
        buffer_structures.append(buffer_structure)
    elif args.n_clusters == 3:
        _, _, buffer_structure = bs.segment_buffer(binary_data[:416].tobytes())
        buffer_structures = [buffer_structure]
        _, _, buffer_structure = bs.segment_buffer(binary_data[416:864].tobytes())
        buffer_structures.append(buffer_structure)
        _, _, buffer_structure = bs.segment_buffer(binary_data[864:].tobytes())
        buffer_structures.append(buffer_structure)
    else:
        raise ValueError("n_clusters must be 1, 2 or 3")
if not bool(args.learn):
    data_model = BufferModel.load('data/model_2018_all.json')

model_length = encoder.k
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
    global data_model
    global encoder
    global buffer_structures
    global logger

    channel = bsc_llr(p=p)
    ldpc_decoder = DecoderWiFi(spec=spec, max_iter=args.segiterations*ldpc_iterations, decoder_type=args.dec_type)
    rectify_decoder = MavlinkRectifyingDecoder(DecoderWiFi(spec=spec, max_iter=ldpc_iterations,
                                                           decoder_type=args.dec_type),
                                               model_length, args.valid_threshold, args.invalid_threshold,
                                               args.n_clusters, args.valid_factor, args.invalid_factor,
                                               args.classifier_train, bool(args.cluster), window_len,
                                               data_model, args.conf_center, args.conf_slope,
                                               segmentation_iterations=args.segiterations)
    if not bool(args.cluster):
        rectify_decoder.set_buffer_structures(buffer_structures)
    no_errors = int(encoder.n * p)
    rx = []
    decoded_ldpc = []
    decoded_rect = []
    errors = np.vstack(
        tuple(rng.choice(encoder.n, size=no_errors, replace=False)
              for _ in range(n))
    )
    step_results: dict[str, Any] = {'data': hc_bin_data[:n]}
    good_fields_performance: NDArray[np.int_] = np.zeros((n, 6), dtype=np.int_)
    bad_fields_performance: NDArray[np.int_] = np.zeros((n, 6), dtype=np.int_)
    for tx_idx in range(n):
        corrupted = BitArray(encoded[tx_idx])
        corrupted.invert(errors[tx_idx])
        rx.append(corrupted)
        channel_llr = channel(np.array(corrupted, dtype=np.int_))
        d = ldpc_decoder.decode(channel_llr)
        decoded_ldpc.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        d = rectify_decoder.decode_buffer(channel_llr, errors[tx_idx])
        decoded_rect.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        logger.info(f"p= {p}, tx id: {tx_idx}")
    pure_success = sum(int(r[-1] == 0) for r in decoded_ldpc)
    rect_success = sum(int(r[-1] == 0) for r in decoded_rect)
    logger.info(f"successful pure decoding for bit flip p= {p}, is: {pure_success}/{n}")
    logger.info(f"successful rect decoding for bit flip p= {p}, is: {rect_success}/{n}")
    if n-pure_success > 0:
        logger.info(f"recover rate for bit flip p= {p}, is: {(rect_success-pure_success)/(n-pure_success)}")
    else:
        logger.info(f"recover rate for bit flip p= {p}, is: 0")

    # log data
    info_errors = np.sum(errors < encoder.k, axis=1)
    parity_errors = np.sum(errors >= encoder.k, axis=1)
    zipped = [[np.array(en, dtype=np.int_), np.array(r, dtype=np.int_), er, inf, pa] for en, r, er, inf, pa in
              zip(encoded, rx, errors, info_errors, parity_errors)]
    step_results["data"] = pd.DataFrame(zipped, columns=["encoded", "corrupted", "error_idx", "info_errors",
                                                         "parity_errors"])

    # params
    step_results["raw_ber"] = no_errors / encoder.n
    step_results["buffer_len"] = len(encoded[0])
    step_results["number_of_buffers"] = n
    step_results["max_ldpc_iterations"] = ldpc_iterations

    # decoding
    decoded_rect_df = pd.DataFrame(decoded_rect,
                                   columns=["estimate", "llr", "decode_success", "iterations", "syndrome",
                                            "vnode_validity", "cluster_label", "segmented_bits", "good_field_idx",
                                            "bad_field_idx", "classifier_performance", "forcing_performance",
                                            "hamming"])
    step_results["decoded_rect"] = decoded_rect_df
    decoded_ldpc_df = pd.DataFrame(decoded_ldpc,
                                   columns=["estimate", "llr", "decode_success", "iterations", "syndrome",
                                            "vnode_validity", "hamming"])
    step_results['decoded_ldpc'] = decoded_ldpc_df
    # performance
    step_results["ldpc_buffer_success_rate"] = sum(int(res[-1] == 0) for res in decoded_ldpc) / float(n)
    step_results["ldpc_decoder_ber"] = sum(res[-1] for res in decoded_ldpc) / float(n * len(encoded[0]))
    step_results["rect_buffer_success_rate"] = sum(int(res[-1] == 0) for res in decoded_rect) / float(n)
    step_results["rect_decoder_ber"] = sum(res[-1] for res in decoded_rect) / float(n * len(encoded[0]))

    timestamp = f'{str(datetime.date.today())}_{str(datetime.datetime.now().hour)}_{str(datetime.datetime.now().minute)}_' \
                f'{str(datetime.datetime.now().second)}'
    with open(f'results/{timestamp}_{p}_simulation_classifying_rect.pickle', 'wb') as f:
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
    logger.info(f"processes: {args.processes}")
    logger.info(f"multiply data: {args.multiply_data}")
    logger.info(f"msg_delay: {args.msg_delay} ")
    logger.info(f"decoder type: {args.dec_type}")
    logger.info(f"classifier_train: {args.classifier_train}")
    logger.info(f"n_clusters: {args.n_clusters}")
    logger.info(f"cluster: {args.cluster}")
    logger.info(f"valid_threshold used in decoder: {args.valid_threshold}")
    logger.info(f"invalid_threshold used in decoder: {args.invalid_threshold}")
    logger.info(f"valid_factor: {args.valid_factor}")
    logger.info(f"invalid_factor: {args.invalid_factor}")
    logger.info(f"decoder window length: {window_len}")
    logger.info(f"confidence center: {args.conf_center}")
    logger.info(f"confidence slope: {args.conf_slope}")
    logger.info(f"learn: {args.learn}")
    logger.info(f"seg_iter: {args.segiterations}")

    cmd = f'python {__file__} --minflip {args.minflip} --maxflip {args.maxflip} --nflips {args.nflips}  ' \
          f'--ldpciterations {ldpc_iterations} --segiterations {args.segiterations} --multiply_data {args.multiply_data} ' \
          f'--msg_delay {args.msg_delay} --dec_type {args.dec_type} --classifier_train {args.classifier_train} ' \
          f'--n_clusters {args.n_clusters} --cluster {args.cluster} --valid_threshold {args.valid_threshold} ' \
          f'--invalid_threshold {args.invalid_threshold} ' \
          f'--valid_factor {args.valid_factor} --invalid_factor {args.invalid_factor} --conf_center {args.conf_center} ' \
          f'--conf_slope {args.conf_slope} --learn {args.learn}'
    if args.N > 0:
        cmd += f' --N {n}'
    if window_len is not None:
        cmd += f' --window_len {window_len}'
    else:
        cmd += ' --window_len 0'
    if processes is not None:
        cmd += f' --processes {processes}'

    with open(os.path.join(path, "cmd.txt"), 'w') as f:
        f.write(cmd)
    logger.info(cmd)

    with Pool(processes=processes) as pool:
        results: list[dict[str, Any]] = pool.map(simulation_step, bit_flip_p)
    # results: list[dict[str, Any]] = list(map(simulation_step, bit_flip_p))

    try:
        with lzma.open(
                os.path.join(path, f'{timestamp}_simulation_rectifying_{args.dec_type}_decoder_WiFi.xz'),
                "wb") as f:
            pickle.dump(results, f)
        logger.info("saved compressed results file")

        raw_ber = np.array([p['raw_ber'] for p in results])
        ldpc_ber = np.array([p['ldpc_decoder_ber'] for p in results])
        rect_ber = np.array([p['rect_decoder_ber'] for p in results])
        fig = plt.figure()
        plt.plot(raw_ber, ldpc_ber, 'bo', raw_ber, raw_ber, 'g^', raw_ber, rect_ber, 'r*')
        plt.xlabel("BSC bit flip probability p")
        plt.ylabel("post decoding BER")
        fig.savefig(os.path.join(path, "ber_vs_error_p.eps"), dpi=150)

        figure = plt.figure()
        ldpc_buffer_success_rate = np.array([p['ldpc_buffer_success_rate'] for p in results])
        rect_buffer_success_rate = np.array([p['rect_buffer_success_rate'] for p in results])
        plt.plot(raw_ber, ldpc_buffer_success_rate, 'bo', raw_ber, rect_buffer_success_rate, 'r*')
        plt.xlabel("BSC bit flip probability p")
        plt.ylabel("Decode success rate")
        figure.savefig(os.path.join(path, "buffer_success_rate_vs_error_p.eps"), dpi=150)

        logger.info("saved figures")
        summary = {"args": args, "raw_ber": raw_ber, "ldpc_ber": ldpc_ber, "rect_ber": rect_ber,
                   "ldpc_buffer_success_rate": ldpc_buffer_success_rate,
                   "rect_buffer_success_rate": rect_buffer_success_rate}
        with open(os.path.join(path, f'{timestamp}_summary_rectifying_{args.dec_type}_decoder_WiFi.pickle'), 'wb') as f:
            pickle.dump(summary, f)
        savemat(os.path.join(path, f'{timestamp}_summary_rectifying_{args.dec_type}_decoder_WiFi.mat'), summary)
        logger.info("saved summary")

        for step in results:
            step['data'] = step['data'].to_dict("list")
            step['decoded_rect'] = step['decoded_rect'].to_dict("list")
            step['decoded_ldpc'] = step['decoded_ldpc'].to_dict("list")

        summary["results"] = results
        savemat(os.path.join(path, f'{timestamp}_simulation_rectifying_{args.dec_type}_decoder_WiFi.mat'),
                summary, do_compression=True)
        logger.info("saved results to mat file")
        shutil.move("results/log.log", os.path.join(path, "log.log"))
    except Exception as e:
        logger.exception(e)
        shutil.move("results/log.log", os.path.join(path, "log.log"))
        raise e
