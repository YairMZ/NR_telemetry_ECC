import pickle
from bitstring import Bits, BitArray
import numpy as np
from ldpc.encoder import EncoderWiFi
from ldpc.wifi_spec_codes import WiFiSpecCode
from ldpc.decoder import bsc_llr, DecoderWiFi
from decoders import ClassifyingDudeDecoder
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
parser.add_argument("--minflip", default=33*1e-3, help="minimal bit flip probability to consider", type=float)
parser.add_argument("--maxflip", default=70*1e-3, help="maximal bit flip probability to consider", type=float)
parser.add_argument("--nflips", default=20, help="number of bit flips to consider", type=int)
parser.add_argument("--ldpciterations", default=20, help="number of iterations of  LDPC decoder", type=int)
parser.add_argument("--ent_threshold", default=0.36, help="entropy threshold", type=float)
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
parser.add_argument("--context_length", default=8, help="context length", type=int)

args = parser.parse_args()

ldpc_iterations = args.ldpciterations
thr = args.ent_threshold
clipping_factor = args.clipping_factor
processes = args.processes if args.processes > 0 else None

with open('data/hc_to_ship.pickle', 'rb') as f:
    hc_tx = pickle.load(f)

hc_bin_data = [Bits(auto=tx.get("bin")) for tx in hc_tx.get(args.msg_delay)]
n = args.N if args.N > 0 else len(hc_bin_data)

# corrupt data
rng = np.random.default_rng()
bit_flip_p = np.linspace(args.minflip, args.maxflip, num=args.nflips)

if args.n_clusters == 1:
    spec = WiFiSpecCode.N1944_R23
    args.classifier_train = 0
elif args.n_clusters == 2:
    spec = WiFiSpecCode.N1296_R12
elif args.n_clusters == 3:
    spec = WiFiSpecCode.N648_R34
encoder = EncoderWiFi(spec=spec)
encoded = []
for binary_data in hc_bin_data[:n]:
    if args.n_clusters == 1:
        pad_len = encoder.k - len(binary_data)
        padded = binary_data + Bits(auto=rng.integers(low=0, high=2, size=pad_len))
        encoded.append(encoder.encode(padded))
    elif args.n_clusters == 2:
        padded = binary_data[:576] + Bits(auto=rng.integers(low=0, high=2, size=encoder.k-576))
        encoded.append(encoder.encode(padded))
        encoded.append(encoder.encode(binary_data[576:]))
    elif args.n_clusters == 3:
        padded = binary_data[:416] + Bits(auto=rng.integers(low=0, high=2, size=encoder.k-416))
        encoded.append(encoder.encode(padded))
        padded = binary_data[416:864] + Bits(auto=rng.integers(low=0, high=2, size=encoder.k - (864-416)))
        encoded.append(encoder.encode(padded))
        padded = binary_data[864:] + Bits(auto=rng.integers(low=0, high=2, size=encoder.k - (1224-864)))
        encoded.append(encoder.encode(padded))

for _ in range(args.multiply_data):  # generate more buffers for statistical reproducibility
    encoded.extend(encoded)

if args.model_length == 'info':
    model_length = encoder.k
else:
    model_length = encoder.n

n = len(encoded)  # redefine n

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
    global n
    global spec
    channel = bsc_llr(p=p)
    ldpc_decoder = DecoderWiFi(spec=spec, max_iter=ldpc_iterations, decoder_type=args.dec_type)
    dude_decoder = ClassifyingDudeDecoder(DecoderWiFi(spec=spec, max_iter=ldpc_iterations,
                                                                decoder_type=args.dec_type),
                                             model_length=model_length, clipping_factor=clipping_factor,
                                             classifier_training=args.classifier_train, n_clusters=args.n_clusters,
                                             context_length=args.context_length, conf_center=args.conf_center,
                                             conf_slope=args.conf_slope, bit_flip=p, cluster=args.cluster,
                                             entropy_threshold=thr)
    no_errors = int(encoder.n * p)
    rx = []
    decoded_ldpc = []
    decoded_dude = []
    errors = np.vstack(
        tuple(rng.choice(encoder.n, size=no_errors, replace=False)
              for _ in range(n))
    )
    step_results: dict[str, Any] = {'data': hc_bin_data[:n]}
    for tx_idx in range(n):
        # pad data - add 72 bits
        corrupted = BitArray(encoded[tx_idx])
        for idx in errors[tx_idx]:
            corrupted[idx] = not corrupted[idx]
        rx.append(corrupted)
        channel_llr = channel(np.array(corrupted, dtype=np.int_))
        d = ldpc_decoder.decode(channel_llr)
        decoded_ldpc.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        d = dude_decoder.decode_buffer(channel_llr)
        decoded_dude.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        if decoded_dude[-1][2] is True and decoded_ldpc[-1][2] is False:
            logger.info(f"p={p}, tx={tx_idx}, DUDE RECOVERY!!!!!!!!")
        elif decoded_dude[-1][2] is False and decoded_ldpc[-1][2] is True:
            logger.info(f"p={p}, tx={tx_idx}, DUDE FAILURE!!!!!!!!")
        else:
            logger.info(f"p={p}, tx={tx_idx}")
    logger.info(f"p={p}, ldpc={sum(int(res[-1] == 0) for res in decoded_ldpc)}/{n}, dude="
                f"{sum(int(res[-1] == 0) for res in decoded_dude)}/{n}")
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
    decoded_dude_df = pd.DataFrame(decoded_dude,
                                      columns=["estimate", "llr", "decode_success", "iterations", "cluster_label",
                                               "denoised_bits", "hamming"])
    step_results["decoded_dude"] = decoded_dude_df
    decoded_ldpc_df = pd.DataFrame(decoded_ldpc,
                                   columns=["estimate", "llr", "decode_success", "iterations", "syndrome",
                                            "vnode_validity", "hamming"])
    step_results['decoded_ldpc'] = decoded_ldpc_df
    # performance
    step_results["ldpc_buffer_success_rate"] = sum(int(res[-1] == 0) for res in decoded_ldpc) / float(n)
    step_results["ldpc_decoder_ber"] = sum(res[-1] for res in decoded_ldpc) / float(n * len(encoded[0]))
    step_results["dude_buffer_success_rate"] = sum(int(res[-1] == 0) for res in decoded_dude) / float(n)
    step_results["dude_decoder_ber"] = sum(res[-1] for res in decoded_dude) / float(n * len(encoded[0]))

    timestamp = f'{str(datetime.date.today())}_{str(datetime.datetime.now().hour)}_{str(datetime.datetime.now().minute)}_' \
                f'{str(datetime.datetime.now().second)}'
    with open(f'{timestamp}_{p}_simulation_classifying_dude.pickle', 'wb') as f:
        pickle.dump(step_results, f)
    return step_results


if __name__ == '__main__':
    timestamp = f'{str(datetime.date.today())}_{str(datetime.datetime.now().hour)}_{str(datetime.datetime.now().minute)}_' \
                f'{str(datetime.datetime.now().second)}'

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
    logger.info(f"context length: {args.context_length}")

    cmd = f'python {__file__} --minflip {args.minflip} --maxflip {args.maxflip} --nflips {args.nflips} --ldpciterations ' \
          f'{ldpc_iterations} --ent_threshold {thr} --clipping_factor {clipping_factor} --conf_center {args.conf_center} ' \
          f'--conf_slope {args.conf_slope} --multiply_data {args.multiply_data} --dec_type {args.dec_type}  --classifier_train ' \
          f'{args.classifier_train} --n_clusters {args.n_clusters} --msg_delay {args.msg_delay} --cluster {args.cluster} ' \
          f'--model_length {args.model_length} --context_length {args.context_length}'

    if args.N > 0:
        cmd += f' --N {n}'
    if processes is not None:
        cmd += f' --processes {processes}'

    timestamp = f'{str(datetime.date.today())}_{str(datetime.datetime.now().hour)}_{str(datetime.datetime.now().minute)}_' \
                f'{str(datetime.datetime.now().second)}'

    os.mkdir(path)
    with open(os.path.join(path, "cmd.txt"), 'w') as f:
        f.write(cmd)
    logger.info(cmd)

    try:
        with Pool(processes=processes) as pool:
            results: list[dict[str, Any]] = pool.map(simulation_step, bit_flip_p)
        # results: list[dict[str, Any]] = list(map(simulation_step, bit_flip_p))


        with lzma.open(
                os.path.join(path, f'{timestamp}_simulation_classifying_DUDE_{args.dec_type}_decoder.xz'),
                "wb") as f:
            pickle.dump(results, f)
        logger.info("saved compressed results file")

        raw_ber = np.array([p['raw_ber'] for p in results])
        ldpc_ber = np.array([p['ldpc_decoder_ber'] for p in results])
        dude_ber = np.array([p['dude_decoder_ber'] for p in results])
        fig = plt.figure()
        plt.plot(raw_ber, ldpc_ber, 'bo', raw_ber, raw_ber, 'g^', raw_ber, dude_ber, 'r*')
        plt.xlabel("BSC bit flip probability p")
        plt.ylabel("post decoding BER")
        fig.savefig(os.path.join(path, "ber_vs_error_p.eps"), dpi=150)

        figure = plt.figure()
        ldpc_buffer_success_rate = np.array([p['ldpc_buffer_success_rate'] for p in results])
        dude_buffer_success_rate = np.array([p['dude_buffer_success_rate'] for p in results])
        plt.plot(raw_ber, ldpc_buffer_success_rate, 'bo', raw_ber, dude_buffer_success_rate, 'r*')
        plt.xlabel("BSC bit flip probability p")
        plt.ylabel("Decode success rate")
        figure.savefig(os.path.join(path, "buffer_success_rate_vs_error_p.eps"), dpi=150)
        logger.info("saved figures")

        summary = {"args": args, "raw_ber": raw_ber, "ldpc_ber": ldpc_ber, "dude_ber": dude_ber,
                   "ldpc_buffer_success_rate": ldpc_buffer_success_rate,
                   "dude_buffer_success_rate": dude_buffer_success_rate}
        with open(os.path.join(path, f'{timestamp}_summary_classifying_dude_{args.dec_type}_decoder.pickle'), 'wb') as f:
            pickle.dump(summary, f)

        savemat(os.path.join(path, f'{timestamp}_summary_classifying_dude_{args.dec_type}_decoder.mat'),
                summary)
        logger.info("saved summary")

        for step in results:
            step['data'] = step['data'].to_dict("list")
            step['decoded_dude'] = step['decoded_dude'].to_dict("list")
            step['decoded_ldpc'] = step['decoded_ldpc'].to_dict("list")

        summary.update({"results": results})
        savemat(os.path.join(path, f'{timestamp}_simulation_classifying_dude_{args.dec_type}_decoder.mat'),
                summary, do_compression=True)
        logger.info("saved results to mat file")
        shutil.move("results/log.log", os.path.join(path, "log.log"))
    except Exception as e:
        logger.exception(e)
        shutil.move("results/log.log", os.path.join(path, "log.log"))
        raise e
