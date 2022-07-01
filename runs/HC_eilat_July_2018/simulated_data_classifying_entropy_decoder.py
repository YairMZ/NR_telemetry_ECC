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
parser.add_argument("--classifier_train", default=100, help="number of buffers to use for classifier training", type=int)
parser.add_argument("--n_clusters", default=1, help="number of clusters", type=int)
parser.add_argument("--msg_delay", default="50000", help="sampling delay", type=str)


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

model_length = encoder.k
n = len(encoded)  # redfine n

# encoded structure: {starting byte index in buffer: msg_id}
# {0: 33, 52: 234, 72: 30, 108: 212, 135: 218}
# last 9 bytes (72 bits) are padding and not messages. Thus last message ends at byte index 152
# bit indices:
# {0: 33, 416: 234, 576: 30, 864: 212, 1080: 218}


print(__file__)
print("number of buffers to process: ", n)
print("smallest bit flip probability: ", args.minflip)
print("largest bit flip probability: ", args.maxflip)
print("number of bit flips: ", args.nflips)
print("number of ldpc decoder iterations: ", ldpc_iterations)
print("entropy threshold used in entropy decoder:", thr)
print("entropy decoder window length:", window_len)
print("clipping factor:", clipping_factor)
print("model center:", args.conf_center)
print("model slope:", args.conf_slope)
print("processes:", args.processes)
print("multiply data:", args.multiply_data)
print("decoder type: ", args.dec_type)
print("classifier_train: ", args.classifier_train)
print("n_clusters: ", args.n_clusters)
print("msg_delay", args.msg_delay)


cmd = f'python {__file__} --minflip {args.minflip} --maxflip {args.maxflip} --nflips {args.nflips} --ldpciterations ' \
      f'{ldpc_iterations} --ent_threshold {thr} --clipping_factor {clipping_factor} --conf_center {args.conf_center} ' \
      f'--conf_slope {args.conf_slope} --multiply_data {args.multiply_data} --dec_type {args.dec_type}  --classifier_train ' \
      f'{args.classifier_train} --n_clusters {args.n_clusters} --msg_delay {args.msg_delay}'

if window_len is not None:
    cmd += f' --window_len {window_len}'
else:
    cmd += ' --window_len 0'
if args.N > 0:
    cmd += f' --N {n}'
if processes is not None:
    cmd += f' --processes {processes}'


def simulation_step(p: float) -> dict[str, Any]:
    global ldpc_iterations
    global model_length
    global thr
    global clipping_factor
    global args
    global window_len
    global n
    global spec
    channel = bsc_llr(p=p)
    ldpc_decoder = DecoderWiFi(spec=spec, max_iter=ldpc_iterations, decoder_type=args.dec_type)
    entropy_decoder = ClassifyingEntropyDecoder(DecoderWiFi(spec=spec, max_iter=ldpc_iterations,
                                                                decoder_type=args.dec_type),
                                                model_length=model_length, entropy_threshold=thr,
                                                clipping_factor=clipping_factor,classifier_training=args.classifier_train,
                                                n_clusters=args.n_clusters, window_length=window_len,
                                                conf_center=args.conf_center*args.n_clusters,
                                                conf_slope=args.conf_slope, bit_flip=p)
    no_errors = int(encoder.n * p)
    rx = []
    decoded_ldpc = []
    decoded_entropy = []
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
        d = entropy_decoder.decode_buffer(channel_llr)
        decoded_entropy.append((*d, hamming_distance(d[0], encoded[tx_idx])))
        print("p= ", p, " tx id: ", tx_idx)
    print("successful pure decoding for bit flip p=", p, ", is: ", sum(int(res[-1] == 0) for res in decoded_ldpc), "/", n)
    print("successful entropy decoding for bit flip p=", p, ", is: ", sum(int(res[-1] == 0) for res in decoded_entropy), "/",
          n)
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
    with Pool(processes=processes) as pool:
        results: list[dict[str, Any]] = pool.map(simulation_step, bit_flip_p)
    # results: list[dict[str, Any]] = list(map(simulation_step, bit_flip_p))

    timestamp = f'{str(datetime.date.today())}_{str(datetime.datetime.now().hour)}_{str(datetime.datetime.now().minute)}_' \
                f'{str(datetime.datetime.now().second)}'

    path = os.path.join("results/", timestamp)
    os.mkdir(path)
    with open(os.path.join(path, "cmd.txt"), 'w') as f:
        f.write(cmd)

    # with open(os.path.join(path, f'{timestamp}_simulation_entropy_vs_pure_LDPC_weighted_model_{args.dec_type}_decoder.pickle'), 'wb') as f:
    #     pickle.dump(results, f)
    with lzma.open(
            os.path.join(path, f'{timestamp}_simulation_classifying_entropy_{args.dec_type}_decoder.xz'),
            "wb") as f:
        pickle.dump(results, f)

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

    summary = {"args": args, "raw_ber": raw_ber, "ldpc_ber": ldpc_ber, "entropy_ber": entropy_ber,
               "ldpc_buffer_success_rate": ldpc_buffer_success_rate,
               "entropy_buffer_success_rate": entropy_buffer_success_rate}
    with open(os.path.join(path, f'{timestamp}_summary_classifying_entropy__{args.dec_type}_decoder.pickle'), 'wb') as f:
        pickle.dump(summary, f)

    savemat(os.path.join(path, f'{timestamp}_summary_classifying_entropy__{args.dec_type}_decoder.mat'),
            summary)

    for step in results:
        step['data'] = step['data'].to_dict("list")
        step['decoded_entropy'] = step['decoded_entropy'].to_dict("list")
        step['decoded_ldpc'] = step['decoded_ldpc'].to_dict("list")

    summary.update({"results": results})
    savemat(os.path.join(path, f'{timestamp}_simulation_classifying_entropy_{args.dec_type}_decoder.mat'),
            summary, do_compression=True)
