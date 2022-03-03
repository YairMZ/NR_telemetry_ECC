import pickle
from bitstring import Bits, BitArray
import numpy as np
from ldpc.encoder import EncoderWiFi
from ldpc.wifi_spec_codes import WiFiSpecCode
from ldpc.decoder import bsc_llr, DecoderWiFi
from decoders import CombinedDecoder
from inference import BufferSegmentation
from protocol_meta import dialect_meta as meta
import random
from utils.bit_operations import hamming_distance
from typing import Any
import matplotlib.pyplot as plt
import argparse
import datetime
import os
from multiprocessing import Pool


parser = argparse.ArgumentParser(description='Run decoding on simulated data using multiprocessing.')
parser.add_argument("--N", default=0, help="max number of transmissions to consider", type=int)
parser.add_argument("--minflip", default=36*1e-3, help="minimal bit flip probability to consider", type=float)
parser.add_argument("--maxflip", default=55*1e-3, help="maximal bit flip probability to consider", type=float)
parser.add_argument("--nflips", default=5, help="number of bit flips to consider", type=int)
parser.add_argument("--ldpciterations", default=10, help="number of iterations of  LDPC decoder", type=int)
parser.add_argument("--ent_threshold", default=0.36, help="entropy threshold to be used in entropy decoder", type=float)
parser.add_argument("--window_len", default=0, help="number of previous samples to use, if 0 all are used", type=int)
parser.add_argument("--clipping_factor", default=2, help="dictates maximal and minimal llr", type=int)
parser.add_argument("--min_data", default=10, help="minimal amount of samples before inference is used", type=int)
parser.add_argument("--segiterations", default=2, help="number of exchanges between LDPC and CB decoder", type=int)
parser.add_argument("--goodp", default=1e-7, help="number of exchanges between LDPC and CB decoder", type=float)
parser.add_argument("--badp", default=0, help="number of exchanges between LDPC and CB decoder", type=float)


args = parser.parse_args()

ldpc_iterations = args.ldpciterations
seg_iter = args.segiterations
thr = args.ent_threshold
clipping_factor = args.clipping_factor

encoder = EncoderWiFi(WiFiSpecCode.N1944_R23)
bs = BufferSegmentation(meta.protocol_parser)

with open('data/hc_to_ship.pickle', 'rb') as f:
    hc_tx = pickle.load(f)

five_sec_bin = [Bits(auto=tx.get("bin")) for tx in hc_tx.get("50000")]
n = args.N if args.N > 0 else len(five_sec_bin)
window_len = args.window_len if args.window_len > 0 else None

# corrupt data
rng = np.random.default_rng()
bit_flip_p = np.linspace(args.minflip, args.maxflip, num=args.nflips)

encoded = []
for binary_data in five_sec_bin[:n]:
    pad_len = encoder.k - len(binary_data)
    padded = binary_data + Bits(uint=random.getrandbits(pad_len), length=pad_len)
    encoded.append(encoder.encode(padded))

model_length = len(five_sec_bin[0])
n = len(encoded)

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
print("min_data:", args.min_data)
print("number of segmentation iterations: ", seg_iter)
print("good probability: ", args.goodp)
print("bad probability: ", args.badp)

cmd = f'python {__file__} --minflip {args.minflip} --maxflip {args.maxflip} --nflips {args.nflips}  --ldpciterations ' +\
      f'{ldpc_iterations} --ent_threshold {thr} --clipping_factor {clipping_factor} --min_data {args.min_data}' +\
      f' --segiterations {seg_iter} --goodp {args.goodp} --badp {args.badp}'

if window_len is not None:
    cmd += f' --window_len {window_len}'
if args.N > 0:
    cmd += f' --N {n}'


def simulation_step(p: float) -> dict[str, Any]:
    global ldpc_iterations
    global model_length
    global thr
    global clipping_factor
    global args
    global window_len
    global seg_iter
    channel = bsc_llr(p=p)
    bad_p = args.badp if args.badp > 0 else p  # this will make channel llr difference of close to 2 between default and bad
    good_p = args.goodp if args.goodp > 0 else p / 7  # this will make channel llr diff of close to 2 between default and good
    ldpc_decoder = DecoderWiFi(spec=WiFiSpecCode.N1944_R23, max_iter=ldpc_iterations)
    combined_decoder = CombinedDecoder(DecoderWiFi(spec=WiFiSpecCode.N1944_R23, max_iter=ldpc_iterations),
                                       model_length=model_length, entropy_threshold=thr, clipping_factor=clipping_factor,
                                       min_data=args.min_data, segmentation_iterations=seg_iter, bad_p=bad_p, good_p=good_p,
                                       window_length=window_len)

    no_errors = int(encoder.n * p)
    rx = []
    decoded_ldpc = []
    decoded_combined = []
    step_results: dict[str, Any] = {'data': five_sec_bin[:n]}
    for tx_idx in range(n):
        # pad data - add 72 bits
        corrupted = BitArray(encoded[tx_idx])
        error_idx = rng.choice(len(corrupted), size=no_errors, replace=False)
        for idx in error_idx:
            corrupted[idx] = not corrupted[idx]
        rx.append(corrupted)
        channel_llr = channel(np.array(corrupted, dtype=np.int_))
        d = ldpc_decoder.decode(channel_llr)
        b = ldpc_decoder.info_bits(d[0]).tobytes()
        parts, v, s = bs.segment_buffer(b)
        decoded_ldpc.append((*d, len(s), hamming_distance(Bits(auto=d[0]), encoded[tx_idx])))
        d = combined_decoder.decode_buffer(channel_llr)
        decoded_combined.append((*d, hamming_distance(Bits(auto=d[0]), encoded[tx_idx])))
        print("p= ", p, " tx id: ", tx_idx)
    print("successful pure decoding for bit flip p=", p, ", is: ", sum(int(res[5] == 0) for res in decoded_ldpc), "/", n)
    print("successful combined decoding for bit flip p=", p, ", is: ", sum(int(res[5] == 0) for res in decoded_combined), "/",
          n)
    step_results['encoded'] = encoded
    step_results['rx'] = rx
    step_results['decoded_ldpc'] = decoded_ldpc
    step_results["ldpc_buffer_success_rate"] = sum(int(res[5] == 0) for res in decoded_ldpc) / float(n)

    step_results["raw_ber"] = no_errors / encoder.n
    step_results["buffer_len"] = len(encoded[0])
    step_results["ldpc_decoder_ber"] = sum(
        hamming_distance(encoded[idx], Bits(auto=decoded_ldpc[idx][0]))
        for idx in range(n)
    ) / float(n * len(encoded[0]))

    step_results["decoded_combined"] = decoded_combined
    step_results["combined_buffer_success_rate"] = sum(int(res[5] == 0) for res in decoded_combined) / float(n)
    step_results["combined_decoder_ber"] = sum(
        hamming_distance(encoded[idx], Bits(auto=decoded_combined[idx][0]))
        for idx in range(n)
    ) / float(n * len(encoded[0]))

    step_results["n"] = n
    step_results["max_ldpc_iterations"] = ldpc_iterations
    return step_results


if __name__ == '__main__':
    with Pool() as pool:
        results: list[dict[str, Any]] = pool.map(simulation_step, bit_flip_p)

    timestamp = f'{str(datetime.date.today())}_{str(datetime.datetime.now().hour)}_{str(datetime.datetime.now().minute)}_' \
                f'{str(datetime.datetime.now().second)}'

    path = os.path.join("results/", timestamp)
    os.mkdir(path)
    with open(os.path.join(path, "cmd.txt"), 'wt') as f:
        f.write(cmd)

    with open(os.path.join(path, timestamp + '_simulation_combined_vs_pure_LDPC.pickle'), 'wb') as f:
        pickle.dump(results, f)

    raw_ber = np.array([p['raw_ber'] for p in results])
    ldpc_ber = np.array([p['ldpc_decoder_ber'] for p in results])
    combined_ber = np.array([p['combined_decoder_ber'] for p in results])
    fig = plt.figure()
    plt.plot(raw_ber, ldpc_ber, 'bo', raw_ber, raw_ber, 'g^', raw_ber, combined_ber, 'r*')
    plt.xlabel("BSC bit flip probability p")
    plt.ylabel("post decoding BER")
    fig.savefig(os.path.join(path, "ber_vs_error_p.eps"), dpi=150)

    figure = plt.figure()
    ldpc_buffer_success_rate = np.array([p['ldpc_buffer_success_rate'] for p in results])
    combined_buffer_success_rate = np.array([p['combined_buffer_success_rate'] for p in results])
    plt.plot(raw_ber, ldpc_buffer_success_rate, 'bo', raw_ber, combined_buffer_success_rate, 'r*')
    plt.xlabel("BSC bit flip probability p")
    plt.ylabel("Decode success rate")
    figure.savefig(os.path.join(path, "buffer_success_rate_vs_error_p.eps"), dpi=150)

    summary = {"args": args, "raw_ber": raw_ber, "ldpc_ber": ldpc_ber, "combined_ber": combined_ber,
               "ldpc_buffer_success_rate": ldpc_buffer_success_rate,
               "combined_buffer_success_rate": combined_buffer_success_rate}
    with open(os.path.join(path, timestamp + '_summary_combined_vs_pure_LDPC.pickle'), 'wb') as f:
        pickle.dump(summary, f)
