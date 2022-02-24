import pickle
from bitstring import Bits, BitArray
import numpy as np
from ldpc.encoder import EncoderWiFi
from ldpc.wifi_spec_codes import WiFiSpecCode
from ldpc.decoder import bsc_llr, DecoderWiFi
from decoders import EntropyBitwiseDecoder, EntropyBitwiseFlippingDecoder
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


args = parser.parse_args()

ldpc_iterations = args.ldpciterations
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

cmd = f'python {__file__} --minflip {str(args.minflip)} --maxflip {str(args.maxflip)} --nflips {str(args.nflips)} ' +\
      " --ldpciterations " + str(ldpc_iterations) + " --ent_threshold " + str(thr) + " --clipping_factor " +\
      str(clipping_factor) + " --min_data " + str(args.min_data)

if window_len is not None:
    cmd += f' --window_len {str(window_len)}'
if args.N > 0:
    cmd += f' --N {n}'


def simulation_step(p: float) -> dict[str, Any]:
    global ldpc_iterations
    global model_length
    global thr
    global clipping_factor
    global args
    global window_len
    channel = bsc_llr(p=p)
    ldpc_decoder = DecoderWiFi(spec=WiFiSpecCode.N1944_R23, max_iter=ldpc_iterations)
    entropy_decoder = EntropyBitwiseDecoder(DecoderWiFi(spec=WiFiSpecCode.N1944_R23, max_iter=ldpc_iterations),
                                            model_length=model_length, entropy_threshold=thr, clipping_factor=clipping_factor,
                                            min_data=args.min_data, window_length=window_len)
    entropy_flipping_decoder = EntropyBitwiseFlippingDecoder(DecoderWiFi(spec=WiFiSpecCode.N1944_R23, max_iter=ldpc_iterations,
                                                                         channel_model=channel),
                                                             model_length=model_length, entropy_threshold=thr,
                                                             min_data=args.min_data, window_length=window_len)
    no_errors = int(encoder.n * p)
    rx = []
    decoded_ldpc = []
    decoded_entropy = []
    decoded_flipping = []
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
        d = entropy_decoder.decode_buffer(channel_llr)
        decoded_entropy.append((*d, hamming_distance(Bits(auto=d[0]), encoded[tx_idx])))
        d = entropy_flipping_decoder.decode_buffer(corrupted)
        decoded_flipping.append((*d, hamming_distance(Bits(auto=d[0]), encoded[tx_idx])))
        print("p= ", p, " tx id: ", tx_idx)
    print("successful pure decoding for bit flip p=", p, ", is: ", sum(int(res[5] == 0) for res in decoded_ldpc), "/", n)
    print("successful entropy decoding for bit flip p=", p, ", is: ", sum(int(res[5] == 0) for res in decoded_entropy), "/",
          n)
    print("successful entropy flipping decoding for bit flip p=", p, ", is: ",
          sum(int(res[5] == 0) for res in decoded_flipping), "/", n)
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

    step_results["decoded_entropy"] = decoded_entropy
    step_results["entropy_buffer_success_rate"] = sum(int(res[5] == 0) for res in decoded_entropy) / float(n)
    step_results["entropy_decoder_ber"] = sum(
        hamming_distance(encoded[idx], Bits(auto=decoded_entropy[idx][0]))
        for idx in range(n)
    ) / float(n * len(encoded[0]))

    step_results["decoded_flipping"] = decoded_flipping
    step_results["flipping_buffer_success_rate"] = sum(int(res[5] == 0) for res in decoded_flipping) / float(n)
    step_results["flipping_decoder_ber"] = sum(
        hamming_distance(encoded[idx], Bits(auto=decoded_flipping[idx][0]))
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
    with open(os.path.join(path, "cmd.txt"), 'w') as f:
        f.write(cmd)

    with open(os.path.join(path, timestamp + '_simulation_entropy_vs_pure_LDPC.pickle'), 'wb') as f:
        pickle.dump(results, f)

    raw_ber = np.array([p['raw_ber'] for p in results])
    ldpc_ber = np.array([p['ldpc_decoder_ber'] for p in results])
    entropy_ber = np.array([p['entropy_decoder_ber'] for p in results])
    flipping_ber = np.array([p['flipping_decoder_ber'] for p in results])
    fig = plt.figure()
    plt.plot(raw_ber, ldpc_ber, 'bo', raw_ber, raw_ber, 'g^', raw_ber, entropy_ber, 'r*', raw_ber, flipping_ber, 'ms')
    plt.xlabel("BSC bit flip probability p")
    plt.ylabel("post decoding BER")
    fig.savefig(os.path.join(path, "ber_vs_error_p.eps"), dpi=150)

    figure = plt.figure()
    ldpc_buffer_success_rate = np.array([p['ldpc_buffer_success_rate'] for p in results])
    entropy_buffer_success_rate = np.array([p['entropy_buffer_success_rate'] for p in results])
    flipping_buffer_success_rate = np.array([p['flipping_buffer_success_rate'] for p in results])
    plt.plot(raw_ber, ldpc_buffer_success_rate, 'bo', raw_ber, entropy_buffer_success_rate, 'r*', raw_ber,
             flipping_buffer_success_rate, 'ms')
    plt.xlabel("BSC bit flip probability p")
    plt.ylabel("Decode success rate")
    figure.savefig(os.path.join(path, "buffer_success_rate_vs_error_p.eps"), dpi=150)

    summary = {"args": args, "raw_ber": raw_ber, "ldpc_ber": ldpc_ber, "entropy_ber": entropy_ber,
               "ldpc_buffer_success_rate": ldpc_buffer_success_rate,
               "entropy_buffer_success_rate": entropy_buffer_success_rate}
    with open(os.path.join(path, timestamp + '_summary_entropy_vs_pure_LDPC.pickle'), 'wb') as f:
        pickle.dump(summary, f)