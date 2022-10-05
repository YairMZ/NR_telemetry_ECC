import pickle
import numpy as np
from ldpc.utils import AList
from ldpc.decoder import LogSpaDecoder
from decoders import ClassifyingEntropyDecoder
from utils.bit_operations import hamming_distance
from typing import Any
import argparse
import datetime
import os
from scipy.io import savemat
import lzma
import pandas as pd
from utils import setup_logger


parser = argparse.ArgumentParser(description='Run decoding on experimental data.')
parser.add_argument("--ldpciterations", default=130, help="number of iterations of  LDPC decoder", type=int)
parser.add_argument("--ent_threshold", default=0.36, help="entropy threshold to be used in entropy decoder", type=float)
parser.add_argument("--window_len", default=50, help="number of previous samples to use, if 0 all are used", type=int)
parser.add_argument("--clipping_factor", default=2, help="dictates maximal and minimal llr", type=int)
parser.add_argument("--conf_center", default=40, help="center of model sigmoid", type=int)
parser.add_argument("--conf_slope", default=0.35, help="slope of model sigmoid", type=float)
parser.add_argument("--dec_type", default="BP", help="scheme for determining confidence", type=str)
parser.add_argument("--model_length", default="all", help="model length", type=str)
parser.add_argument("--experiment_date", default="14", help="date of experiment", type=str)

args = parser.parse_args()

ldpc_iterations = args.ldpciterations
thr = args.ent_threshold
clipping_factor = args.clipping_factor
if args.experiment_date == 'all':
    tx = np.genfromtxt('data/feb_14_tx.csv', dtype=np.uint8, delimiter=',')
    rx = np.genfromtxt('data/feb_14_llr.csv', dtype=np.float_, delimiter=',')

    temp = np.genfromtxt('data/feb_16_tx.csv', dtype=np.uint8, delimiter=',')
    tx = np.vstack((tx, temp))
    temp = np.genfromtxt('data/feb_16_llr.csv', dtype=np.float_, delimiter=',')
    rx = np.vstack((rx, temp))
else:
    tx = np.genfromtxt(f'data/feb_{args.experiment_date}_tx.csv', dtype=np.uint8, delimiter=',')
    rx = np.genfromtxt(f'data/feb_{args.experiment_date}_llr.csv', dtype=np.float_, delimiter=',')

h = AList.from_file("spec/4098_3095_non_sys_h.alist")
number_of_messages, n = rx.shape
k = 984
window_len = args.window_len if args.window_len > 0 else None

model_length = k if args.model_length == 'info' else n

timestamp = f'{str(datetime.date.today())}_{str(datetime.datetime.now().hour)}_{str(datetime.datetime.now().minute)}_' \
            f'{str(datetime.datetime.now().second)}'

path = os.path.join("results/", timestamp)
os.mkdir(path)

logger = setup_logger(name=__file__, log_file=os.path.join(path, 'log.log'))


logger.info(__file__)
logger.info(f"number of buffers to process: {number_of_messages}")
logger.info(f"number of ldpc decoder iterations: {ldpc_iterations}")
logger.info(f"entropy threshold used in entropy decoder: {thr}")
logger.info(f"entropy decoder window length: {window_len}")
logger.info(f"clipping factor: {clipping_factor}")
logger.info(f"model center: {args.conf_center}")
logger.info(f"model slope: {args.conf_slope}")
logger.info(f"decoder type: {args.dec_type}")
logger.info(f"model_length: {args.model_length}")
logger.info(f"experiment_date: {args.experiment_date}")

cmd = f'python {__file__} --ldpciterations {ldpc_iterations} --ent_threshold {thr} --clipping_factor {clipping_factor} ' \
      f'--conf_center {args.conf_center} --conf_slope {args.conf_slope} --dec_type {args.dec_type} --model_length ' \
      f'{args.model_length} --experiment_date {args.experiment_date}'
if window_len is not None:
    cmd += f' --window_len {window_len}'
else:
    cmd += ' --window_len 0'

with open(os.path.join(path, "cmd.txt"), 'w') as f:
    f.write(cmd)
logger.info(cmd)

ldpc_decoder = LogSpaDecoder(h=h.to_array(), max_iter=args.ldpciterations, decoder_type=args.dec_type,
                             info_idx=np.array([True] * k + [False] * (n - k)))
entropy_decoder = ClassifyingEntropyDecoder(
    LogSpaDecoder(h=h.to_array(), max_iter=args.ldpciterations, decoder_type=args.dec_type, info_idx=np.array(
        [True] * k + [False] * (n - k))),
    model_length=model_length, entropy_threshold=thr, clipping_factor=clipping_factor,
    classifier_training=0, n_clusters=1, window_length=window_len, conf_center=args.conf_center,
    conf_slope=args.conf_slope, bit_flip=0)

decoded_ldpc = []
decoded_entropy = []
errors = 2 * np.ones(rx.shape)

for tx_idx in range(number_of_messages):
    d = ldpc_decoder.decode(rx[tx_idx, :])
    ldpc_success = d[2]
    if tx[tx_idx, 0] < 2:  # valid tx identified
        errors[tx_idx, :] = tx[tx_idx, :] != (rx[tx_idx, :] < 0)
        hamm = hamming_distance(d[0], tx[tx_idx, :])
    else:
        hamm = -1
    decoded_ldpc.append((*d, hamm))
    d = entropy_decoder.decode_buffer(rx[tx_idx, :])
    entropy_success = d[2]
    if tx[tx_idx, 0] < 2:  # valid tx identified
        hamm = hamming_distance(d[0], tx[tx_idx, :])
    else:
        hamm = -1
    decoded_entropy.append((*d, hamm))
    logger.info(f"tx id:, {tx_idx}, ldpc: {ldpc_success}, entropy: {entropy_success}")

logger.info(f"successful pure decoding is: {sum(res[2] for res in decoded_ldpc)}/{number_of_messages}")
logger.info(f"successful entropy decoding is: {sum(res[2] for res in decoded_entropy)}/{number_of_messages}")

# analyze results
# log data
good_tx = tx[:, 0] != 2
bad_tx = tx[:, 0] == 2
input_ber = np.sum(errors, axis=1) / n
input_ber[bad_tx] = -1
info_errors = np.sum(errors[:, :k], axis=1)
info_errors[bad_tx] = -1
parity_errors = np.sum(errors[:, k:], axis=1)
parity_errors[bad_tx] = -1
zipped = [[en, r, er, inf, pa, i_ber] for en, r, er, inf, pa, i_ber in
          zip(tx, rx, errors, info_errors, parity_errors, input_ber)]

# noinspection PyDictCreation
results: dict[str, Any] = {"data": pd.DataFrame(zipped, columns=["encoded", "corrupted", "error_idx", "info_errors",
                                                                 "parity_errors", "input_ber"])}
# params
results["buffer_len"] = n
results["number_of_buffers"] = number_of_messages
results["max_ldpc_iterations"] = ldpc_iterations

# decoding
decoded_entropy_df = pd.DataFrame(decoded_entropy,
                                  columns=["estimate", "llr", "decode_success", "iterations", "syndrome",
                                           "vnode_validity", "dist", "structural_idx", "cluster_label", "hamming"])
results["decoded_entropy"] = decoded_entropy_df
decoded_ldpc_df = pd.DataFrame(decoded_ldpc,
                               columns=["estimate", "llr", "decode_success", "iterations", "syndrome",
                                        "vnode_validity", "hamming"])
results['decoded_ldpc'] = decoded_ldpc_df

# performance
results["ldpc_buffer_success_rate"] = sum(res[2] for res in decoded_ldpc) / float(number_of_messages)
results["entropy_buffer_success_rate"] = sum(res[2] for res in decoded_entropy) / float(number_of_messages)
results["ldpc_decoder_ber"] = sum(res[-1] for res in decoded_ldpc if res[-1] >= 0) / float(n * sum(good_tx))
results["entropy_decoder_ber"] = sum(res[-1] for res in decoded_entropy if res[-1] >= 0) / float(n * sum(good_tx))
results["input_ber"] = input_ber

# save results
with lzma.open(
        os.path.join(path, f'{timestamp}_experimental_data_analysis.xz'), "wb") as f:
    pickle.dump(results, f)

results['data'] = results['data'].to_dict("list")
results['decoded_entropy'] = results['decoded_entropy'].to_dict("list")
results['decoded_ldpc'] = results['decoded_ldpc'].to_dict("list")
results['args'] = args

savemat(os.path.join(path, f'{timestamp}_experimental_data_analysis.mat'), results, do_compression=True)

summary_txt = f'successful pure decoding is: {sum(res[2] for res in decoded_ldpc)}/{number_of_messages}\n' \
          f'successful entropy decoding is: {sum(res[2] for res in decoded_entropy)}/{number_of_messages}'
with open(os.path.join(path, "summary.txt"), 'w') as f:
    f.write(summary_txt)
