import pickle
import numpy as np
from ldpc.utils import AList
from ldpc.decoder import LogSpaDecoder
from decoders import MavlinkRectifyingDecoder
from inference import BufferSegmentation, BufferModel
from protocol_meta import dialect_meta as meta
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
parser.add_argument("--segiterations", default=1, help="number of exchanges between LDPC and CB decoder", type=int)
parser.add_argument("--dec_type", default="BP", help="scheme for determining confidence", type=str)
parser.add_argument("--valid_factor", default=2.0, help="valid factor", type=float)
parser.add_argument("--invalid_factor", default=0.7, help="invalid factor", type=float)
parser.add_argument("--valid_threshold", default=0.03,
                    help="valid_threshold  as outlier probability for classifying a field as valid", type=float)
parser.add_argument("--invalid_threshold", default=0.08,
                    help="invalid_threshold  as outlier probability for classifying a field as invalid", type=float)
parser.add_argument("--window_len", default=50, help="number of previous samples to use, if 0 all are used", type=int)
parser.add_argument("--learn", default=0, help="0 predefined model is used. 1 model is learned online", type=int)
parser.add_argument("--conf_center", default=40, help="center of model sigmoid", type=int)
parser.add_argument("--conf_slope", default=0.35, help="slope of model sigmoid", type=float)
parser.add_argument("--experiment_date", default="14", help="date of experiment", type=str)

args = parser.parse_args()

ldpc_iterations = args.ldpciterations
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

k = 984
bs = BufferSegmentation(meta.protocol_parser)
_, _, buffer_structure = bs.segment_buffer(np.packbits(tx[0, :k]).tobytes())  # segment first buffer to get structure
data_model = BufferModel.load('data/model_2022.json')
h = AList.from_file("spec/4098_3095_non_sys_h.alist")
number_of_messages, n = rx.shape
window_len = args.window_len if args.window_len > 0 else None

timestamp = f'{str(datetime.date.today())}_{str(datetime.datetime.now().hour)}_{str(datetime.datetime.now().minute)}_' \
            f'{str(datetime.datetime.now().second)}'

path = os.path.join("results/", timestamp)
os.mkdir(path)

logger = setup_logger(name=__file__, log_file=os.path.join(path, 'log.log'))

logger.info(__file__)
logger.info(f"number of buffers to process: {number_of_messages}")
logger.info(f"number of ldpc decoder iterations: {ldpc_iterations}")
logger.info(f"decoder window length: {window_len}")
logger.info(f"model center: {args.conf_center}")
logger.info(f"model slope: {args.conf_slope}")
logger.info(f"decoder type: {args.dec_type}")
logger.info(f"experiment_date: {args.experiment_date}")
logger.info(f"valid_threshold used in decoder: {args.valid_threshold}")
logger.info(f"invalid_threshold used in decoder: {args.invalid_threshold}")
logger.info(f"valid_factor: {args.valid_factor}")
logger.info(f"invalid_factor: {args.invalid_factor}")
logger.info(f"learn: {args.learn}")
logger.info(f"seg_iter: {args.segiterations}")

cmd = f'python {__file__} --ldpciterations {ldpc_iterations}  --learn {args.learn} --segiterations {args.segiterations} ' \
      f'--valid_factor {args.valid_factor} --invalid_factor {args.invalid_factor} --valid_threshold {args.valid_threshold} ' \
      f'--invalid_threshold {args.invalid_threshold} --window_len {window_len} --conf_center {args.conf_center} ' \
      f'--conf_slope {args.conf_slope} --dec_type {args.dec_type} --experiment_date {args.experiment_date}'

if window_len is not None:
    cmd += f' --window_len {window_len}'
else:
    cmd += ' --window_len 0'

with open(os.path.join(path, "cmd.txt"), 'w') as f:
    f.write(cmd)
logger.info(cmd)

model_length = k

ldpc_decoder = LogSpaDecoder(h=h.to_sparse(), max_iter=args.ldpciterations, decoder_type=args.dec_type,
                             info_idx=np.array([True] * k + [False] * (n - k)))
rectify_decoder = MavlinkRectifyingDecoder(LogSpaDecoder(h=h.to_sparse(), max_iter=args.ldpciterations,
                                                         decoder_type=args.dec_type, info_idx=np.array(
        [True] * k + [False] * (n - k))),
                                           model_length, args.valid_threshold, args.invalid_threshold,
                                           1, args.valid_factor, args.invalid_factor,
                                           0, False, window_len,
                                           data_model, args.conf_center, args.conf_slope,
                                           segmentation_iterations=args.segiterations)

rectify_decoder.set_buffer_structures([buffer_structure])
decoded_ldpc = []
decoded_rect = []
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
    d = rectify_decoder.decode_buffer(rx[tx_idx, :], np.where(errors[tx_idx, :] == 1)[0])
    mavlink_success = d[2]
    if tx[tx_idx, 0] < 2:  # valid tx identified
        hamm = hamming_distance(d[0], tx[tx_idx, :])
    else:
        hamm = -1
    decoded_rect.append((*d, hamm))
    logger.info(f"tx id:, {tx_idx}, ldpc: {ldpc_success}, mavlink: {mavlink_success}")

logger.info(f"successful pure decoding is: {sum(res[2] for res in decoded_ldpc)}/{number_of_messages}")
logger.info(f"successful mavlink decoding is: {sum(res[2] for res in decoded_rect)}/{number_of_messages}")

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
decoded_rect_df = pd.DataFrame(decoded_rect,
                               columns=["estimate", "llr", "decode_success", "iterations", "syndrome",
                                        "vnode_validity", "cluster_label", "segmented_bits", "good_field_idx",
                                        "bad_field_idx", "classifier_performance", "forcing_performance",
                                        "hamming"])
results["decoded_rect"] = decoded_rect_df
decoded_ldpc_df = pd.DataFrame(decoded_ldpc,
                               columns=["estimate", "llr", "decode_success", "iterations", "syndrome",
                                        "vnode_validity", "hamming"])
results['decoded_ldpc'] = decoded_ldpc_df

# performance
results["ldpc_buffer_success_rate"] = sum(res[2] for res in decoded_ldpc) / float(number_of_messages)
results["mavlink_buffer_success_rate"] = sum(res[2] for res in decoded_rect) / float(number_of_messages)
results["ldpc_decoder_ber"] = sum(res[-1] for res in decoded_ldpc if res[-1] >= 0) / float(n * sum(good_tx))
results["rectify_decoder_ber"] = sum(res[-1] for res in decoded_rect if res[-1] >= 0) / float(n * sum(good_tx))
results["input_ber"] = input_ber

# save results
with lzma.open(
        os.path.join(path, f'{timestamp}_experimental_data_analysis.xz'), "wb") as f:
    pickle.dump(results, f)

results['data'] = results['data'].to_dict("list")
results['decoded_rect'] = results['decoded_rect'].to_dict("list")
results['decoded_ldpc'] = results['decoded_ldpc'].to_dict("list")
results['args'] = args

savemat(os.path.join(path, f'{timestamp}_experimental_data_analysis.mat'), results, do_compression=True)

summary_txt = f'successful pure decoding is: {sum(res[2] for res in decoded_ldpc)}/{number_of_messages}\n' \
              f'successful mavlink decoding is: {sum(res[2] for res in decoded_rect)}/{number_of_messages}'
with open(os.path.join(path, "summary.txt"), 'w') as f:
    f.write(summary_txt)
