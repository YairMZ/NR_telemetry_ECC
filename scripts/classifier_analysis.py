import pickle
from typing import Any
from bitstring import Bits, BitArray
import numpy as np
from ldpc.encoder import EncoderWiFi
from ldpc.wifi_spec_codes import WiFiSpecCode
from ldpc.decoder import bsc_llr
from numpy.typing import NDArray
from inference import BufferSegmentation, BufferModel
from protocol_meta import dialect_meta as meta
import argparse
import datetime
import os
from multiprocessing import Pool
from scipy.io import savemat
import pandas as pd
from utils import setup_logger
import shutil

parser = argparse.ArgumentParser(description='Run decoding on simulated data using multiprocessing.')
parser.add_argument("--minflip", default=33 * 1e-3, help="minimal bit flip probability to consider", type=float)
parser.add_argument("--maxflip", default=70 * 1e-3, help="maximal bit flip probability to consider", type=float)
parser.add_argument("--nflips", default=20, help="number of bit flips to consider", type=int)
parser.add_argument("--multiply_data", default=0, help="multiplies amount of buffers by 2 to power of arg", type=int)
parser.add_argument("--processes", default=0, help="number of processes to spawn", type=int)
parser.add_argument("--msg_delay", default="20000", help="sampling delay", type=str)
parser.add_argument("--threshold", default=0,
                    help="threshold  as outlier probability for classifying a field as valid or invalid", type=float)
parser.add_argument("--window_len", default=50,
                    help="number of previous samples to use for training the model, if 0 all are used", type=int)
parser.add_argument("--learn", default=0, help="0 predefined model is used. 1 model is learned online", type=int)
parser.add_argument("--conf_center", default=40, help="center of model sigmoid", type=int)
parser.add_argument("--conf_slope", default=0.35, help="slope of model sigmoid", type=float)
parser.add_argument("--n_clusters", default=1, help="number of clusters", type=int)

args = parser.parse_args()

processes = args.processes if args.processes > 0 else None

with open('../runs/HC_eilat_July_2018/data/hc_to_ship.pickle', 'rb') as f:
    hc_tx = pickle.load(f)

hc_bin_data = [Bits(auto=tx.get("bin")) for tx in hc_tx.get(args.msg_delay)]
n = len(hc_bin_data)
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
    _, _, buffer_structure = bs.segment_buffer(binary_data[:416])
    buffer_structures = [buffer_structure]
    _, _, buffer_structure = bs.segment_buffer(binary_data[416:864])
    buffer_structures.append(buffer_structure)
    _, _, buffer_structure = bs.segment_buffer(binary_data[864:])
    buffer_structures.append(buffer_structure)
else:
    raise ValueError("n_clusters must be 1, 2 or 3")

model_length = encoder.k
# encoded structure: {starting byte index in buffer: msg_id}
# {0: 33, 52: 234, 72: 30, 108: 212, 135: 218}
# last 9 bytes (72 bits) are padding and not messages. Thus last message ends at byte index 152
# bit indices:
# {0: 33, 416: 234, 576: 30, 864: 212, 1080: 218}

logger = setup_logger(name=__file__, log_file=os.path.join("results/", 'log.log'))


def classifier_characterization(valid_field_p: list[tuple[str, float, float]], valid_bits_p: NDArray[np.float_], thr,
                                damaged_fields_idx: NDArray[np.int_]
                                ) -> dict[str, Any]:
    good_bits: NDArray[np.bool_] = valid_bits_p > 1 - thr
    bad_bits: NDArray[np.bool_] = valid_bits_p < thr
    good_fields: list[int] = [idx for idx, vfp in enumerate(valid_field_p) if vfp[1] > 1 - thr]
    bad_fields: list[int] = [idx for idx, vfp in enumerate(valid_field_p) if vfp[1] < thr]

    good_fields_true = 0
    good_fields_false = 0
    bad_fields_true = 0
    bad_fields_false = 0
    for idx in good_fields:
        if idx in damaged_fields_idx:
            good_fields_false += 1
        else:
            good_fields_true += 1
    for idx in bad_fields:
        if idx in damaged_fields_idx:
            bad_fields_true += 1
        else:
            bad_fields_false += 1

    n_fields_in_buffer = len(valid_field_p)
    n_damaged_fields_in_buffer = len(damaged_fields_idx)

    true_positive = good_fields_true
    false_positive = good_fields_false
    positive = n_fields_in_buffer - n_damaged_fields_in_buffer
    negative = n_damaged_fields_in_buffer
    # true positive, false positive, true negative, false negative, positive, negative
    good_fields_performance = np.array([true_positive, false_positive, negative - false_positive,
                                        positive - true_positive, positive, negative], dtype=np.int_)

    true_positive = bad_fields_true
    false_positive = bad_fields_false
    positive = n_damaged_fields_in_buffer
    negative = n_fields_in_buffer - n_damaged_fields_in_buffer
    bad_fields_performance = np.array([true_positive, false_positive, negative - false_positive,
                                       positive - true_positive, positive, negative], dtype=np.int_)

    return {"good_bits": good_bits, "bad_bits": bad_bits, "good_fields": good_fields, "bad_fields": bad_fields,
            "n_fields_in_buffer": n_fields_in_buffer, "n_damaged_fields_in_buffer": n_damaged_fields_in_buffer,
            "good_fields_performance": good_fields_performance, "bad_fields_performance": bad_fields_performance}


def forcing_characterization(valid_field_p: list[tuple[str, float, float]], valid_bits_p: NDArray[np.float_],
                             bitwise_std: NDArray[np.float_], label: int,
                             damaged_fields_idx: NDArray[np.int_], errors: NDArray[np.int_], data_model: BufferModel,
                             rx: NDArray[np.int_], tx: NDArray[np.int_]) -> dict[str, Any]:
    buffer_len_bits = len(rx)

    forced_bits = np.where((valid_bits_p <= 0) & (bitwise_std == 0))[0]
    correctly_forced_bits = np.intersect1d(forced_bits, errors)
    incorrectly_forced_bits = np.setdiff1d(forced_bits, errors)
    missed_forced_bits = np.setdiff1d(errors, forced_bits)
    forced_fields = np.array([idx for idx, vfp in enumerate(valid_field_p) if (vfp[1] <= 0 and vfp[2] == 0)], dtype=np.int_)
    correctly_forced_fields = np.intersect1d(forced_fields, damaged_fields_idx)
    incorrectly_forced_fields = np.setdiff1d(forced_fields, damaged_fields_idx)
    missed_forced_fields = np.setdiff1d(damaged_fields_idx, forced_fields)

    true_positive = correctly_forced_bits.size
    false_positive = incorrectly_forced_bits.size
    positive = errors.size
    negative = buffer_len_bits - positive
    forced_bits_performance = np.array([true_positive, false_positive, negative - false_positive,
                                        positive - true_positive, positive, negative], dtype=np.int_)

    n_fields_in_buffer = len(valid_field_p)
    n_damaged_fields_in_buffer = len(damaged_fields_idx)
    true_positive = correctly_forced_fields.size
    false_positive = incorrectly_forced_fields.size
    positive = n_damaged_fields_in_buffer
    negative = n_fields_in_buffer - positive
    forced_fields_performance = np.array([true_positive, false_positive, negative - false_positive,
                                          positive - true_positive, positive, negative], dtype=np.int_)

    model_bits = data_model.bitwise_model_mean(forced_bits, buffer_len_bits // 8, buffer_structures[label])
    bits_flipped_from_forced = forced_bits[(model_bits ^ rx[forced_bits]).astype(np.bool_)]
    n_bits_flipped = bits_flipped_from_forced.size
    correctly_flipped_bits = np.intersect1d(bits_flipped_from_forced, errors)
    incorrectly_flipped_bits = np.setdiff1d(bits_flipped_from_forced, errors)

    forcing_quality = np.array([errors.size, forced_bits.size, n_bits_flipped, correctly_flipped_bits.size,
                                incorrectly_flipped_bits.size, sum(tx[forced_bits] ^ model_bits) / forced_bits.size], dtype=np.float_)

    true_positive = correctly_flipped_bits.size
    false_positive = incorrectly_flipped_bits.size
    positive = errors.size
    negative = buffer_len_bits - positive
    flipped_bits_performance = np.array([true_positive, false_positive, negative - false_positive,
                                        positive - true_positive, positive, negative], dtype=np.int_)

    return {"forced_fields": forced_fields, "forced_bits": forced_bits, "correctly_forced_bits": correctly_forced_bits,
            "incorrectly_forced_bits": incorrectly_forced_bits, "missed_forced_bits": missed_forced_bits,
            "correctly_forced_fields": correctly_forced_fields, "incorrectly_forced_fields": incorrectly_forced_fields,
            "missed_forced_fields": missed_forced_fields, "forced_bits_performance": forced_bits_performance,
            "forced_fields_performance": forced_fields_performance, "forcing_quality": forcing_quality,
            "flipped_bits_performance": flipped_bits_performance
            }


def simulation_step(p: float) -> dict[str, Any]:
    global model_length
    global args
    global window_len
    global n
    global spec
    global encoder
    global buffer_structures
    global logger

    if bool(args.learn):
        data_model = BufferModel()
    else:
        data_model = BufferModel.load('../runs/HC_eilat_July_2018/data/model_2018_all.json')

    channel = bsc_llr(p=p)
    no_errors = int(encoder.n * p)
    rx: list[NDArray[np.int_]] = []
    errors = np.vstack(
        tuple(sorted(rng.choice(encoder.n, size=no_errors, replace=False))
              for _ in range(n))
    )
    step_results: dict[str, Any] = {"p": p, "number_of_buffers": n, "no_errors": no_errors, "errors": errors,
                                    "raw_ber": no_errors / encoder.n, "buffer_len": encoder.n, "window_len": window_len,
                                    "n_clusters": args.n_clusters, "learn": args.learn, "model_length": model_length}

    running_idx = 0
    forced_bits: list[NDArray[np.int_]] = [None] * n
    good_fields: list[list[int]] = [None] * n
    bad_fields: list[list[int]] = [None] * n
    good_bits: list[NDArray[np.bool_]] = [None] * n
    bad_bits: list[NDArray[np.bool_]] = [None] * n

    # good_fields_performance: specifies the classification performance of fields which are deemed good / valid
    # bad_fields_performance: specifies the classification performance of fields which are deemed bad / invalid
    good_fields_performance: NDArray[np.int_] = np.zeros((n, 6), dtype=np.int_)
    bad_fields_performance: NDArray[np.int_] = np.zeros((n, 6), dtype=np.int_)
    n_fields_in_buffer: NDArray[np.int_] = np.zeros(n, dtype=np.int_)
    n_damaged_fields_in_buffer: NDArray[np.int_] = np.zeros(n, dtype=np.int_)

    forced_fields: list[NDArray[np.int_]] = [None] * n
    damaged_fields: list[NDArray[np.int_]] = [None] * n

    # forced fields/bits are bits which are forced to be 0 or 1 by the decoder according to the model's mean. This is done for
    # fields with zero probability of being good
    correctly_forced_fields: list[NDArray[np.int_]] = [None] * n
    incorrectly_forced_fields: list[NDArray[np.int_]] = [None] * n
    missed_forced_fields: list[NDArray[np.int_]] = [None] * n
    correctly_forced_bits: list[NDArray[np.int_]] = [None] * n
    incorrectly_forced_bits: list[NDArray[np.int_]] = [None] * n
    missed_forced_bits: list[NDArray[np.int_]] = [None] * n
    forced_fields_performance: NDArray[np.int_] = np.zeros((n, 6), dtype=np.int_)
    forced_bits_performance: NDArray[np.int_] = np.zeros((n, 6), dtype=np.int_)
    # forcing quality states how good is te mean of the model as a predictor of actual bit values.
    # It is calculated as the error rate of the buffer when the mean is used as a predictor of the actual bit values versus
    # the error rate of the buffer when the channel bit values are used.
    # The matrix also contains the number of flipped bits and a breakdown of good versus bad flips.
    forcing_quality: NDArray[np.float_] = np.zeros((n, 6), dtype=np.float_)
    flipped_bits_performance: NDArray[np.int_] = np.zeros((n, 6), dtype=np.int_)

    for tx_idx in range(n):
        corrupted = BitArray(encoded[tx_idx])
        corrupted.invert(errors[tx_idx])
        corrupted = np.array(corrupted, dtype=np.int_)
        rx.append(corrupted)
        channel_llr = channel(corrupted)

        # predict
        valid_field_p, valid_bits_p, bits_field_std = data_model.predict(corrupted, buffer_structures[running_idx])
        damaged = data_model.find_damaged_fields(errors[tx_idx], buffer_structures[running_idx], len(corrupted) // 8)
        damaged_fields[tx_idx] = np.array(tuple({field[1] for field in damaged}), dtype=np.int_)

        # characterize the classifier
        classifier_res = classifier_characterization(valid_field_p, valid_bits_p, args.threshold, damaged_fields[tx_idx])
        good_bits[tx_idx] = classifier_res["good_bits"]
        bad_bits[tx_idx] = classifier_res["bad_bits"]
        good_fields[tx_idx] = classifier_res["good_fields"]
        bad_fields[tx_idx] = classifier_res["bad_fields"]
        n_fields_in_buffer[tx_idx] = classifier_res["n_fields_in_buffer"]
        n_damaged_fields_in_buffer[tx_idx] = classifier_res["n_damaged_fields_in_buffer"]
        good_fields_performance[tx_idx] = classifier_res["good_fields_performance"]
        bad_fields_performance[tx_idx] = classifier_res["bad_fields_performance"]

        # look for constant bits for forcing
        forcing_res = forcing_characterization(valid_field_p, valid_bits_p, bits_field_std, running_idx, damaged_fields[tx_idx],
                                               errors[tx_idx], data_model, corrupted, np.array(encoded[tx_idx], dtype=np.int_))
        forced_fields[tx_idx] = forcing_res["forced_fields"]
        forced_bits[tx_idx] = forcing_res["forced_bits"]
        correctly_forced_bits[tx_idx] = forcing_res["correctly_forced_bits"]
        incorrectly_forced_bits[tx_idx] = forcing_res["incorrectly_forced_bits"]
        missed_forced_bits[tx_idx] = forcing_res["missed_forced_bits"]
        correctly_forced_fields[tx_idx] = forcing_res["correctly_forced_fields"]
        incorrectly_forced_fields[tx_idx] = forcing_res["incorrectly_forced_fields"]
        missed_forced_fields[tx_idx] = forcing_res["missed_forced_fields"]
        forced_fields_performance[tx_idx] = forcing_res["forced_fields_performance"]
        forced_bits_performance[tx_idx] = forcing_res["forced_bits_performance"]
        forcing_quality[tx_idx] = forcing_res["forcing_quality"]
        flipped_bits_performance[tx_idx] = forcing_res["flipped_bits_performance"]

        running_idx = (running_idx + 1) % args.n_clusters
        logger.info(f"p= {p}, tx id: {tx_idx}")

    # log data
    info_errors = np.sum(errors < encoder.k, axis=1)
    parity_errors = np.sum(errors >= encoder.k, axis=1)
    zipped = [[np.array(en, dtype=np.int_), np.array(r, dtype=np.int_), er, inf, pa] for en, r, er, inf, pa in
              zip(encoded, rx, errors, info_errors, parity_errors)]
    step_results["data"] = pd.DataFrame(zipped, columns=["encoded", "corrupted", "error_idx", "info_errors",
                                                         "parity_errors"])

    classifier_df = pd.DataFrame({"good_fields": good_fields, "bad_fields": bad_fields,
                                  "good_bits": good_bits, "bad_bits": bad_bits,
                                  "damaged_fields": damaged_fields, "cluster_label": running_idx})
    step_results["classifier"] = classifier_df
    forcing_df = pd.DataFrame({"forced_fields": forced_fields, "forced_bits": forced_bits,
                               "correctly_forced_bits": correctly_forced_bits,
                               "incorrectly_forced_bits": incorrectly_forced_bits,
                               "missed_forced_bits": missed_forced_bits,
                               "correctly_forced_fields": correctly_forced_fields,
                               "incorrectly_forced_fields": incorrectly_forced_fields,
                               "missed_forced_fields": missed_forced_fields})
    step_results["forcing"] = forcing_df
    # performance
    step_results["good_fields_performance"] = good_fields_performance
    step_results["bad_fields_performance"] = bad_fields_performance
    step_results["forced_fields_performance"] = forced_fields_performance
    step_results["forced_bits_performance"] = forced_bits_performance
    step_results["forcing_quality"] = forcing_quality
    step_results["flipped_bits_performance"] = flipped_bits_performance


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
    logger.info(f"processes: {args.processes}")
    logger.info(f"multiply data: {args.multiply_data}")
    logger.info(f"msg_delay: {args.msg_delay} ")
    logger.info(f"n_clusters: {args.n_clusters}")
    logger.info(f"threshold used in decoder: {args.threshold}")
    logger.info(f"window length: {window_len}")
    logger.info(f"confidence center: {args.conf_center}")
    logger.info(f"confidence slope: {args.conf_slope}")
    logger.info(f"learn: {args.learn}")

    cmd = f'python {__file__} --minflip {args.minflip} --maxflip {args.maxflip} --nflips {args.nflips}  ' \
          f'--multiply_data {args.multiply_data} --msg_delay {args.msg_delay} --n_clusters {args.n_clusters} ' \
          f'--threshold {args.threshold} --conf_center {args.conf_center} --conf_slope {args.conf_slope} --learn {args.learn}'
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
        # with open(os.path.join(path, f'{timestamp}_classifier_analysis_2018.pickle'), "wb") as f:
        #     pickle.dump(results, f)
        # logger.info("saved pickle results file")

        good_fields_performance = np.array([p['good_fields_performance'].sum(axis=0) for p in results])
        bad_fields_performance = np.array([p['bad_fields_performance'].sum(axis=0) for p in results])
        forced_fields_performance = np.array([p['forced_fields_performance'].sum(axis=0) for p in results])
        forced_bits_performance = np.array([p['forced_bits_performance'].sum(axis=0) for p in results])
        forcing_quality = np.array([p['forcing_quality'].sum(axis=0) for p in results])
        forcing_quality[:,-1] = forcing_quality[:, -1] / n
        flipped_bits_performance = np.array([p['flipped_bits_performance'].sum(axis=0) for p in results])
        # fig = plt.figure()
        # plt.plot(raw_ber, ldpc_ber, 'bo', raw_ber, raw_ber, 'g^', raw_ber, rect_ber, 'r*')
        # plt.xlabel("BSC bit flip probability p")
        # plt.ylabel("post decoding BER")
        # fig.savefig(os.path.join(path, "ber_vs_error_p.eps"), dpi=150)
        #
        # figure = plt.figure()
        # ldpc_buffer_success_rate = np.array([p['ldpc_buffer_success_rate'] for p in results])
        # rect_buffer_success_rate = np.array([p['rect_buffer_success_rate'] for p in results])
        # plt.plot(raw_ber, ldpc_buffer_success_rate, 'bo', raw_ber, rect_buffer_success_rate, 'r*')
        # plt.xlabel("BSC bit flip probability p")
        # plt.ylabel("Decode success rate")
        # figure.savefig(os.path.join(path, "buffer_success_rate_vs_error_p.eps"), dpi=150)

        # logger.info("saved figures")
        summary = {"args": args, "good_fields_performance": good_fields_performance,
                   "bad_fields_performance": bad_fields_performance, "forced_fields_performance": forced_fields_performance,
                   "forced_bits_performance": forced_bits_performance, "bit_flip_p": bit_flip_p,
                   "forcing_quality": forcing_quality, "flipped_bits_performance": flipped_bits_performance}
        # with open(os.path.join(path, f'{timestamp}_summary_classifier_analysis_2018.pickle'), 'wb') as f:
        #     pickle.dump(summary, f)
        savemat(os.path.join(path, f'{timestamp}_summary_classifier_analysis_2018.mat'), summary)
        logger.info("saved summary")

        for step in results:
            step['data'] = step['data'].to_dict("list")
            step['classifier'] = step['classifier'].to_dict("list")
            step["forcing"] = step["forcing"].to_dict("list")

        summary = {"results": results}
        savemat(os.path.join(path, f'{timestamp}_classifier_analysis_2018.mat'),
                summary, do_compression=True)
        logger.info("saved results to mat file")
        shutil.move("results/log.log", os.path.join(path, "log.log"))
    except Exception as e:
        logger.exception(e)
        shutil.move("results/log.log", os.path.join(path, "log.log"))
        raise e
