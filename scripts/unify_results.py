import os
import pickle
import datetime
import matplotlib.pyplot as plt
import numpy as np

search_dir = "../runs/HC_eilat_July_2018"
results_file_name = '_simulation_entropy_vs_pure_LDPC_weighted_model.pickle'
last_results_file = "../runs/HC_eilat_July_2018/results/2022-03-20_18_18_6/2022-03-20_18_18_6_summary_entropy_vs_pure_LDPC_weighted_model.pickle"
resuls_dir = "../runs/HC_eilat_July_2018/results"

timestamp = f'{str(datetime.date.today())}_{str(datetime.datetime.now().hour)}_{str(datetime.datetime.now().minute)}_' \
                f'{str(datetime.datetime.now().second)}'

path = os.path.join(resuls_dir, timestamp)
os.mkdir(path)
files = os.listdir(search_dir)
files = sorted(list(filter(lambda f: f.endswith('.pickle'), files)))
files = [os.path.join(search_dir, f) for f in files]  # add path to each file
files.sort(key=lambda x: os.path.getmtime(x))

with open(last_results_file, 'rb') as f:
    last_summary = pickle.load(f)
    args = last_summary['args']

results = []
for file in files:
    with open(file, 'rb') as f:
        step_res = pickle.load(f)
    results.append(step_res)

with open(os.path.join(path, timestamp + results_file_name), 'wb') as f:
    pickle.dump(results, f)

raw_ber = np.array([p['raw_ber'] for p in results])
ldpc_ber = np.array([p['ldpc_decoder_ber'] for p in results])
entropy_ber = np.array([p['entropy_decoder_ber'] for p in results])
fig = plt.figure()
plt.plot(raw_ber, ldpc_ber, 'bo', raw_ber, raw_ber, 'g^', raw_ber, entropy_ber, 'r*')
plt.xlabel("BSC bit flip probability p")
plt.ylabel("post decoding BER")
fig.savefig(os.path.join(path, "ber_vs_error_p.eps"), dpi=150)
#
figure = plt.figure()
ldpc_buffer_success_rate = np.array([p['ldpc_buffer_success_rate'] for p in results])
entropy_buffer_success_rate = np.array([p['entropy_buffer_success_rate'] for p in results])
plt.plot(raw_ber, ldpc_buffer_success_rate, 'bo', raw_ber, entropy_buffer_success_rate, 'r*')
plt.xlabel("BSC bit flip probability p")
plt.ylabel("Decode success rate")
figure.savefig(os.path.join(path, "buffer_success_rate_vs_error_p.eps"), dpi=150)

args.minflip = raw_ber[0]
summary = {"args": args, "raw_ber": raw_ber, "ldpc_ber": ldpc_ber, "entropy_ber": entropy_ber,
           "ldpc_buffer_success_rate": ldpc_buffer_success_rate,
           "entropy_buffer_success_rate": entropy_buffer_success_rate}
with open(os.path.join(path, f'{timestamp}_summary{results_file_name}'), 'wb') as f:
    pickle.dump(summary, f)
