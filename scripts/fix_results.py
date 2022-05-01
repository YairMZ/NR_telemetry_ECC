import pickle
import lzma
from scipy.io import savemat
import os

path = "/home/lar/Documents/repos/NR_telemetry_ECC/runs/HC_eilat_July_2018/results/2022-05-01_0_36_56"
timestamp = path.split("/")[-1]
dec_type = "BP"

# load summary
with open(os.path.join(path, f'{timestamp}_summary_entropy_vs_pure_LDPC_weighted_model_{dec_type}_decoder.pickle'),
          'rb') as f:
    summary = pickle.load(f)

# load results
with open(
        os.path.join(path, f'{timestamp}_simulation_entropy_vs_pure_LDPC_weighted_model_{dec_type}_decoder.pickle'),
        "rb") as f:
    results = pickle.load(f)

for step in results:
    step['data'] = step['data'].to_dict("list")
decoded_entropy = [step.pop('decoded_entropy').to_dict("list") for step in results]
decoded_ldpc = [step.pop('decoded_ldpc').to_dict("list") for step in results]

summary.update({"stats": results})

savemat(os.path.join(path, f'{timestamp}_data_{dec_type}_decoder.mat'),
        summary, do_compression=True)
savemat(os.path.join(path, f'{timestamp}_decoded_ldpc_{dec_type}_decoder.mat'),
        {"decoded_ldpc": decoded_ldpc}, do_compression=True)
savemat(os.path.join(path, f'{timestamp}_decoded_entropy_weighted_model_{dec_type}_decoder.mat'),
        {"decoded_entropy": decoded_entropy}, do_compression=True)
# savemat(os.path.join(path, f'{timestamp}_decoded_entropy_weighted_model_{dec_type}_decoder_pt_1.mat'),
#         {"decoded_entropy_1": decoded_entropy[:15]}, do_compression=True)
# savemat(os.path.join(path, f'{timestamp}_decoded_entropy_weighted_model_{dec_type}_decoder_pt_2.mat'),
#         {"decoded_entropy_2": decoded_entropy[15:]}, do_compression=True)
