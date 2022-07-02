# used to sav a mat file in parts due to scipy limit
import pickle
import lzma
from scipy.io import savemat
import os

path = "/home/lar/Documents/repos/NR_telemetry_ECC/runs/HC_eilat_July_2018/results/2022-07-01_22_36_52"
timestamp = path.split("/")[-1]
dec_type = "BP"

# load summary
# 2022-07-01_22_36_52_summary_classifying_entropy__BP_decoder
with open(os.path.join(path, f'{timestamp}_summary_classifying_entropy__{dec_type}_decoder.pickle'),
          'rb') as f:
    summary = pickle.load(f)

# load results
# 2022-07-01_22_36_52_simulation_classifying_entropy_BP_decoder.xz
with lzma.open(os.path.join(path, f'{timestamp}_simulation_classifying_entropy_{dec_type}_decoder.xz'), "rb") as f:
    results = pickle.load(f)

for step in results:
    step['data'] = step['data'].to_dict("list")
decoded_entropy = [step.pop('decoded_entropy').to_dict("list") for step in results]
decoded_ldpc = [step.pop('decoded_ldpc').to_dict("list") for step in results]

summary.update({"stats": results})

# savemat(os.path.join(path, f'{timestamp}_simulation_classifying_entropy_{args.dec_type}_decoder.mat'),
#             summary, do_compression=True)
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
