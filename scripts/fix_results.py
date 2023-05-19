# used to sav a mat file in parts due to scipy limit
import pickle
import lzma
from scipy.io import savemat
import os

path = "/Users/yairmazal/google_drive/My Drive/University/PhD/NR_telemetry_ECC/runs/HC_eilat_July_2018/results/AWGN/" \
       "/two_decoders/2023-05-16_21_47_18"
timestamp = path.split("/")[-1]
dec_type = "BP"

# load summary
# 2022-07-01_22_36_52_summary_classifying_entropy__BP_decoder
with open(os.path.join(path, f'{timestamp}_summary_classifying_entropy__{dec_type}_decoder.pickle'),
          'rb') as f:
    summary = pickle.load(f)
print("loaded summary")
# load results
# 2022-07-01_22_36_52_simulation_classifying_entropy_BP_decoder.xz
with lzma.open(os.path.join(path, f'{timestamp}_simulation_AWGN_classifying_entropy_{dec_type}_decoder.xz'), "rb") as f:
    results = pickle.load(f)
print("loaded results")
for step in results:
    step['data'] = step['data'].to_dict("list")
decoded_entropy = [step.pop('decoded_entropy').to_dict("list") for step in results]
decoded_ldpc = [step.pop('decoded_ldpc').to_dict("list") for step in results]

summary.update({"stats": results})

savemat(os.path.join(path, f'{timestamp}_data_{dec_type}_decoder.mat'),
        summary, do_compression=True)
print("saved data")
savemat(os.path.join(path, f'{timestamp}_decoded_ldpc_{dec_type}_decoder.mat'),
        {"decoded_ldpc": decoded_ldpc}, do_compression=True)
print("saved ldpc")
savemat(os.path.join(path, f'{timestamp}_decoded_entropy_{dec_type}_decoder.mat'),
        {"decoded_entropy": decoded_entropy}, do_compression=True)

# savemat(os.path.join(path, f'{timestamp}_decoded_entropy_{dec_type}_decoder_pt_1.mat'),
#         {"decoded_entropy_1": decoded_entropy[:20]}, do_compression=True)
# savemat(os.path.join(path, f'{timestamp}_decoded_entropy_{dec_type}_decoder_pt_2.mat'),
#         {"decoded_entropy_2": decoded_entropy[20:]}, do_compression=True)
print("saved entropy")