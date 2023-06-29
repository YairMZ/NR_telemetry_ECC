from inference import BufferSegmentation, BufferModel
from protocol_meta import dialect_meta as meta
import numpy as np

tx = np.genfromtxt('data/feb_14_tx.csv', dtype=np.uint8, delimiter=',')
temp = np.genfromtxt('data/feb_16_tx.csv', dtype=np.uint8, delimiter=',')
tx = np.vstack((tx, temp))
bs = BufferSegmentation(meta.protocol_parser)
buffer_model = BufferModel(window_size=None)
parts, validity, structure = bs.segment_buffer(np.packbits(tx[0, :984]).tobytes())  # segment first buffer to get structure
bad_idx = []
for tx_idx in range(tx.shape[0]):
    if tx[tx_idx, 0] < 2:  # tx identified
        # bad idx 400,402,476, 1765,1767,1768,1769, 2709,2710,2711,2712,2713,2714
        if np.packbits(tx[tx_idx, :8])[0] != 254:
            bad_idx.append(tx_idx)
            continue
        buffer_model.add_buffer(tx[tx_idx, :984], structure)
buffer_model.save('data/model_2022.json')
