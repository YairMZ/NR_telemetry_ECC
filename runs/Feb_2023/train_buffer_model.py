from inference import BufferSegmentation, BufferModel
from protocol_meta import dialect_meta as meta
import numpy as np

msg_type = 'telemetry'
tx = np.genfromtxt(f'data/feb_17_tx_{msg_type}.csv', dtype=np.uint8, delimiter=',')
temp = np.genfromtxt(f'data/feb_18_tx_{msg_type}.csv', dtype=np.uint8, delimiter=',')
tx = np.vstack((tx, temp))
bs = BufferSegmentation(meta.protocol_parser)
buffer_model = BufferModel(window_size=None)
parts, validity, structure = bs.segment_buffer(np.packbits(tx[0, :984]).tobytes())  # segment first buffer to get structure
bad_idx = []
for tx_idx in range(tx.shape[0]):
    if tx[tx_idx, 0] < 2:  # tx identified
        if np.packbits(tx[tx_idx, :8])[0] != 254:
            bad_idx.append(tx_idx)
            continue
        buffer_model.add_buffer(tx[tx_idx, :984], structure)
buffer_model.save('data/model_2023.json')
