from inference import BufferSegmentation, BufferModel
from protocol_meta import dialect_meta as meta
import numpy as np
import pickle
from bitstring import Bits, BitArray


# setup test data
good_buffer = b'\xfe\x13\x00\x01\x00\xd4\x92\x01K\x16+G\xc0D+\xf4\n?\x00\x00\x80?<\t\x01\x896\xfe\n\x01\x01\x00\xda' \
              b'\x92\x01K\x16Z\x0c>\x10\x01d\xd0\xf5\xfe,\x02\x02\x00!\xd1\xffJ\x16\x12?\x97\x11\xb3"\xd2\x14\x00\x00' \
              b'\xc8B\x00\x00\xc8B\x00\x00\xc8B\xef\xfa\x10@\xbe\x1b^>\xb1\xd4\xda>\x05\x9d\xb8=D\x0b}BZ\xb9\xfe\x0c' \
              b'\x03\x01\xc8\xea\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xb6r'
bad_buffer = good_buffer[:20] + bytes([1]) + good_buffer[21:]

bs = BufferSegmentation(meta.protocol_parser)
msg_parts, bit_validity, buffer_structure = bs.segment_buffer(bad_buffer)
msg_parts, bit_validity, buffer_structure = bs.segment_buffer(good_buffer)
msg_fields, buffer_description = bs.get_msg_fields(good_buffer)
buffer_model = BufferModel()
for fieldname, vals in msg_fields.items():
    buffer_model.add_sample(fieldname, vals[0], vals[1])
buffer_model.set_buffer_description(buffer_description)
buffer_model.save('test_fields_collection.json')

# load from file
buffer_model2 = BufferModel.load('test_fields_collection.json')

# test prediction on good buffer with hard bit values, return prediction per field
bit_arr = np.unpackbits(np.frombuffer(good_buffer, dtype=np.uint8))
prediction = buffer_model2.predict(bit_arr, buffer_structure, False)
# test prediction on good buffer by sending a bytes object, return prediction per bit
prediction = buffer_model2.predict(good_buffer, buffer_structure)
# test prediction on bad buffer by sending a bytes object, return prediction per field
prediction = buffer_model2.predict(bad_buffer, buffer_structure, False)

# train models
window_size = 100
with open('../runs/HC_eilat_July_2018/data/hc_to_ship.pickle', 'rb') as f:
    hc_tx = pickle.load(f)
    hc_bin_data = [Bits(auto=tx.get("bin")) for tx in hc_tx.get("20000")]
bs = BufferSegmentation(meta.protocol_parser)
buffer_model = BufferModel(window_size=window_size)
for tx in hc_bin_data:
    parts, validity, structure = bs.segment_buffer(tx.tobytes())
    msg_fields, buffer_description = bs.get_msg_fields(tx.tobytes())
    for field_name, vals in msg_fields.items():
        buffer_model.add_sample(field_name, vals[0], vals[1])

# make prediction regarding good buffer
p_good = buffer_model.predict(np.array(hc_bin_data[-1], dtype=np.uint8), structure, False)
# corrupt some bits
# take last buffer
corrupt_tx = BitArray(hc_bin_data[-1])
no_errors = 5
# random error indices
rng = np.random.default_rng()
error_indices = np.sort(rng.choice(50*8, size=no_errors, replace=False))  # first 50 bytes, first msg (global_position_int)
corrupt_tx.invert(error_indices)
bit_arr = np.array(corrupt_tx, dtype=np.uint8)
# make prediction regarding corrupt buffer
p_bad = buffer_model.predict(bit_arr, structure, False)
# save model
buffer_model.save(f'model_2018_window_size_{window_size}.json')
# # find damaged fields based on error indices
damaged_fields = buffer_model.find_damaged_fields(error_indices, structure)
for idx in range(len(p_bad)):
    if p_bad[idx][1] != p_good[idx][1]:
        print(f'{p_bad[idx][0]}: delta={(p_good[idx][1] - p_bad[idx][1])/p_good[idx][1]}')