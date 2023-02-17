from inference import BufferSegmentation, BufferModel
from protocol_meta import dialect_meta as meta


# setup test data
good_buffer = b'\xfe\x13\x00\x01\x00\xd4\x92\x01K\x16+G\xc0D+\xf4\n?\x00\x00\x80?<\t\x01\x896\xfe\n\x01\x01\x00\xda' \
              b'\x92\x01K\x16Z\x0c>\x10\x01d\xd0\xf5\xfe,\x02\x02\x00!\xd1\xffJ\x16\x12?\x97\x11\xb3"\xd2\x14\x00\x00' \
              b'\xc8B\x00\x00\xc8B\x00\x00\xc8B\xef\xfa\x10@\xbe\x1b^>\xb1\xd4\xda>\x05\x9d\xb8=D\x0b}BZ\xb9\xfe\x0c' \
              b'\x03\x01\xc8\xea\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xb6r'
bad_buffer = good_buffer[:27] + bytes([1]) + good_buffer[28:]

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
