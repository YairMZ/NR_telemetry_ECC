import glob
import os
from datetime import datetime


def break_down(buffers: list[bytes]):
    res = []
    for buffer in buffers:
        valid = buffer.find(b'time: ')
        if valid >= 0:
            t = buffer.split(b'data success: ')
            t = [t[1][:3], t[1][3:]]
            # [buffer.find(b'data success: ')+len('data success: ')+3:]
            d = {
                'r_time': datetime.strptime(buffer[buffer.find(b'time: ') + 5:buffer.find(b'\r\n')].decode('UTF-8'),
                                            ' %d_%m_%Y_%H_%M_%S'),
                'size': int(buffer[buffer.find(b'data size: ') + 11:buffer.find(b'data success')].decode('UTF-8')),
                'success': int(t[0][:1].decode('UTF-8')),
                'payload': t[1][:123]
            }
            res.append(d)
    return res


folder_name = "ship_logs"  # "/Users/yairmazal/Downloads/hc_data/to_parse"
rx_files = [f for f in glob.glob(f'{folder_name}/**/*Rafael_rx.log', recursive=True) if os.path.getsize(f) > 0]

rx_files.sort(key=lambda x: datetime.strptime(x.split('/')[1], '%d_%m_%Y_%H_%M_%S'))
acquisitions = []
for f_name in rx_files:
    with open(f_name, 'rb') as f:
        log_file = f.read()
    temp_acquisitions = log_file.split(b'Aquisition ')

    acquisitions += break_down(temp_acquisitions)

acquisitions_14 = [a for a in acquisitions if a['r_time'].day == 14]
acquisitions_15 = [a for a in acquisitions if a['r_time'].day == 15]
acquisitions_16 = [a for a in acquisitions if a['r_time'].day == 16]
