import pickle
from bitstring import Bits
from ldpc.encoder import EncoderWiFi
import random


def encode_data(encoder: EncoderWiFi) -> list[Bits]:
    with open('data/hc_to_ship.pickle', 'rb') as f:
        hc_tx = pickle.load(f)

    buffers = [Bits(auto=tx.get("bin")) for tx in hc_tx.get("50000")]
    encoded = []
    for binary_data in buffers:
        if len(binary_data) <= encoder.k:
            pad_len = encoder.k - len(binary_data)
            padded = binary_data + Bits(uint=random.getrandbits(pad_len), length=pad_len)
            encoded.append(encoder.encode(padded))
        else:  # need to split buffer to several codewords
            for i in range(len(binary_data)//encoder.k):
                partial = binary_data[i*encoder.k: (i+1)*encoder.k]
                encoded.append(encoder.encode(partial))

            if (i+1)*encoder.k < len(binary_data):  # remaining tail to encode
                pad_len = encoder.k - len(binary_data[(i+1)*encoder.k:])
                padded = binary_data[(i+1)*encoder.k:] + Bits(uint=random.getrandbits(pad_len), length=pad_len)
                encoded.append(encoder.encode(padded))
    return encoded


if __name__ == '__main__':
    from ldpc.wifi_spec_codes import WiFiSpecCode
    enc = EncoderWiFi(WiFiSpecCode.N648_R12)
    data = encode_data(enc)
    pass
