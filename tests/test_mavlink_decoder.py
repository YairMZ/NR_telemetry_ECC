"""unit tests for the MavlinkDecoder class"""
import bitstring
import numpy as np

from decoders import MavlinkRectifyingDecoder
from ldpc.decoder import DecoderWiFi
from ldpc.wifi_spec_codes import WiFiSpecCode
import pickle


class TestMavlinkRectifyingDecoder:
    def test_successful_decode(self) -> None:
        with open("tests/test_data/mavlink_decoder_successful_decode.pickle", 'rb') as f:
            d = pickle.load(f)
        encoded = d['encoded']
        channel_llr = d['channel_llr']
        decoder = MavlinkRectifyingDecoder(DecoderWiFi(spec=WiFiSpecCode.N1944_R23, max_iter=60),model_length=1224,
                                           threshold=0.01,n_clusters=1, valid_factor=1.1, invalid_factor=0.9, cluster=False)
        d = decoder.decode_buffer(channel_llr,np.array([]))
        assert bitstring.Bits(d[0]) == encoded
        assert d[2] is True

    def test_unsuccessful_decode(self) -> None:
        with open("tests/test_data/mavlink_decoder_unsuccessful_decode.pickle", 'rb') as f:
            d = pickle.load(f)
        encoded = d['encoded']
        channel_llr = d['channel_llr']
        decoder = MavlinkRectifyingDecoder(DecoderWiFi(spec=WiFiSpecCode.N1944_R23, max_iter=60),model_length=1224,
                                           threshold=0.01,n_clusters=1, valid_factor=1.1, invalid_factor=0.9, cluster=False)
        d = decoder.decode_buffer(channel_llr, np.array([]))
        assert d[2] is False
