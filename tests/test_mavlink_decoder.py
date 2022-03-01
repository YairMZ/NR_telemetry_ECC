"""unit tests for the MavlinkDecoder class"""
import bitstring
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
        decoder = MavlinkRectifyingDecoder(DecoderWiFi(spec=WiFiSpecCode.N1944_R23, max_iter=20),
                                           segmentation_iterations=2, ldpc_iterations=20, k=1296, bad_p=0.047, good_p=1e-7)
        d = decoder.decode_buffer(channel_llr)
        assert bitstring.Bits(d[0]) == encoded  # no hamming distance
        assert d[2] is True  # valid codeword
        assert d[4] == 5  # all five Mavlink messages were recovered

    def test_unsuccessful_decode(self) -> None:
        with open("tests/test_data/mavlink_decoder_unsuccessful_decode.pickle", 'rb') as f:
            d = pickle.load(f)
        encoded = d['encoded']
        channel_llr = d['channel_llr']
        decoder = MavlinkRectifyingDecoder(DecoderWiFi(spec=WiFiSpecCode.N1944_R23, max_iter=20),
                                           segmentation_iterations=2, ldpc_iterations=20, k=1296, bad_p=0.047, good_p=1e-7)
        d = decoder.decode_buffer(channel_llr)
        assert d[2] is False
