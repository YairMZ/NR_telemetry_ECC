"""unit tests for the EntropyBitwiseDecoder class"""
import pytest
import numpy as np
import bitstring
from utils.custom_exceptions import IncorrectBufferLength
from decoders import EntropyBitwiseDecoder, EntropyBitwiseFlippingDecoder
from ldpc.decoder import bsc_llr, DecoderWiFi
from ldpc.wifi_spec_codes import WiFiSpecCode
import pickle


class TestEntropyBitwiseFlippingDecoder:
    def test_wrong_buffer_length(self) -> None:
        entropy_decoder = EntropyBitwiseDecoder(DecoderWiFi(spec=WiFiSpecCode.N1944_R23, max_iter=10),
                                                model_length=10, entropy_threshold=1, clipping_factor=1, min_data=1)
        buffer = np.unpackbits(np.array(list(range(10)), dtype=np.uint8))
        with pytest.raises(Exception):
            entropy_decoder.decode_buffer(buffer)  # type: ignore

    def test_wrong_model_length(self) -> None:
        entropy_decoder = EntropyBitwiseDecoder(DecoderWiFi(spec=WiFiSpecCode.N1944_R23, max_iter=10),
                                                model_length=10, entropy_threshold=1, clipping_factor=1, min_data=1)
        with pytest.raises(IncorrectBufferLength):
            entropy_decoder.update_model(np.array([1]))

    def test_successful_decode(self) -> None:
        with open("tests/test_data/entropy_bitwise_decoder_successful_decode.pickle", 'rb') as f:
            d = pickle.load(f)
        encoded = d['encoded']
        corrupted = d['corrupted']
        channel = bsc_llr(p=0.02)
        channel_llr = channel(np.array(corrupted, dtype=np.int_))
        decoder = EntropyBitwiseDecoder(DecoderWiFi(spec=WiFiSpecCode.N1944_R23, max_iter=20, ),
                                        model_length=1224, entropy_threshold=0.36,
                                        clipping_factor=2, min_data=1)
        d = decoder.decode_buffer(channel_llr)
        assert bitstring.Bits(d[0]) == encoded

    def test_unsuccessful_decode(self) -> None:
        with open("tests/test_data/entropy_bitwise_decoder_unsuccessful_decode.pickle", 'rb') as f:
            d = pickle.load(f)
        corrupted = d['corrupted']
        channel = bsc_llr(p=0.1)
        channel_llr = channel(np.array(corrupted, dtype=np.int_))
        decoder = EntropyBitwiseDecoder(DecoderWiFi(spec=WiFiSpecCode.N1944_R23, max_iter=20, ),
                                        model_length=1224, entropy_threshold=0.36,
                                        clipping_factor=2, min_data=1)
        d = decoder.decode_buffer(channel_llr)
        assert d[2] is False


class TestEntropyBitwiseDecoder:
    def test_wrong_buffer_length(self) -> None:
        entropy_decoder = EntropyBitwiseFlippingDecoder(DecoderWiFi(spec=WiFiSpecCode.N1944_R23, max_iter=10,
                                                                    channel_model=bsc_llr(p=0.1)),
                                                        model_length=10, entropy_threshold=1, min_data=1)
        buffer = np.unpackbits(np.array(list(range(10)), dtype=np.uint8))
        with pytest.raises(Exception):
            entropy_decoder.decode_buffer(buffer)  # type: ignore

    def test_wrong_model_length(self) -> None:
        entropy_decoder = EntropyBitwiseFlippingDecoder(DecoderWiFi(spec=WiFiSpecCode.N1944_R23, max_iter=10,
                                                                    channel_model=bsc_llr(p=0.1)),
                                                        model_length=10, entropy_threshold=1, min_data=1)
        with pytest.raises(IncorrectBufferLength):
            entropy_decoder.update_model(np.array([1]))

    def test_successful_decode(self) -> None:
        with open("tests/test_data/entropy_bitwise_decoder_successful_decode.pickle", 'rb') as f:
            d = pickle.load(f)
        encoded = d['encoded']
        corrupted = d['corrupted']
        channel = bsc_llr(p=0.02)
        channel_llr = channel(np.array(corrupted, dtype=np.int_))
        decoder = EntropyBitwiseFlippingDecoder(DecoderWiFi(spec=WiFiSpecCode.N1944_R23, max_iter=20,
                                                            channel_model=bsc_llr(p=0.1)),
                                                model_length=1224, entropy_threshold=0.36, min_data=1)
        d = decoder.decode_buffer(corrupted)
        assert bitstring.Bits(d[0]) == encoded

    def test_unsuccessful_decode(self) -> None:
        with open("tests/test_data/entropy_bitwise_decoder_unsuccessful_decode.pickle", 'rb') as f:
            d = pickle.load(f)
        corrupted = d['corrupted']
        channel = bsc_llr(p=0.1)
        channel_llr = channel(np.array(corrupted, dtype=np.int_))
        decoder = EntropyBitwiseFlippingDecoder(DecoderWiFi(spec=WiFiSpecCode.N1944_R23, max_iter=20,
                                                            channel_model=bsc_llr(p=0.1)),
                                                model_length=1224, entropy_threshold=0.36, min_data=1)
        d = decoder.decode_buffer(corrupted)
        assert d[2] is False