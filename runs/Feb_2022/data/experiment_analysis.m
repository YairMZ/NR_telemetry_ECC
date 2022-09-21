entropy_success = decoded_entropy.decode_success;
successeful_entropy_decoding_input_ber = input_ber(entropy_success);
unsuccesseful_entropy_decoding_input_ber = input_ber(~entropy_success);
% omit bad data
unsuccesseful_entropy_decoding_input_ber(unsuccesseful_entropy_decoding_input_ber==-1) = [];

figure
h1 = histogram(successeful_entropy_decoding_input_ber);
h1.BinWidth = 0.02;
hold on
h2 = histogram(unsuccesseful_entropy_decoding_input_ber);
h2.BinWidth = 0.02;
xlabel('input ber')
legend('successful decodeing','unsuccessful decodeing')

clear h1 h2