%% dvbs2 - 6073B
h = dvbs2ldpc(3/4);
k = size(h, 2) - size(h, 1);
n = size(h, 2);
nB = k/8-2;

%% dvbs2 - 4048B
h = dvbs2ldpc(1/2);
k = size(h, 2) - size(h, 1);
n = size(h, 2);
nB = k/8-2;

%% dvbs2 - 2023B
h = dvbs2ldpc(1/4);
k = size(h, 2) - size(h, 1);
n = size(h, 2);
nB = k/8-2;

%% custom code
load('4098.mat')
k = size(H, 2) - size(H, 1);
n = size(H, 2);
info_bits = randi([0 1],k,1);

%encoding:
[rH, cH] = find(H);
EncHInd = [rH, cH];
hEnc = comm.LDPCEncoder(EncHInd);
EncBits = step(hEnc, info_bits(1: k));

% decoding:
[rH, cH] = find(Hnonsys);
DecHInd = single([rH, cH]);
hDec = comm.LDPCDecoder(DecHInd);
DecBits = step(hDec, EncBits(1: n));