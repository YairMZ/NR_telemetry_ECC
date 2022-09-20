clear
clc
close all
%% 14-th
load('DecList-hc-14-2-2022.mat')
hc_feb_14 = DecList;
hc_feb_14_tx = [];
hc_feb_14_rx = [];
for idx = 1:length(hc_feb_14)
    if length(hc_feb_14(idx).Rawbits) == 4098 % only interseted in telemetry
        hc_feb_14(idx).coded_bin = hc_feb_14(idx).Rawbits < 0;
        hc_feb_14(idx).packet_id = str2num(hc_feb_14(idx).filename(11:14));
        hc_feb_14(idx).time_stamp = datetime(hc_feb_14(idx).filename(16:end-4),'InputFormat','yy-MM-dd-HH-mm-ss');
%         datetime(my_string,'InputFormat','yyyy-MM-dd HH:mm:ss.S')
        if contains(hc_feb_14(idx).filename, 'rx')
            hc_feb_14_rx = [hc_feb_14_rx; hc_feb_14(idx)];
        elseif contains(hc_feb_14(idx).filename, 'tx')
            hc_feb_14_tx = [hc_feb_14_tx; hc_feb_14(idx)];
        else
            disp(hc_feb_14(idx).filename);
        end
    end
end


load('DecList-ship-14-2-2022.mat')
ship_feb_14 = DecList;
ship_feb_14_tx = [];
ship_feb_14_rx = [];
for idx = 1:length(ship_feb_14)
    if length(ship_feb_14(idx).Rawbits) == 4098 % only interseted in telemetry
        ship_feb_14(idx).coded_bin = ship_feb_14(idx).Rawbits < 0;
        ship_feb_14(idx).packet_id = str2num(ship_feb_14(idx).filename(11:14));
        ship_feb_14(idx).time_stamp = datetime(ship_feb_14(idx).filename(16:end-4),'InputFormat','yy-MM-dd-HH-mm-ss');
        if contains(ship_feb_14(idx).filename, 'rx')
            %only single channel (transduceer) was used, retain only first rx
            if  length(ship_feb_14_rx)==0 || ~strcmp(ship_feb_14(idx).filename,ship_feb_14_rx(end).filename)
                ship_feb_14_rx = [ship_feb_14_rx; ship_feb_14(idx)];
            end
        elseif contains(ship_feb_14(idx).filename, 'tx')
            ship_feb_14_tx = [ship_feb_14_tx; ship_feb_14(idx)];
        else
            disp(ship_feb_14(idx).filename);
        end
    end
end
clear DecList hc_feb_14 idx ship_feb_14
clear hc_feb_14_rx ship_feb_14_tx
%% 16-th
load('DecList-hc-16-2-2022.mat')
hc_feb_16 = DecList;
hc_feb_16_tx = [];
hc_feb_16_rx = [];
for idx = 1:length(hc_feb_16)
    if length(hc_feb_16(idx).Rawbits) == 4098 % only interseted in telemetry
        hc_feb_16(idx).coded_bin = hc_feb_16(idx).Rawbits < 0;
        hc_feb_16(idx).packet_id = str2num(hc_feb_16(idx).filename(11:14));
        hc_feb_16(idx).time_stamp = datetime(hc_feb_16(idx).filename(16:end-4),'InputFormat','yy-MM-dd-HH-mm-ss');
        if contains(hc_feb_16(idx).filename, 'rx')
            hc_feb_16_rx = [hc_feb_16_rx; hc_feb_16(idx)];
        elseif contains(hc_feb_16(idx).filename, 'tx')
            hc_feb_16_tx = [hc_feb_16_tx; hc_feb_16(idx)];
        else
            disp(hc_feb_16(idx).filename);
        end
    end
end


load('DecList-ship-16-2-2022.mat')
ship_feb_16 = DecList;
ship_feb_16_tx = [];
ship_feb_16_rx = [];
for idx = 1:length(ship_feb_16)
    if length(ship_feb_16(idx).Rawbits) == 4098 % only interseted in telemetry
        ship_feb_16(idx).coded_bin = ship_feb_16(idx).Rawbits < 0;
        ship_feb_16(idx).packet_id = str2num(ship_feb_16(idx).filename(11:14));
        ship_feb_16(idx).time_stamp = datetime(ship_feb_16(idx).filename(16:end-4),'InputFormat','yy-MM-dd-HH-mm-ss');
        if contains(ship_feb_16(idx).filename, 'rx')
            ship_feb_16_rx = [ship_feb_16_rx; ship_feb_16(idx)];
        elseif contains(ship_feb_16(idx).filename, 'tx')
            ship_feb_16_tx = [ship_feb_16_tx; ship_feb_16(idx)];
        else
            disp(ship_feb_16(idx).filename);
        end
    end
end

clear DecList hc_feb_16 idx ship_feb_16
clear hc_feb_16_rx ship_feb_16_tx

%% sorting
T = struct2table(ship_feb_14_rx);
sortedT = sortrows(T, 'time_stamp');
ship_feb_14_rx = table2struct(sortedT);

T = struct2table(ship_feb_16_rx);
sortedT = sortrows(T, 'time_stamp');
ship_feb_16_rx = table2struct(sortedT);

T = struct2table(hc_feb_14_tx);
sortedT = sortrows(T, 'time_stamp');
hc_feb_14_tx = table2struct(sortedT);

T = struct2table(hc_feb_16_tx);
sortedT = sortrows(T, 'time_stamp');
hc_feb_16_tx = table2struct(sortedT);
clear T sortedT
%% timestamps and mission idx
t = [ship_feb_14_rx.time_stamp];
dt = diff(t);
mask = logical([1, dt > minutes(5)]);
% figure
% plot(t,'-o')
% hold on
% plot(find(mask),t(mask),'*','MarkerSize',12)
% title('ship feb 14 rx')
% set(gca,'FontSize',16)
% saveas(gcf,'exp_figures/ship_feb_14_rx_times.eps','epsc')

mission = 0;
for idx=1:length(mask)
    if mask(idx) == 1
        mission = mission + 1;
    end
    ship_feb_14_rx(idx).mission_idx = mission;
end
rx_per_mission_feb_14 = hist([ship_feb_14_rx.mission_idx],1:mission)

t = [ship_feb_16_rx.time_stamp];
dt = diff(t);
mask = logical([1, dt > minutes(5)]);
% figure
% plot(t,'-o')
% hold on
% plot(find(mask),t(mask),'*','MarkerSize',12)
% title('ship feb 16 rx')
% set(gca,'FontSize',16)
% saveas(gcf,'exp_figures/ship_feb_16_rx_times.eps','epsc')

mission = 0;
for idx=1:length(mask)
    if mask(idx) == 1
        mission = mission + 1;
    end
    ship_feb_16_rx(idx).mission_idx = mission;
end

rx_per_mission_feb_16 = hist([ship_feb_16_rx.mission_idx],1:mission)

t = [hc_feb_14_tx.time_stamp];
dt = diff(t);
mask = logical([1, dt > minutes(5)]);
% figure
% plot(t,'-o')
% hold on
% plot(find(mask),t(mask),'*','MarkerSize',12)
% title('HC feb 14 tx')
% set(gca,'FontSize',16)
% saveas(gcf,'exp_figures/HC_feb_14_tx_times.eps','epsc')

mission = 0;
for idx=1:length(mask)
    if mask(idx) == 1
        mission = mission + 1;
    end
    hc_feb_14_tx(idx).mission_idx = mission;
end
tx_per_mission_feb_14 = hist([hc_feb_14_tx.mission_idx],1:mission)

t = [hc_feb_16_tx.time_stamp];
dt = diff(t);
mask = logical([1, dt > minutes(5)]);
% figure
% plot(t,'-o')
% hold on
% plot(find(mask),t(mask),'*','MarkerSize',12)
% title('HC feb 16 tx')
% set(gca,'FontSize',16)
% saveas(gcf,'exp_figures/HC_feb_16_tx_times.eps','epsc')

mission = 0;
for idx=1:length(mask)
    if mask(idx) == 1
        mission = mission + 1;
    end
    hc_feb_16_tx(idx).mission_idx = mission;
end

tx_per_mission_feb_16 = hist([hc_feb_16_tx.mission_idx],1:mission)

clear t dt idx mask mission
%% adjust raw ber, find good tx

% load code
load('4098.mat')
[rH, cH] = find(Hnonsys);
HInd = [rH, cH];
hDec = comm.LDPCDecoder(HInd, 'OutputValue', 'Whole codeword','FinalParityChecksOutputPort',true,'NumIterationsOutputPort',true,...
    'IterationTerminationCondition','Parity check satisfied','MaximumIterationCount',130);

rx_idx = 0;
ship_feb_14_rx(1).rx_idx = 0;
ship_feb_14_rx(1).input_ber = ship_feb_14_rx(1).RawBER;
ship_feb_14_rx(1).success = ship_feb_14_rx(1).RawBER ~= -1;
for idx = 2:length(ship_feb_14_rx)
    if ship_feb_14_rx(idx).time_stamp > ship_feb_14_rx(idx-1).time_stamp
        rx_idx = rx_idx + 1;
    end
    ship_feb_14_rx(idx).rx_idx = rx_idx;
    ship_feb_14_rx(idx).input_ber = ship_feb_14_rx(idx).RawBER;
    ship_feb_14_rx(idx).success = ship_feb_14_rx(idx).RawBER ~= -1;
end


rx_idx = 0;
ship_feb_16_rx(1).rx_idx = 0;
ship_feb_16_rx(1).input_ber = ship_feb_16_rx(1).RawBER;
ship_feb_16_rx(1).success = ship_feb_16_rx(1).RawBER ~= -1;
for idx = 2:length(ship_feb_16_rx)
    if ship_feb_16_rx(idx).time_stamp > ship_feb_16_rx(idx-1).time_stamp
        rx_idx = rx_idx + 1;
    end
    ship_feb_16_rx(idx).rx_idx = rx_idx;
    ship_feb_16_rx(idx).input_ber = ship_feb_16_rx(idx).RawBER;
    ship_feb_16_rx(idx).success = ship_feb_16_rx(idx).RawBER ~= -1;
end


for idx = 1:length(ship_feb_16_rx)
    if ship_feb_16_rx(idx).input_ber == -1
        rx_idx = ship_feb_16_rx(idx).rx_idx;
        mask = [ship_feb_16_rx.rx_idx] == rx_idx;
        input_ber_vec = [ship_feb_16_rx(mask).RawBER];
        % was there any successful decoding?
        if max(input_ber_vec) > 0
            [val, t] = min(abs(input_ber_vec));
            t2 = find(mask,t);
            t2 = t2(end);
            raw_bits = double(ship_feb_16_rx(t2).Rawbits.');
            [DecBits, num_iter, parity] = hDec(raw_bits);
            info_bits = DecBits(1:984);
            if sum(parity) ~= 0
                disp('problem')
            end
            test = double(ship_feb_16_rx(t2).Decbits.');
            if max(abs(double(info_bits)-test)) > 0
                disp('problem')
            end         
            ship_feb_16_rx(idx).input_ber = sum(ship_feb_16_rx(idx).coded_bin.' ~= DecBits)/4098;
            ship_feb_16_rx(idx).tx_bits = DecBits.';
        end
    else
        raw_bits = double(ship_feb_16_rx(idx).Rawbits.');
        [DecBits, num_iter, parity] = hDec(raw_bits);
        info_bits = DecBits(1:984);
        if sum(parity) ~= 0
            disp('problem')
        end
        test = double(ship_feb_16_rx(idx).Decbits.');
        if max(abs(double(info_bits)-test)) > 0
            disp('problem')
        end
        if ship_feb_16_rx(idx).input_ber ~= single(sum(ship_feb_16_rx(idx).coded_bin.' ~= DecBits)/4098)
            disp('problem')
        end
        ship_feb_16_rx(idx).tx_bits = DecBits.';
    end
end

for idx = 1:length(ship_feb_14_rx)
    if ship_feb_14_rx(idx).input_ber ~= -1
        raw_bits = double(ship_feb_14_rx(idx).Rawbits.');
        [DecBits, num_iter, parity] = hDec(raw_bits);
        info_bits = DecBits(1:984);
        if sum(parity) ~= 0
            disp('problem')
        end
        test = double(ship_feb_14_rx(idx).Decbits.');
        if max(abs(double(info_bits)-test)) > 0
            disp('problem')
        end
        if ship_feb_14_rx(idx).input_ber ~= single(sum(ship_feb_14_rx(idx).coded_bin.' ~= DecBits)/4098)
            disp('problem')
        end
        ship_feb_14_rx(idx).tx_bits = DecBits.';
    end
end

clear rH cH HInd hDec H Hnonsys idx mask input_ber_vec t t2 val raw_bits DecBits info_bits num_iter ...
    parity test rx_idx

%% try finding more tx feb 14
% find missing tx
missing_tx = [];
figure
for idx = 1:length(ship_feb_14_rx)
    if isempty(ship_feb_14_rx(idx).tx_bits)
        missing = idx;
        above = 0;
        below = 0;
        for shift = 1:idx-1  % find good index above
            if ~isempty(ship_feb_14_rx(idx-shift).tx_bits)
                above = idx-shift;
                break;
            end
        end
        for shift = 1:478-idx  % find good index below
            if ~isempty(ship_feb_14_rx(idx+shift).tx_bits)
                below = idx+shift;
                break;
            end
        end
        if above > 0 && below > 0
            delta = below-above;
        else
            delta = 0;
        end
        missing_tx = [missing_tx; missing, above, below, delta];
    end
end


for idx = 1:size(missing_tx,1)
    %find above tx
    above = 0;
    if missing_tx(idx,2) > 0
        for ii =1:length(hc_feb_14_tx)
            if isequal(hc_feb_14_tx(ii).coded_bin, ship_feb_14_rx(missing_tx(idx,2)).tx_bits)
                above = ii;
                break
            end
        end
    else % what if no good rx at top?
        above = 1;
    end
    %find below tx
    below = 0;
    if missing_tx(idx,3) > 0
        for ii = above+1:length(hc_feb_14_tx)
            if isequal(hc_feb_14_tx(ii).coded_bin, ship_feb_14_rx(missing_tx(idx,3)).tx_bits)
                below = ii;
                break
            end
        end
    else % what if no good rx at bottom?
        below = length(hc_feb_14_tx);
    end
    if above >0 && below > 0
        delta = below - above;
    else
        delta = 0;
    end
    if delta > 0 % found span of buffrs to look within
        rx = ship_feb_14_rx(missing_tx(idx,1)).coded_bin;
        tx = [];
        for ii = above:below
            tx = [tx, hc_feb_14_tx(ii).coded_bin];
        end
        [r,lags] = xcorr(tx,rx);
%         plot(lags,r)
        [m,i] = max(r);
        delay = lags(i)/4098; % delay in buffers
        if abs(delay - floor(delay)) > 0
            plot(lags,r)      
            disp('problem')
            idx
            missing_tx(idx,1)
            delta
            above
            below
            delay = -1;
        else % if found good candidate
            % assign ship_feb_14_rx(missing_tx(idx,1)).tx_bits
            ship_feb_14_rx(missing_tx(idx,1)).tx_bits = hc_feb_14_tx(above+delay).coded_bin;
            ship_feb_14_rx(missing_tx(idx,1)).input_ber = ...
                sum(ship_feb_14_rx(missing_tx(idx,1)).coded_bin ~= ship_feb_14_rx(missing_tx(idx,1)).tx_bits)/4098;
        end
    else
        disp('no delta')
        idx
    end
end
tx = [hc_feb_14_tx.coded_bin];
missing_14 = 0;
for idx = 1:length(ship_feb_14_rx)
    if isempty(ship_feb_14_rx(idx).tx_bits)
        buff = ship_feb_14_rx(idx).coded_bin;
        [r,lags] = xcorr(tx,buff);
        plot(lags,r)
        [m,i] = max(r);
        delay = lags(i)/4098;
        missing_14 = missing_14 +1;
    end
end

clear missing_tx idx buff r lags m i delay missing below above delta shift ii rx tx
%% try finding more tx feb 16
% find missing tx
missing_tx = [];
for idx = 1:length(ship_feb_16_rx)
    if isempty(ship_feb_16_rx(idx).tx_bits)
        missing = idx;
        above = 0;
        below = 0;
        for shift = 1:idx-1  % find good index above
            if ~isempty(ship_feb_16_rx(idx-shift).tx_bits)
                above = idx-shift;
                break;
            end
        end
        for shift = 1:2771-idx  % find good index above
            if ~isempty(ship_feb_16_rx(idx+shift).tx_bits)
                below = idx+shift;
                break;
            end
        end
        if above > 0 && below > 0
            delta = below-above;
        else
            delta = 0;
        end
        missing_tx = [missing_tx; missing, above, below, delta];
    end
end


for idx = 1:size(missing_tx,1)
    %find above tx
    above = 0;
    if missing_tx(idx,2) > 0
        for ii =flip(1:length(hc_feb_16_tx))
            if isequal(hc_feb_16_tx(ii).coded_bin, ship_feb_16_rx(missing_tx(idx,2)).tx_bits)
                above = ii;
                break
            end
        end
    end
    %find below tx
    below = 0;
    if missing_tx(idx,3) > 0
        for ii = above+1:length(hc_feb_16_tx)
            if isequal(hc_feb_16_tx(ii).coded_bin, ship_feb_16_rx(missing_tx(idx,3)).tx_bits)
                below = ii;
                break
            end
        end
    end
    if above >0 && below > 0
        delta = below - above;
    else
        delta = 0;
    end
    if delta > 0 % found span of buffrs to look within
        rx = ship_feb_16_rx(missing_tx(idx,1)).coded_bin;
        tx = [];
        for ii = above:below
            tx = [tx, hc_feb_16_tx(ii).coded_bin];
        end
        [r,lags] = xcorr(tx,rx);
%         plot(lags,r)
        [m,i] = max(r);
        delay = lags(i)/4098; % delay in buffers
        if abs(delay - floor(delay)) > 0
            plot(lags,r)      
            disp('problem')
            idx
            missing_tx(idx,1)
            delta
            above
            below
            delay = -1;
        else % if found good candidate
            % assign ship_feb_16_rx(missing_tx(idx,1)).tx_bits
            ship_feb_16_rx(missing_tx(idx,1)).tx_bits = hc_feb_16_tx(above+delay).coded_bin;
            ship_feb_16_rx(missing_tx(idx,1)).input_ber = ...
                sum(ship_feb_16_rx(missing_tx(idx,1)).coded_bin ~= ship_feb_16_rx(missing_tx(idx,1)).tx_bits)/4098;
        end
    else
        disp('no delta')
        idx
    end
end
tx = [hc_feb_16_tx.coded_bin];
missing_16 = 0;
for idx = 1:length(ship_feb_16_rx)
    if isempty(ship_feb_16_rx(idx).tx_bits)
        buff = ship_feb_16_rx(idx).coded_bin;
        [r,lags] = xcorr(tx,buff);
        plot(lags,r)
        [m,i] = max(r);
        delay = lags(i)/4098;
        missing_16 = missing_16 +1;
    end
end

clear missing_tx idx buff r lags m i delay missing below above delta shift ii rx tx

%% stats
feb_14_raw_ber = double([ship_feb_14_rx.input_ber]);
feb_14_succsess_rate = sum([ship_feb_14_rx.success])/length(ship_feb_14_rx)

feb_16_raw_ber = double([ship_feb_16_rx.input_ber]);
feb_16_succsess_rate = sum([ship_feb_16_rx.success])/length(ship_feb_16_rx)

overall_success_rate = (sum([ship_feb_16_rx.success]) + sum([ship_feb_14_rx.success]))/(length(ship_feb_14_rx) + length(ship_feb_16_rx))
successeful_decodeing_ber = [feb_14_raw_ber([ship_feb_14_rx.success]==1), feb_16_raw_ber([ship_feb_16_rx.success]==1)];
unsuccesseful_decodeing_ber = [feb_14_raw_ber([ship_feb_14_rx.success]==0), feb_16_raw_ber([ship_feb_16_rx.success]==0)];
% omit bad data
unsuccesseful_decodeing_ber(unsuccesseful_decodeing_ber==-1) = [];

figure
h1 = histogram(successeful_decodeing_ber);
h1.BinWidth = 0.02;
hold on
h2 = histogram(unsuccesseful_decodeing_ber);
h2.BinWidth = 0.02;
xlabel('input ber')
legend('successful decodeing','unsuccessful decodeing')
%% success rate vs ber
bin_width = 0.02;
max_input_ber = max([feb_14_raw_ber feb_16_raw_ber]);
n_bins = ceil(max_input_ber/bin_width);
bin_edges = 0:bin_width:max_input_ber+bin_width;

feb_14_succsess = [ship_feb_14_rx.success];
feb_16_succsess = [ship_feb_16_rx.success];

success_rate_per_input_ber = zeros(1,n_bins);
attempts_per_input_ber = zeros(1,n_bins);

for bin_idx = 1:length(bin_edges)-1
    mask = (feb_14_raw_ber >= bin_edges(bin_idx)) & (feb_14_raw_ber < bin_edges(bin_idx+1));
    success = feb_14_succsess(mask);
    mask = (feb_16_raw_ber >= bin_edges(bin_idx)) & (feb_16_raw_ber < bin_edges(bin_idx+1));
    success = [success feb_16_succsess(mask)];
    attempts_per_input_ber(bin_idx) = length(success);
    success_rate_per_input_ber(bin_idx) = sum(success)/length(success);
end
figure;
histogram('BinEdges',bin_edges,'BinCounts',attempts_per_input_ber)
xlabel('imput ber')
ylabel('attempts')

figure;
plot(bin_edges(1:end-1)+bin_width/2,success_rate_per_input_ber,'*')
xlabel('imput ber')
ylabel('success rate')

clear success_rate max_input_ber n_bins bin_edges feb_14_succsess feb_16_succsess bin_idx mask success
%% llr
feb_16_llr = double(reshape([ship_feb_16_rx.Rawbits],4098,[]).');
% writematrix(feb_16_llr, 'feb_16_llr.csv');
feb_14_llr = double(reshape([ship_feb_14_rx.Rawbits],4098,[]).');
% writematrix(feb_14_llr, 'feb_14_llr.csv');

%% encoded
feb_14_tx = 2*ones(length(ship_feb_14_rx),4098);
for ii = 1:length(ship_feb_14_rx)
    if ~isempty(ship_feb_14_rx(ii).tx_bits)
        feb_14_tx(ii,:) = ship_feb_14_rx(ii).tx_bits;
    end
end
% writematrix(feb_14_tx, 'feb_14_tx.csv');

feb_16_tx = 2*ones(length(ship_feb_16_rx),4098);
for ii = 1:length(ship_feb_16_rx)
    if ~isempty(ship_feb_16_rx(ii).tx_bits)
        feb_16_tx(ii,:) = ship_feb_16_rx(ii).tx_bits;
    end
end
% writematrix(feb_16_tx, 'feb_16_tx.csv');
clear ii

%% save mat file
save('parsed_data.mat')