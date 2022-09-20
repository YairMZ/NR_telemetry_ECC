%% baseline
close all
clear
set(0, 'defaultTextInterpreter', 'latex');
% set(0, 'defaultTextInterpreter', 'none');
%%
load('parsed_data.mat')

figure
h1 = histogram(successeful_decodeing_ber);
h1.BinWidth = 0.02;
hold on
h2 = histogram(unsuccesseful_decodeing_ber);
h2.BinWidth = 0.02;
xlabel('Input ber')
legend('successful decodeing','unsuccessful decodeing','Interpreter','none')
set(gca,'FontSize',16)
% saveas(gcf,'exp_figures/exp_success_hitograms','svg')


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
xlabel('Input ber')
ylabel('attempts')
set(gca,'FontSize',16)
% saveas(gcf,'exp_figures/exp_attmepts_hitograms','svg')

f = figure;
plot(bin_edges(1:end-1)+bin_width/2,success_rate_per_input_ber,'*')
xlabel('Input ber')
ylabel('Decoding success rate')
set(gca,'FontSize',16,'FontName','mwa_cmr10')
% saveas(gcf,'exp_figures/exp_success_rate','epsc')

clear success_rate max_input_ber n_bins bin_edges feb_14_succsess feb_16_succsess bin_idx mask success

%% my algorithm
uiopen('load')
bin_width = 0.02;
max_input_ber = max(input_ber);
n_bins = ceil(max_input_ber/bin_width);
bin_edges = 0:bin_width:max_input_ber+bin_width;

my_succsess = decoded_entropy.decode_success;

success_rate_per_input_ber = zeros(1,n_bins);
attempts_per_input_ber = zeros(1,n_bins);
for bin_idx = 1:length(bin_edges)-1
    mask = (input_ber >= bin_edges(bin_idx)) & (input_ber < bin_edges(bin_idx+1));
    success = my_succsess(mask);
    attempts_per_input_ber(bin_idx) = length(success);
    success_rate_per_input_ber(bin_idx) = sum(success)/length(success);
end

figure;
histogram('BinEdges',bin_edges,'BinCounts',attempts_per_input_ber)
xlabel('Input ber')
ylabel('attempts')
set(gca,'FontSize',16,'FontName','mwa_cmr10')


figure(f)
hold on 
plot(bin_edges(1:end-1)+bin_width/2,success_rate_per_input_ber,'o')
xlabel('Input ber')
ylabel('Decoding success rate')
legend('BP','BP+NR' )
set(gca,'FontSize',16,'FontName','mwa_cmr10')
