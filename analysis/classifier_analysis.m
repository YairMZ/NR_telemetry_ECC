%% find easily correctable bits
close all
n_buffers = results{1,1}.number_of_buffers;
n_flips = length(results);
easily_correctable = zeros(n_flips,n_buffers);
figure
hold on
for p_idx = 1:n_flips
   no_errors = results{1,p_idx}.no_errors;
   for buffer_idx = 1:n_buffers
       % easily correctable errors stem from zero std fields which are classified as outliers
       % find the intersection of buffer errros and forced bits.
       correctable = length(intersect(results{1,p_idx}.errors(buffer_idx,:),results{1,p_idx}.classifier.forced_bits{1, buffer_idx}));
       easily_correctable(p_idx, buffer_idx) = correctable/double(no_errors);
   end
   histogram(100*easily_correctable(p_idx,:),'Normalization','probability')
   legend_txt{p_idx} = ['p=' num2str(results{1,p_idx}.p)];
end
legend(legend_txt)
xlabel('easily correctable errors [%]')
hold off