clear;
% load audio
audioFile = 'music-filename';
format = '.mp3';
[audio, fs] = audioread(['C:\path\to\workspace\' audioFile format]);
if size(audio, 2) > 1 
    audio = mean(audio, 2);         % Convert to mono
end
%%
% Short Time Fourier Transform
winSize =768-1;
oversampling = 4;

win = sqrt(hann(winSize,'periodic'));
overlapLen = floor(numel(win)-numel(win)/oversampling);
fftLen = 2^nextpow2(numel(win)+1);

[y,f,t] = stft(audio, fs, window=win, OverlapLength=overlapLen, FFTLength= fftLen, FrequencyRange='onesided');
ymag = abs(y);

figure;
imagesc(t, f, log10(ymag.^2));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
title('Base audio spectrogram');
%%
% Creating Percussive/Harmonic-enhanced masks
Freq_filterLen =550;            % Hz
Time_filterLen =0.2;            % s

% Calculate bins
Freq_medianLen = ceil(Freq_filterLen/(fs/fftLen));
Time_medianLen = ceil(Time_filterLen/((winSize-overlapLen)/fs));
% Round to upper odd integer
Freq_medianLen = Freq_medianLen + (mod(Freq_medianLen, 2) == 0);
Time_medianLen = Time_medianLen + (mod(Time_medianLen, 2) == 0);

P_enhanced = movmedian(ymag, Freq_medianLen, 1);
H_enhanced = movmedian(ymag, Time_medianLen, 2);

[~,P_elements] = size(P_enhanced);
[H_elements,~] = size(H_enhanced);
%%
% Plot percussive enhanced (P_enhanced)
P_idx =100; 

figure;
plot(f, log10(ymag(:,P_idx).^2), Color=[0.5 0.5 0.5], DisplayName='Original'); 
hold on;
plot(f, log10(P_enhanced(:, P_idx).^2), LineWidth=2, Color='r', DisplayName='P-Enhanced');
hold off;

title(['Frequency Slice at Frame ' num2str(P_idx)]);
xlabel('Frequency (Hz)');
ylabel('Log Magnitude');
legend;
grid on;
%%
% Plot Harmonic-enhanced (H_enhanced)
H_idx =100;

figure;
plot(t, log10(ymag(H_idx, :).^2), Color=[0.5 0.5 0.5], DisplayName='Original'); 
hold on;
plot(t, log10(H_enhanced(H_idx, :).^2), LineWidth=2, Color='b', DisplayName='H-Enhanced');
hold off;

title(['Time Slice at Bin ' num2str(H_idx) ' (' num2str(f(H_idx), '%.1f') ' Hz)']);
xlabel('time (s)');
ylabel('Log Magnitude');
legend;
grid on;
%%
% display specrographs
figure;
tiledlayout(2,1);

nexttile;
imagesc(t, f, log10(H_enhanced.^2));
axis xy; colorbar;
title('Harmonic Enhanced (Horizontal Lines Preserved)');
xlabel('Time (s)'); ylabel('Hz');

nexttile;
imagesc(t, f, log10(P_enhanced.^2));
axis xy; colorbar;
title('Percussive Enhanced (Vertical Spikes Preserved)');
xlabel('Time (s)'); ylabel('Hz');
%%
% Calculate masks 
% Soft mask with exponent
% gamma = 2.2;
% P_mask = (P_enhanced./(P_enhanced + H_enhanced + eps)).^gamma;
% H_mask = (H_enhanced./(P_enhanced + H_enhanced + eps)).^gamma;

% Standard Wiener
P_power = P_enhanced.^2;
H_power = H_enhanced.^2;

P_mask = P_power ./ (P_power + H_power + eps);
H_mask = H_power ./ (P_power + H_power + eps);
%%
% display specrographs
figure;
tiledlayout(2,1);

nexttile;
imagesc(t, f, log10(H_mask.^2));
axis xy; colorbar;
title('Harmonic mask');
xlabel('Time (s)'); ylabel('Hz');

nexttile;
imagesc(t, f, log10(P_mask.^2));
axis xy; colorbar;
title('Percussive mask');
xlabel('Time (s)'); ylabel('Hz')
%%
% Apply mask
y_drums = y .* P_mask;
y_harmonic = y .* H_mask;
% Rebuild audio
drums_isolated = istft(y_drums, fs, Window=win, OverlapLength=overlapLen, FFTLength=fftLen, FrequencyRange='onesided', ConjugateSymmetric=true);
Harmonics_isolated = istft(y_harmonic, fs, Window=win, OverlapLength=overlapLen, FFTLength=fftLen, FrequencyRange='onesided', ConjugateSymmetric=true);
% Normalize
drums_isolated = drums_isolated / (max(abs(drums_isolated)) + eps);
Harmonics_isolated = Harmonics_isolated / (max(abs(Harmonics_isolated)) + eps);
%%
% Write to audio file
% Create the directory safely
if ~exist(audioFile, 'dir')
    mkdir(audioFile);
end
% Save Voice
drums_path = fullfile(audioFile, 'drums_isolated.wav');
audiowrite(drums_path, drums_isolated, fs);
% Save Backing
Harmonics_path = fullfile(audioFile, 'Harmonics_isolated.wav');
audiowrite(Harmonics_path, Harmonics_isolated, fs);