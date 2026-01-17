clear;
% load audio
audioFile = 'music-filename';
format = '.mp3';
[audio, fs] = audioread(['C:\path\\to\workspace\' audioFile format]);
if size(audio, 2) > 1 
    audio = mean(audio, 2);         % Convert to mono
end
%%
% Extract percussives
winSize =1024-1;
oversampling = 4;

win = sqrt(hann(winSize,'periodic'));
overlapLen = floor(numel(win)-numel(win)/oversampling);
fftLen = 2^nextpow2(numel(win)+1);

[y,f,t] = stft(audio, fs, window=win, OverlapLength=overlapLen, FFTLength= fftLen, FrequencyRange='onesided');
ymag = abs(y);

Freq_filterLen =200;            % Hz
Time_filterLen =0.05;            % s

Freq_medianLen = ceil(Freq_filterLen/(fs/fftLen));
Time_medianLen = ceil(Time_filterLen/((winSize-overlapLen)/fs));
Freq_medianLen = Freq_medianLen + (mod(Freq_medianLen, 2) == 0);
Time_medianLen = Time_medianLen + (mod(Time_medianLen, 2) == 0);

P_enhanced = movmedian(ymag, Freq_medianLen, 1);
H_enhanced = movmedian(ymag, Time_medianLen, 2);

P_power = P_enhanced.^2;
H_power = H_enhanced.^2;
P_mask = P_power ./ (P_power + H_power + eps);
H_mask = H_power ./ (P_power + H_power + eps);

y_percussive = y.* P_mask;
y_harmonic = y .* H_mask;
percussives = istft(y_percussive, fs, Window=win, OverlapLength=overlapLen, FFTLength=fftLen, FrequencyRange='onesided', ConjugateSymmetric=true);
percussiveless = istft(y_harmonic, fs, Window=win, OverlapLength=overlapLen, FFTLength=fftLen, FrequencyRange='onesided', ConjugateSymmetric=true);
%%
% Short Time Fourier Transform
winSize =8192-1;
oversampling = 8;

win = sqrt(hann(winSize,'periodic'));
overlapLen = floor(numel(win)-numel(win)/oversampling);
fftLen = 2^nextpow2(numel(win)+1);

[y,f,t] = stft(percussiveless, fs, window=win, OverlapLength=overlapLen, FFTLength= fftLen, FrequencyRange='onesided');
ymag = abs(y);

figure;
imagesc(t, f, log10(ymag.^2));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
title('audio spectrogram - percussives removed');
%%
% Creating Percussive/Harmonic-enhanced masks
Freq_filterLen =6;            % Hz
Time_filterLen =1;            % s

Freq_medianLen = ceil(Freq_filterLen/(fs/fftLen));
Time_medianLen = ceil(Time_filterLen/((winSize-overlapLen)/fs));
Freq_medianLen = Freq_medianLen + (mod(Freq_medianLen, 2) == 0);
Time_medianLen = Time_medianLen + (mod(Time_medianLen, 2) == 0);

P_enhanced = movmedian(ymag, Freq_medianLen, 1);
H_enhanced = movmedian(ymag, Time_medianLen, 2);

[~,P_elements] = size(P_enhanced);
[H_elements,~] = size(H_enhanced);
%%
% Plot percussive enhanced (P_enhanced)
P_idx =1805; 

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
H_idx =111;

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
xlabel('Time (s)'); ylabel('Hz');
%%
% reconstruct audio
y_percussive = y .* P_mask;
y_harmonic = y .* H_mask;
% Rebuild audio
voice = istft(y_percussive, fs, Window=win, OverlapLength=overlapLen, FFTLength=fftLen, FrequencyRange='onesided', ConjugateSymmetric=true);
backing = istft(y_harmonic, fs, Window=win, OverlapLength=overlapLen, FFTLength=fftLen, FrequencyRange='onesided', ConjugateSymmetric=true);
%%
% combine backing and percussives
% ensure lengths match before addition (ISTFT might introduce minor length differences)
min_len = min([length(backing), length(percussives), length(voice)]);
backing = backing(1:min_len);
percussives = percussives(1:min_len);
voice = voice(1:min_len);

% combine
backing = backing + percussives;
% Normalize
voice = voice / (max(abs(voice)) + eps);
backing = backing / (max(abs(backing)) + eps);
%%
% Write to audio file
% Create the directory safely
if ~exist(audioFile, 'dir')
    mkdir(audioFile);
end
% Save Percusives
Percussion_path = fullfile(audioFile, 'percussions.wav');
audiowrite(Percussion_path, percussives, fs);
% Save Voice
voicePath = fullfile(audioFile, 'voice_isolated.wav');
audiowrite(voicePath, voice, fs);
% Save Backing
backingPath = fullfile(audioFile, 'backing_only.wav');
audiowrite(backingPath, backing, fs);