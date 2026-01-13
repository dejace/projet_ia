clear; clc;

inFile = "test3.wav";
[x, fs] = audioread(inFile);
if size(x,2) > 1, x = mean(x,2); end
x = x / max(abs(x) + 1e-12);

% --- STFT ---
winLen = 2048; hop = 512; nfft = 2048;
w = hann(winLen,"periodic");
[S,f,t] = stft(x, fs, "Window", w, "OverlapLength", winLen-hop, "FFTLength", nfft);
Mag = abs(S);

% --- HPSS (médian) ---
timeMedianSize = 31;
freqMedianSize = 31;
H = medfilt2(Mag, [1, timeMedianSize], "symmetric");   % harmonique
P = medfilt2(Mag, [freqMedianSize, 1], "symmetric");   % percussif

% ==========================================================
% 1) Gate temporel par attaques (sur percussif brut)
% ==========================================================
% Percussif brut (soft mask léger juste pour calculer le flux)
pFlux = 2; eps0 = 1e-12;
Mp_soft = (P.^pFlux) ./ (H.^pFlux + P.^pFlux + eps0);
Sp_soft = Mp_soft .* S;
MagPsoft = abs(Sp_soft);

dMag = diff(MagPsoft,1,2);
dMag = max(dMag,0);
flux = sum(dMag,1);
flux = flux / (max(flux)+1e-12);
flux_s = movmean(flux, 5);

thr = 1.6 * movmedian(flux_s, 41);  % augmente (1.6..2.0) si trop de voix
onset = flux_s > thr;

pre = 2; post = 5;                  % baisse post (2..4) si voix reste trop
gate = false(1, length(onset));
for i=1:length(onset)
    if onset(i)
        gate(max(1,i-pre):min(length(gate),i+post)) = true;
    end
end
gate = [gate(1), gate];             % aligner sur nb frames

% ==========================================================
% 2) Masque spectral DUR : garder uniquement là où P domine H
% ==========================================================
beta = 2.5;  % plus grand => enlève plus la voix (essaye 2, 3, 4)
Mp_bin = P > (beta * H);

% Appliquer le masque binaire + gate temporel
Sp = S .* Mp_bin .* (ones(size(S,1),1) * gate);

% ==========================================================
% 3) Atténuation bande "voix" UNIQUEMENT dans les frames d’attaque
% (ça enlève la voix qui reste au moment des hits)
% ==========================================================
f1 = 200; f2 = 4000;
atten_dB = -12;                   % essaye -9, -12, -18
atten = 10^(atten_dB/20);

idxVoice = (f >= f1) & (f <= f2);
G = ones(size(f));
G(idxVoice) = atten;

% Appliquer uniquement quand gate=1
Sp(:, gate) = (G * ones(1, nnz(gate))) .* Sp(:, gate);

% --- ISTFT ---
xp = istft(Sp, fs, "Window", w, "OverlapLength", winLen-hop, "FFTLength", nfft);
xp = real(xp);
xp = xp / max(abs(xp) + 1e-12);

audiowrite("percussions.wav", xp, fs);
disp("OK -> percussions.wav (onset + masque binaire + anti-voix)");
