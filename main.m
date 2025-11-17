clear; close all; clc;

% -------------------------------
% PARAMÈTRES
% -------------------------------
f0 = 392;          % fondamentale du SOL
N_harm = 10;       % nombre d'harmoniques à analyser

% Liste des fichiers audio dans ton dossier
fichiers = { ...
    'violon.wav', ...
    'guitar.wav', ...
    'harmonica.wav', ...
    'melodica.wav', ...
    'flute_a_bec.wav', ...
    'flute_traversiere.wav', ...
    'tin_whistle.wav' ...
};

nInstruments = numel(fichiers);

% Créer automatiquement les noms des instruments (optionnel)
noms = erase(fichiers, '.wav');

% -------------------------------
% ANALYSE ET AFFICHAGE
% -------------------------------
figure;

for k = 1:nInstruments
    
    % --- Lecture du fichier ---
    [x, Fs] = audioread(fichiers{k});
    x = mean(x,2);                 % passe en mono
    x = x .* hann(length(x));      % fenêtrage
    
    % --- FFT ---
    N = 2^nextpow2(length(x));
    X = fft(x, N);
    f = (0:N-1) * Fs / N;
    mag = abs(X);
    
    % On ne garde que les fréquences positives
    f = f(1:floor(N/2));
    mag = mag(1:floor(N/2));

    % --- Extraction des harmoniques ---
    harm = 1:N_harm;
    amps = zeros(1, N_harm);

    for n = 1:N_harm
        f_cible = n * f0;
        [~, idx] = min(abs(f - f_cible));
        amps(n) = mag(idx);
    end

    % Normalisation
    amps = amps / max(amps);

    % --- Affichage ---
    subplot(nInstruments, 1, k);
    stem(harm, amps, 'filled');
    grid on;
    title(['Harmoniques - ' noms{k}]);
    xlabel('Numéro d''harmonique');
    ylabel('Amplitude normalisée');
    xlim([0.5 N_harm + 0.5]);
end
