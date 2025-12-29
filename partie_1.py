import os
import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf

def ensure_wav_format(file_path):
    """
    Vérifie le format d'entrée. 
    - Si c'est un MP3 : le convertit en WAV et retourne le nouveau chemin.
    - Si c'est déjà un WAV : retourne le chemin tel quel.
    """
    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier {file_path} est introuvable.")
        return None

    # Cas 1 : C'est déjà un WAV
    if file_path.lower().endswith(".wav"):
        print(f"Fichier WAV détecté (pas de conversion nécessaire) : {file_path}")
        return file_path

    # Cas 2 : C'est un MP3, on convertit
    if file_path.lower().endswith(".mp3"):
        wav_path = file_path.replace(".mp3", ".wav")
        # On évite de reconvertir si le WAV existe déjà
        if os.path.exists(wav_path):
            print(f"Le fichier WAV converti existe déjà : {wav_path}")
            return wav_path
            
        print(f"Conversion de {file_path} en {wav_path}...")
        try:
            data, samplerate = librosa.load(file_path, sr=None)
            sf.write(wav_path, data, samplerate)
            print("Conversion réussie.")
            return wav_path
        except Exception as e:
            print(f"Erreur lors de la conversion : {e}")
            return None
            
    print(f"Format non supporté : {file_path}")
    return None

def clean_silence(audio_data, sr, threshold_db, attack_release_smoothness=0.05):
    """
    Nettoie les parties où l'instrument/voix n'est pas présent.
    
    Args:
        audio_data: Le signal audio (numpy array).
        sr: Taux d'échantillonnage.
        threshold_db: Seuil en décibels. Tout ce qui est en dessous est considéré comme du silence/bruit.
                      Ex: -40dB est un bon standard, -60dB est très permissif, -20dB est très strict.
        attack_release_smoothness: Temps de lissage pour éviter les coupures sèches.
    """
    # Calcul de l'énergie (RMS) sur des petites fenêtres
    hop_length = 512
    mse = librosa.feature.rms(y=audio_data, frame_length=2048, hop_length=hop_length)
    mse_db = librosa.amplitude_to_db(mse, ref=np.max)
    
    # Création du masque : 1 si le son est fort, 0 si c'est du bruit de fond
    mask = mse_db > threshold_db
    
    # On convertit ce masque (qui est par fenêtres) à la taille de l'audio (par échantillons)
    # L'interpolation crée un effet de fondu naturel
    mask_indices = np.linspace(0, len(audio_data), num=mask.shape[1])
    target_indices = np.arange(len(audio_data))
    
    # Interpolation linéaire pour lisser le masque
    smooth_mask = np.interp(target_indices, mask_indices, mask[0].astype(float))
    
    # Application d'un lissage supplémentaire pour éviter les "clics" (Convolution)
    # Cela simule l'Attack et le Release d'un compresseur
    kernel_size = int(sr * attack_release_smoothness)
    kernel = np.ones(kernel_size) / kernel_size
    smooth_mask = np.convolve(smooth_mask, kernel, mode='same')
    
    # On s'assure que le masque reste entre 0 et 1
    smooth_mask = np.clip(smooth_mask, 0, 1)
    
    # Si le masque est très proche de 0 (silence), on force le 0 absolu pour éviter le souffle
    smooth_mask[smooth_mask < 0.01] = 0
    
    return audio_data * smooth_mask

def separate_with_open_unmix(file_path):
    """
    Utilise le réseau de neurones profond Open-Unmix (UMX) avec un post-traitement
    de nettoyage des silences (Noise Gate).
    """
    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier {file_path} est introuvable.")
        return

    print(f"Chargement du réseau de neurones Open-Unmix...")
    
    try:
        # 1. Chargement du modèle pré-entraîné
        device = "cuda" if torch.cuda.is_available() else "cpu"
        separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxhq', device=device, trust_repo=True)
        
        # 2. Chargement de l'audio via Librosa
        audio_np, rate = librosa.load(file_path, sr=44100, mono=False)
        
        if audio_np.ndim == 1:
            audio_np = np.stack([audio_np, audio_np])
            
        audio_tensor = torch.tensor(audio_np, dtype=torch.float32).to(device)

        print(f"Inférence de l'IA en cours sur {device} (Modèle UMXHQ)...")
        
        # 3. Séparation (Inférence)
        with torch.no_grad():
            estimates = separator(audio_tensor.unsqueeze(0))
            estimates = estimates.squeeze(0)

        # 4. Extraction et Post-Traitement
        base_dir = os.path.dirname(file_path)
        
        # Extraction brute (Transposée pour format (samples, channels))
        vocals = estimates[0].detach().cpu().numpy()
        drums = estimates[1].detach().cpu().numpy()
        # Musique = Basse + Other
        music = (estimates[2] + estimates[3]).detach().cpu().numpy()

        print("Nettoyage des pistes (Suppression des artefacts dans les zones de silence)...")
        
        # Application du Smart Gate canal par canal (Gauche/Droite)
        # On transpose pour itérer sur les canaux (OpenUnmix donne [channels, samples])
        
        # --- Nettoyage VOIX ---
        # Seuil strict (-45dB) car la voix ne doit pas avoir de résidus constants
        for ch in range(vocals.shape[0]):
            vocals[ch] = clean_silence(vocals[ch], rate, threshold_db=-45)

        # --- Nettoyage PERCUSSIONS ---
        # Seuil moyen (-40dB) pour garder les petites notes fantômes mais virer le fond
        for ch in range(drums.shape[0]):
            drums[ch] = clean_silence(drums[ch], rate, threshold_db=-40)
            
        # --- Nettoyage MUSIQUE ---
        # Seuil léger (-60dB) car la musique est souvent continue
        for ch in range(music.shape[0]):
            music[ch] = clean_silence(music[ch], rate, threshold_db=-60)

        # Sauvegarde (On transpose ici .T pour que soundfile accepte le format)
        sf.write(os.path.join(base_dir, "vocals_output.wav"), vocals.T, 44100)
        sf.write(os.path.join(base_dir, "percussions_output.wav"), drums.T, 44100)
        sf.write(os.path.join(base_dir, "music_output.wav"), music.T, 44100)

        print("\nSéparation terminée et nettoyée !")
        print(f"-> Voix isolée (sans interférences) : vocals_output.wav")
        print(f"-> Percussions isolées : percussions_output.wav")
        print(f"-> Musique isolée : music_output.wav")
        return True

    except Exception as e:
        print(f"Erreur lors de l'exécution du modèle : {e}")
        return False

if __name__ == "__main__":
    print("--- Séparateur Audio IA (Open-Unmix) ---")
    # On demande à l'utilisateur de saisir le chemin ou le nom du fichier
    chemin_audio = input("Entrez le chemin complet ou le nom du fichier audio (ex: test1.mp3) : ").strip()
    
    # Nettoyage des guillemets si l'utilisateur fait un copier-coller de chemin Windows
    chemin_audio = chemin_audio.strip('"').strip("'")

    if not chemin_audio:
        print("Erreur : Aucun fichier spécifié.")
    else:
        # Le script gère maintenant automatiquement le format
        wav_path = ensure_wav_format(chemin_audio)
        
        if wav_path:
            separate_with_open_unmix(wav_path)