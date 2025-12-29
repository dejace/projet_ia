import os
import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf

def convert_mp3_to_wav(mp3_path):
    """
    Convertit un fichier MP3 en WAV pour garantir la compatibilité.
    """
    wav_path = mp3_path.replace(".mp3", ".wav")
    print(f"Conversion de {mp3_path} en {wav_path}...")
    try:
        data, samplerate = librosa.load(mp3_path, sr=None)
        sf.write(wav_path, data, samplerate)
        print("Conversion réussie.")
        return wav_path
    except Exception as e:
        print(f"Erreur lors de la conversion : {e}")
        return None

def separate_with_open_unmix(file_path):
    """
    Utilise le réseau de neurones profond Open-Unmix (UMX) pour séparer 
    l'audio en Voix, Batterie et Autres (Musique).
    Version corrigée pour éviter l'erreur torchcodec.
    """
    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier {file_path} est introuvable.")
        return

    print(f"Chargement du réseau de neurones Open-Unmix...")
    
    try:
        # 1. Chargement du modèle pré-entraîné
        # trust_repo=True évite l'alerte de sécurité dans la console
        device = "cuda" if torch.cuda.is_available() else "cpu"
        separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxhq', device=device, trust_repo=True)
        
        # 2. Chargement de l'audio via Librosa pour éviter l'erreur torchcodec
        # Le modèle UMX nécessite du 44100Hz
        audio_np, rate = librosa.load(file_path, sr=44100, mono=False)
        
        # S'assurer que l'audio est en stéréo (2, samples)
        if audio_np.ndim == 1:
            audio_np = np.stack([audio_np, audio_np])
            
        # Conversion en Tensor PyTorch
        audio_tensor = torch.tensor(audio_np, dtype=torch.float32).to(device)

        print(f"Inférence de l'IA en cours sur {device} (Modèle UMXHQ)...")
        
        # 3. Séparation (Inférence)
        # On ajoute la dimension batch attendue par le modèle : [1, channels, samples]
        estimates = separator(audio_tensor.unsqueeze(0)) 
        estimates = estimates.squeeze(0) # On retire le batch

        # 4. Exportation des résultats
        base_dir = os.path.dirname(file_path)
        
        # Index de UMX : 0:vocals, 1:drums, 2:bass, 3:other
        vocals = estimates[0].detach().cpu().numpy().T
        drums = estimates[1].detach().cpu().numpy().T
        # On combine Bass et Other pour la piste musique
        music = (estimates[2] + estimates[3]).detach().cpu().numpy().T

        # Sauvegarde des fichiers en haute qualité
        sf.write(os.path.join(base_dir, "vocals_output.wav"), vocals, 44100)
        sf.write(os.path.join(base_dir, "percussions_output.wav"), drums, 44100)
        sf.write(os.path.join(base_dir, "music_output.wav"), music, 44100)

        print("\nSéparation par Réseau de Neurones terminée !")
        print(f"-> Voix isolée : vocals_output.wav")
        print(f"-> Percussions isolées : percussions_output.wav")
        print(f"-> Musique isolée : music_output.wav")
        return True

    except Exception as e:
        print(f"Erreur lors de l'exécution du modèle : {e}")
        return False

if __name__ == "__main__":
    # Chemin vers votre fichier
    chemin_audio = r"c:/Users/dejac/Documents/HEPL/Master_1/Q1/ia/Projet_ia/test1.mp3"
    
    # 1. Vérification/Conversion
    if chemin_audio.lower().endswith(".mp3"):
        wav_path = convert_mp3_to_wav(chemin_audio)
    else:
        wav_path = chemin_audio
        
    # 2. Lancement du modèle
    if wav_path:
        separate_with_open_unmix(wav_path)