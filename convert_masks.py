import os
import json
import numpy as np
import cv2

# --- CONFIGURATION ---
# Assurez-vous que ces dossiers existent au même endroit que ce script
input_dir = "json_files"   # Là où vous mettrez vos fichiers .json (venant de LabelMe)
output_dir = "masks"       # Là où les images noir/blanc seront créées (le script le créera)

# Créer le dossier de sortie s'il n'existe pas
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Début de la conversion des fichiers de '{input_dir}' vers '{output_dir}'...")

# Parcourir tous les fichiers JSON
files = os.listdir(input_dir)
count = 0
for file in files:
    if file.endswith(".json"):
        json_path = os.path.join(input_dir, file)
        
        # 1. Lire le fichier JSON
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # 2. Récupérer la taille originale de l'image
        h = data["imageHeight"]
        w = data["imageWidth"]
        
        # Créer un masque noir (fond)
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 3. Dessiner les polygones en BLANC
        for shape in data["shapes"]:
            points = shape["points"]
            points_array = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points_array], 255)
            
        # 4. Sauvegarder le masque
        filename = file.replace(".json", ".png")
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, mask)
        
        print(f"Converti : {file} -> {filename}")
        count += 1

if count == 0:
    print("ATTENTION : Aucun fichier .json trouvé dans le dossier 'json_files' !")
else:
    print(f"Terminé ! {count} masques créés dans le dossier 'masks'.")