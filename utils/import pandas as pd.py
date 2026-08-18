import matplotlib.pyplot as plt

chemin = r"C:\Users\alach\OneDrive\Desktop\modele_1\yolov8s_kolomverse\results.csv"

epochs = []
map50 = []

with open(chemin, 'r') as f:
    lignes = f.readlines()

# Ignorer l'en-tête
for ligne in lignes[1:]:
    valeurs = ligne.strip().split(',')
    if valeurs:
        try:
            epochs.append(float(valeurs[0].strip()))
            map50.append(float(valeurs[6].strip()))
        except:
            pass

print(f"✅ {len(epochs)} époques chargées")
print(f"mAP50 finale: {map50[-1]:.4f}")
print(f"Amélioration: +{(map50[-1] - map50[0])*100:.1f}%")

# Graphique
plt.figure(figsize=(12, 6))
plt.plot(epochs, map50, 'g-', linewidth=2)
plt.xlabel("Époque", fontsize=12)
plt.ylabel("mAP50", fontsize=12)
plt.title("YOLOv8s - Progression de la mAP50", fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()