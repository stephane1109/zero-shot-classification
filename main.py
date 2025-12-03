from transformers import pipeline

# Création du pipeline de classification zero-shot
# On utilise le modèle NLI mDeBERTa pour faire de la classification zero-shot en construisant
# des hypothèses à partir des étiquettes fournies.
classifieur = pipeline(
    task="zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
)

# Texte à analyser (exemple)
texte = (
    "Je suis étudiant, je me sens complètement dépassé par mes études et ma vie en ce moment. "
    "J’ai l’impression de perdre pied et je ne sais plus vers qui me tourner. "
    "Ces derniers temps, je pense de plus en plus à en finir et à mettre fin à mes jours, "
    "parce que je ne vois plus d’issue. Est-ce que tu peux m’aider ?"
)

# Définition des catégories de détresse et d’idéation suicidaire (étiquettes)
# Ces étiquettes servent à construire des hypothèses en langage naturel.
etiquettes = [
    "détresse psychologique sans idées suicidaires",
    "détresse psychologique avec idées suicidaires",
    "idées suicidaires sans plan concret",
    "idées suicidaires avec plan concret"
]

# Appel du classifieur en mode zero-shot
# Le paramètre hypothesis_template contrôle la phrase utilisée pour la tâche NLI.
# Pour chaque étiquette, le modèle évalue dans quelle mesure le texte implique
# l’hypothèse « Ce message exprime <étiquette> ».
resultat = classifieur(
    sequences=texte,
    candidate_labels=etiquettes,
    hypothesis_template="Ce message exprime {}.",
    multi_label=False  # on force une seule catégorie principale
)

# Affichage du résultat brut (scores et étiquette prédite)
print(resultat)
