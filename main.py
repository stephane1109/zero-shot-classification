# pip install torch
# pip install transformers
from transformers import pipeline

def afficher_resultats_table(resultat):
    """
    Affiche les résultats de la classification zero-shot sous forme de tableau
    dans le terminal à partir du dictionnaire renvoyé par le pipeline.
    """
    sequence = resultat["sequence"]
    labels = resultat["labels"]
    scores = resultat["scores"]

    # Affichage du texte analysé
    print("Texte analysé :")
    print(sequence)
    print()

    # Titre du tableau
    print("Résultats de la classification zero-shot")
    print("-" * 90)
    print(f"{'Rang':<6}{'Étiquette':<65}{'Score':>10}")
    print("-" * 90)

    # Affichage de chaque étiquette avec son score
    for i, (label, score) in enumerate(zip(labels, scores), start=1):
        print(f"{i:<6}{label:<65}{score:>10.4f}")

    print("-" * 90)


if __name__ == "__main__":
    # Création du pipeline de classification zero-shot
    # On utilise le modèle NLI mDeBERTa pour faire de la classification zero-shot
    # en construisant des hypothèses à partir des étiquettes fournies.
    classifieur = pipeline(
        task="zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    )

    # Texte brut de la personne (exemple)
    texte_brut = (
        "Je suis étudiant, je me sens complètement dépassé par mes études et ma vie en ce moment. "
        "J’ai l’impression de perdre pied et je ne sais plus vers qui me tourner. "
        "Ces derniers temps, je pense de plus en plus à en finir et à mettre fin à mes jours, "
        "parce que je ne vois plus d’issue. Est-ce que tu peux m’aider ?"
    )

    # Demande éventuelle de contextualisation à l'utilisateur
    print("Vous pouvez saisir une contextualisation (exemple : 'message envoyé à Allo Suicide').")
    print("Si vous ne souhaitez pas ajouter de contexte, appuyez simplement sur Entrée.")
    contextualisation = input("Contextualisation : ").strip()

    # Construction du texte envoyé au modèle et du gabarit d'hypothèse
    if contextualisation:
        # Si une contextualisation est saisie, on la place avant le message
        texte = contextualisation + "\nMessage : " + texte_brut
        hypothesis_template = "Dans ce contexte, ce message correspond à {}."
    else:
        # Si aucune contextualisation n'est saisie, on utilise uniquement le message brut
        texte = texte_brut
        hypothesis_template = "Ce message exprime {}."

    # Définition des catégories de détresse et d’idéation suicidaire (étiquettes)
    etiquettes = [
        "détresse psychologique sans idées suicidaires",
        "détresse psychologique avec idées suicidaires",
        "idées suicidaires sans plan concret",
        "idées suicidaires avec plan concret"
    ]

    # Appel du classifieur en mode zero-shot
    resultat = classifieur(
        sequences=texte,
        candidate_labels=etiquettes,
        hypothesis_template=hypothesis_template,
        multi_label=False  # on force une seule catégorie principale
    )

    # Affichage du dictionnaire brut (comme dans ton exemple)
    print("Dictionnaire brut renvoyé par le pipeline :")
    print(resultat)
    print()

    # Affichage sous forme de tableau dans le terminal
    afficher_resultats_table(resultat)

