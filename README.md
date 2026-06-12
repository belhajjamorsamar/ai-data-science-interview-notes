# Guide de Préparation Entretien Data Scientist
## Partie 1 — L'Intelligence Artificielle : Vue d'ensemble

### 1.1 Définitions fondamentales

**Intelligence Artificielle (IA)** : champ scientifique visant à créer des systèmes capables d'exécuter des tâches qui nécessitent normalement l'intelligence humaine (raisonnement, perception, prise de décision, langage). C'est le terme le plus large, englobant tous les autres.

**Machine Learning (ML)** : sous-domaine de l'IA où les systèmes apprennent à partir de données, sans être explicitement programmés pour chaque règle. Au lieu d'écrire `if temperature > 30: chaud`, on entraîne un modèle sur des exemples passés pour qu'il découvre lui-même les règles.

**Deep Learning (DL)** : sous-domaine du ML basé sur des réseaux de neurones artificiels à plusieurs couches ("profonds"). Ces réseaux apprennent automatiquement des représentations hiérarchiques des données (par exemple, en vision : contours → formes → objets).

**IA Générative (GenAI)** : sous-domaine du DL spécialisé dans la génération de nouveau contenu (texte, image, audio, code) plutôt que la simple prédiction ou classification.

### 1.2 Schéma hiérarchique

```
┌─────────────────────────────────────────────┐
│  INTELLIGENCE ARTIFICIELLE (IA)              │
│  Tout système simulant un comportement       │
│  "intelligent" (même des règles if/else)     │
│                                               │
│  ┌─────────────────────────────────────┐    │
│  │  MACHINE LEARNING (ML)               │    │
│  │  Apprentissage à partir de données   │    │
│  │  (régression linéaire, arbres,       │    │
│  │   SVM, random forest...)             │    │
│  │                                       │    │
│  │  ┌─────────────────────────────┐    │    │
│  │  │  DEEP LEARNING (DL)           │    │    │
│  │  │  Réseaux de neurones profonds │    │    │
│  │  │  (CNN, RNN, Transformers)      │    │    │
│  │  │                                 │    │    │
│  │  │  ┌───────────────────────┐    │    │    │
│  │  │  │  IA GÉNÉRATIVE         │    │    │    │
│  │  │  │  (GPT, Stable Diff.,   │    │    │    │
│  │  │  │   LLaMA, DALL-E...)    │    │    │    │
│  │  │  └───────────────────────┘    │    │    │
│  │  └─────────────────────────────┘    │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

**Analogie** : pensez à des poupées russes. L'IA est la plus grande poupée — elle contient tout. Le ML est une poupée plus petite à l'intérieur — une famille de techniques précises pour "apprendre" plutôt que "programmer". Le DL est encore plus petit — une sous-famille du ML qui utilise des réseaux de neurones empilés. La GenAI est la plus petite poupée — des modèles de DL spécialisés dans la création de contenu inédit.

### 1.3 Les trois niveaux d'IA : ANI, AGI, ASI

| Type | Nom complet | Description | Exemples actuels |
|------|------------|--------------|-------------------|
| **ANI** | Artificial Narrow Intelligence (IA étroite/faible) | Excelle dans **une seule tâche spécifique**, sans capacité de généralisation | ChatGPT, AlphaGo, systèmes de recommandation Netflix, reconnaissance faciale |
| **AGI** | Artificial General Intelligence (IA générale) | Capable d'apprendre et de raisonner dans **n'importe quel domaine**, comme un humain. **N'existe pas encore.** | Aucun système actuel — sujet de recherche actif (OpenAI, DeepMind, Anthropic) |
| **ASI** | Artificial Superintelligence (IA superintelligente) | Surpasserait l'intelligence humaine dans **tous** les domaines. **Purement hypothétique.** | Aucun — concept théorique/philosophique |

**Point clé pour l'entretien** : Tous les systèmes d'IA actuels, y compris les LLM les plus avancés (GPT-4, Claude, Gemini), sont des **ANI**. Même s'ils semblent généralistes (ils répondent à des questions très variées), ils restent fondamentalement des modèles statistiques entraînés sur un objectif précis (prédire le token suivant), sans compréhension, conscience ou raisonnement causal au sens humain.

### 1.4 Pourquoi cette distinction compte en entretien

Un recruteur pose souvent : *"Quelle est la différence entre IA, ML et DL ?"* pour vérifier que le candidat ne confond pas marketing et technique. La réponse attendue doit montrer :
1. La relation d'inclusion (IA ⊃ ML ⊃ DL ⊃ GenAI)
2. La distinction "programmation explicite des règles" (IA symbolique classique) vs "apprentissage à partir de données" (ML)
3. La distinction "features faites à la main" (ML classique) vs "features apprises automatiquement" (DL)

### Référence
- Russell, S. & Norvig, P. — *Artificial Intelligence: A Modern Approach* (4ᵉ édition, Pearson). Référence académique standard, utilisée notamment à Berkeley et Stanford pour les cours d'introduction à l'IA.

---

## Questions d'entretien typiques — Partie 1

**Q1. Quelle est la différence entre IA, ML et DL ?**
> L'IA est le champ global visant à simuler l'intelligence (incluant règles écrites à la main). Le ML est un sous-ensemble où le système apprend des patterns à partir de données. Le DL est un sous-ensemble du ML utilisant des réseaux de neurones profonds capables d'apprendre automatiquement les représentations (features), sans ingénierie manuelle.

**Q2. ChatGPT est-il une AGI ?**
> Non. C'est une ANI extrêmement performante et généraliste en apparence, mais elle reste un modèle statistique entraîné sur un objectif fixe (prédiction du prochain token), sans compréhension générale, planification autonome ou transfert de connaissance comparable à un humain dans tous les domaines.

**Q3. Donnez un exemple où l'IA n'implique pas de Machine Learning.**
> Un système expert basé sur des règles métier codées en dur (ex : un moteur de règles fiscales "if revenu > X alors taux = Y") est de l'IA symbolique classique, sans apprentissage à partir de données — donc pas du ML.

**Q4. Pourquoi le DL a-t-il explosé depuis 2012 ?**
> Trois facteurs convergents : (1) disponibilité de très grands jeux de données (ImageNet), (2) puissance de calcul GPU abordable, (3) avancées algorithmiques (ReLU, dropout, batch normalization, architectures comme AlexNet puis Transformers).

**Q5. Qu'est-ce qui distingue la GenAI du ML/DL "classique" ?**
> Le ML/DL classique apprend principalement à **prédire** une sortie (label, score, catégorie) à partir d'une entrée. La GenAI apprend la **distribution** des données pour pouvoir **générer** de nouveaux échantillons plausibles (texte, image, audio) jamais vus pendant l'entraînement.

---

## Partie 2 — Types d'Apprentissage

### 2.1 Vue d'ensemble

Il existe cinq grandes familles d'apprentissage, qui se distinguent principalement par **la nature et la disponibilité des labels** (étiquettes/cibles) dans les données.

### 2.2 Apprentissage supervisé (Supervised Learning)

**Définition** : on dispose d'un jeu de données où chaque exemple `x` est associé à une cible connue `y`. Le modèle apprend la fonction `f` telle que `f(x) ≈ y`.

**Deux sous-types** :
- **Classification** : `y` est une catégorie discrète (ex : spam / non-spam)
- **Régression** : `y` est une valeur continue (ex : prix d'une maison)

**Exemple concret — Prédire le prix d'une maison (régression)**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Données simplifiées : surface (m2), nb pièces, âge -> prix (k€)
data = pd.DataFrame({
    "surface": [50, 80, 120, 60, 200, 90, 45, 150],
    "pieces":  [2,  3,  5,   2,  6,   4,  1,  5],
    "age":     [10, 5,  20,  30, 2,   15, 40, 8],
    "prix":    [150, 240, 320, 140, 600, 270, 100, 450]
})

X = data[["surface", "pieces", "age"]]
y = data["prix"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, predictions))
print("Coefficients:", dict(zip(X.columns, model.coef_)))
```

**Intuition** : c'est comme un élève qui s'entraîne avec un corrigé. Il voit des centaines d'exercices résolus (`x` → `y`), et apprend la "formule" générale pour résoudre de nouveaux exercices similaires.

### 2.3 Apprentissage non supervisé (Unsupervised Learning)

**Définition** : on dispose uniquement de `x`, sans labels `y`. L'objectif est de découvrir une **structure cachée** dans les données.

**Deux sous-types principaux** :
- **Clustering** : regrouper des observations similaires (ex : segmentation clients)
- **Réduction de dimension** : compresser l'information en gardant l'essentiel (ex : PCA, t-SNE, UMAP)

**Exemple concret — Segmentation clients (clustering)**

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Données clients : montant dépensé/an, fréquence d'achat
clients = pd.DataFrame({
    "depense_annuelle": [1200, 300, 5000, 250, 4800, 1100, 200, 5200],
    "frequence_achat":  [12,   2,   40,   3,   38,   11,   1,   42]
})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(clients)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clients["segment"] = kmeans.fit_predict(X_scaled)

print(clients)
# segment 0 : petits acheteurs occasionnels
# segment 1 : gros clients fidèles (VIP)
```

**Intuition** : c'est comme demander à quelqu'un de trier des photos de fruits sans lui dire les noms — il va naturellement regrouper les ronds rouges, les allongés jaunes, etc., en se basant sur les similarités visuelles, sans connaître les étiquettes "pomme" ou "banane".

### 2.4 Apprentissage semi-supervisé (Semi-Supervised Learning)

**Définition** : on dispose d'une **petite quantité de données labellisées** et d'une **grande quantité de données non labellisées**. Le modèle exploite les deux pour améliorer ses performances par rapport à un entraînement purement supervisé sur le petit échantillon labellisé.

**Exemple concret — Classification d'images avec peu de labels**

Cas typique : sur 10 000 images médicales, seules 200 ont été annotées par un radiologue (coût élevé de l'annotation). Approche :
1. Entraîner un modèle initial sur les 200 images labellisées
2. Utiliser ce modèle pour prédire des "pseudo-labels" sur les 9 800 images restantes
3. Ne garder que les prédictions à haute confiance comme pseudo-labels
4. Ré-entraîner le modèle sur l'ensemble (200 vrais labels + pseudo-labels fiables)

```python
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
import numpy as np

# y = -1 signifie "non labellisé"
X = np.random.rand(1000, 10)
y = np.full(1000, -1)
y[:50] = np.random.randint(0, 2, size=50)  # seulement 50 labels connus

base_classifier = SVC(probability=True, gamma="auto")
self_training_model = SelfTrainingClassifier(base_classifier, threshold=0.8)
self_training_model.fit(X, y)
```

**Intuition** : c'est comme apprendre une langue avec seulement quelques phrases traduites par un professeur, puis deviner le sens des autres phrases en s'appuyant sur les patterns déjà appris, et ne retenir que les déductions dont on est très confiant.

### 2.5 Apprentissage par renforcement (Reinforcement Learning, RL)

**Définition** : un **agent** interagit avec un **environnement**. À chaque action, il reçoit une **récompense** (positive ou négative) et observe un nouvel **état**. L'objectif est d'apprendre une **politique** (stratégie) qui maximise la récompense cumulée à long terme.

**Composants clés** :
- **État (state)** : situation actuelle
- **Action** : décision possible
- **Récompense (reward)** : signal de feedback
- **Politique (policy)** : stratégie qui mappe états → actions

**Exemples célèbres** : AlphaGo (Go), agents jouant aux jeux Atari (DQN, DeepMind 2013), robots apprenant à marcher, RLHF pour aligner les LLM (ChatGPT).

```python
# Exemple minimal : Q-Learning sur un environnement simple (Gym/Gymnasium)
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n
Q = np.zeros((n_states, n_actions))

alpha, gamma, epsilon = 0.8, 0.95, 0.1

for episode in range(2000):
    state, _ = env.reset()
    done = False
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # exploration
        else:
            action = np.argmax(Q[state])  # exploitation

        next_state, reward, done, _, _ = env.step(action)

        # Mise à jour Q-Learning (équation de Bellman)
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        state = next_state

print("Table Q apprise :")
print(Q)
```

**Intuition** : c'est comme entraîner un chien avec des friandises. Il essaie différentes actions, reçoit une récompense (friandise) ou rien, et ajuste son comportement pour maximiser les friandises futures — sans qu'on lui dise explicitement "assieds-toi" au sens d'une supervision directe à chaque instant.

### 2.6 Apprentissage auto-supervisé (Self-Supervised Learning)

**Définition** : le modèle **génère ses propres labels** à partir des données brutes elles-mêmes, sans annotation humaine. C'est la base de l'entraînement des grands modèles modernes (BERT, GPT).

**Exemples concrets** :
- **BERT** : on masque aléatoirement 15% des mots d'une phrase, et le modèle doit prédire les mots masqués (Masked Language Modeling). Le "label" est le mot original — déjà présent dans le texte, donc pas besoin d'annotation humaine.
- **GPT** : on entraîne le modèle à prédire le mot suivant à partir des mots précédents (Causal Language Modeling). Encore une fois, le label (le mot suivant) est déjà dans le texte.

```python
from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

text = "Paris is the [MASK] of France."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

mask_idx = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]
predicted_token_id = outputs.logits[0, mask_idx].argmax(dim=-1)
print(tokenizer.decode(predicted_token_id))  # -> "capital"
```

**Intuition** : c'est comme un exercice à trous où le texte lui-même contient déjà la réponse cachée — on cache une partie de l'information et on demande au modèle de la retrouver, ce qui génère automatiquement des millions d'exercices "gratuits" à partir de textes bruts non annotés.

### 2.7 Tableau comparatif

| Type | Données disponibles | Objectif | Exemples |
|------|---------------------|----------|----------|
| **Supervisé** | `(x, y)` complets et labellisés | Prédire `y` à partir de `x` | Régression prix, classification spam |
| **Non supervisé** | `x` seul, sans label | Découvrir une structure cachée | Clustering, réduction de dimension |
| **Semi-supervisé** | Peu de `(x,y)` + beaucoup de `x` seuls | Exploiter les données non labellisées pour améliorer le modèle | Classif. images médicales avec peu d'annotations |
| **Par renforcement** | Pas de dataset fixe ; interaction agent/environnement | Maximiser une récompense cumulée | AlphaGo, robotique, RLHF |
| **Auto-supervisé** | `x` seul, labels générés automatiquement à partir de `x` | Apprendre des représentations générales réutilisables | BERT, GPT, modèles de fondation |

### Référence
- Bishop, C. M. — *Pattern Recognition and Machine Learning* (Springer, 2006). Référence académique classique sur les fondations probabilistes du ML, utilisée dans de nombreux cours universitaires (dont Cambridge, MIT).

---

## Questions d'entretien typiques — Partie 2

**Q1. Quelle est la différence entre apprentissage supervisé et non supervisé ?**
> En supervisé, on dispose de la cible `y` pour chaque exemple, et le but est d'apprendre à la prédire. En non supervisé, il n'y a pas de cible : le but est de découvrir une structure (clusters, axes principaux de variance) directement dans les données.

**Q2. Donnez un cas réel où l'apprentissage semi-supervisé est utile.**
> En imagerie médicale, annoter des images (par un radiologue) est coûteux et lent. On a souvent quelques centaines d'images annotées et des milliers non annotées. Le semi-supervisé permet d'exploiter les images non annotées via des pseudo-labels pour améliorer le modèle sans coût d'annotation supplémentaire.

**Q3. En quoi le Reinforcement Learning diffère-t-il fondamentalement du supervisé ?**
> En supervisé, on connaît la "bonne réponse" immédiate pour chaque exemple. En RL, il n'y a pas de bonne réponse explicite : l'agent reçoit seulement une récompense, parfois retardée (ex : gagner une partie d'échecs après 40 coups), et doit apprendre par essais/erreurs quelle séquence d'actions maximise la récompense future.

**Q4. Pourquoi dit-on que BERT et GPT sont "auto-supervisés" et pas "non supervisés" ?**
> Parce qu'il existe bien un signal de supervision (un "label"), mais celui-ci est généré automatiquement à partir des données elles-mêmes (le mot masqué, le mot suivant), sans intervention humaine — contrairement au non supervisé où aucun signal cible n'existe.

**Q5. Quelle est la différence entre classification et régression dans le supervisé ?**
> La classification prédit une catégorie discrète (ex : "chat"/"chien", "fraude"/"non-fraude"), tandis que la régression prédit une valeur numérique continue (ex : un prix, une température, un âge).
