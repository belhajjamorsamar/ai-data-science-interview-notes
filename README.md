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
>
> # Guide de Préparation Entretien Data Scientist
## Partie 3 — Les Bibliothèques Python essentielles

Pour chaque bibliothèque : rôle, cas d'usage typique, et exemple de code minimal mais fonctionnel.

---

### 3.1 NumPy — Calcul numérique et matrices

**Rôle** : fournit le type `ndarray` (tableau multidimensionnel) et des opérations vectorisées ultra-rapides (implémentées en C), base de tout l'écosystème data science Python.

**Cas d'usage** : opérations matricielles, calculs statistiques, manipulation de tenseurs avant de passer à PyTorch/TensorFlow.

```python
import numpy as np

# Création de matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Opérations vectorielles (sans boucle for !)
print("Somme élément par élément:\n", A + B)
print("Produit matriciel:\n", A @ B)
print("Transposée:\n", A.T)
print("Moyenne par colonne:", A.mean(axis=0))

# Broadcasting : opérer sur des tableaux de tailles différentes
vecteur = np.array([10, 20])
print("Broadcasting (A + vecteur):\n", A + vecteur)
```

**Intuition** : NumPy traite des tableaux entiers d'un coup ("vectorisation"), au lieu de boucler élément par élément en Python pur — c'est 10 à 100x plus rapide.

---

### 3.2 Pandas — Manipulation de données tabulaires

**Rôle** : structures `DataFrame` (tableau 2D type Excel) et `Series` (colonne 1D) pour charger, nettoyer, transformer et agréger des données tabulaires.

```python
import pandas as pd

df = pd.DataFrame({
    "nom": ["Alice", "Bob", "Carla", "David"],
    "age": [25, 32, 29, None],
    "salaire": [3000, 4200, 3800, 5000],
    "ville": ["Paris", "Lyon", "Paris", "Lyon"]
})

# Nettoyage : remplir les valeurs manquantes
df["age"] = df["age"].fillna(df["age"].median())

# Filtrage
parisiens = df[df["ville"] == "Paris"]

# Agrégation par groupe
moyenne_par_ville = df.groupby("ville")["salaire"].mean()
print(moyenne_par_ville)

# Création de nouvelle colonne (feature engineering simple)
df["salaire_annuel"] = df["salaire"] * 12

print(df)
```

---

### 3.3 Matplotlib / Seaborn — Visualisation

**Rôle** : Matplotlib est la librairie de base pour tracer des graphiques (bas niveau, très flexible). Seaborn est construit sur Matplotlib et offre des visualisations statistiques élégantes avec moins de code.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "age": np.random.randint(20, 60, 200),
    "salaire": np.random.normal(40000, 10000, 200),
    "departement": np.random.choice(["RH", "IT", "Finance"], 200)
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Matplotlib : histogramme simple
axes[0].hist(df["salaire"], bins=20, color="steelblue", edgecolor="black")
axes[0].set_title("Distribution des salaires (Matplotlib)")

# Seaborn : boxplot par catégorie + style esthétique automatique
sns.boxplot(data=df, x="departement", y="salaire", ax=axes[1])
axes[1].set_title("Salaire par département (Seaborn)")

plt.tight_layout()
plt.savefig("graphiques.png")
```

**Quand utiliser quoi** : Seaborn pour l'exploration rapide (EDA) avec des données tabulaires (boxplots, heatmaps de corrélation, pairplots) ; Matplotlib pour un contrôle fin (graphiques personnalisés, dashboards, figures pour publication).

---

### 3.4 Scikit-learn — Machine Learning classique

**Rôle** : implémentation cohérente et standardisée de tous les algorithmes ML classiques (régression, classification, clustering, réduction de dimension), avec une API unifiée (`fit`, `predict`, `transform`).

**Exemple de pipeline complet** :

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Données fictives
df = pd.DataFrame({
    "age": [25, 45, 35, 50, 23, 41, 38, 29],
    "salaire": [2500, 5000, 3800, 6200, 2200, 4700, 3900, 2800],
    "ville": ["Paris", "Lyon", "Paris", "Lyon", "Paris", "Lyon", "Paris", "Lyon"],
    "achat": [0, 1, 0, 1, 0, 1, 1, 0]
})

X = df.drop("achat", axis=1)
y = df["achat"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Preprocessing différencié selon le type de colonne
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["age", "salaire"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["ville"])
])

# Pipeline complet : preprocessing + modèle
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred, zero_division=0))
```

**Pourquoi le Pipeline est crucial** : il garantit que le même preprocessing (fit sur train uniquement) est appliqué à l'inférence, évitant le data leakage et simplifiant le déploiement.

---

### 3.5 PyTorch — Deep Learning, autograd

**Rôle** : framework de DL le plus utilisé en recherche, basé sur les tensors (similaires à NumPy mais avec calcul GPU et différenciation automatique).

**Concept clé : autograd** — PyTorch construit dynamiquement un graphe de calcul et calcule automatiquement les gradients par rétropropagation.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Exemple : différenciation automatique
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x
y.backward()  # calcule dy/dx automatiquement
print("dy/dx en x=2 :", x.grad.item())  # -> 3*x^2 + 2 = 14

# Réseau de neurones simple (régression)
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Données fictives
X = torch.randn(32, 3)   # batch de 32 exemples, 3 features
y_true = torch.randn(32, 1)

# Une étape d'entraînement
optimizer.zero_grad()
y_pred = model(X)
loss = criterion(y_pred, y_true)
loss.backward()    # rétropropagation
optimizer.step()   # mise à jour des poids

print("Loss:", loss.item())
```

---

### 3.6 TensorFlow / Keras — Alternative à PyTorch

**Rôle** : framework DL de Google. Keras est son API haut niveau, très utilisée en industrie et pour le prototypage rapide grâce à sa syntaxe déclarative.

**Exemple : CNN simple pour classification d'images**

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")  # 10 classes
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
```

**PyTorch vs Keras/TF** : PyTorch domine la recherche (flexibilité, debug intuitif, graphe dynamique) ; Keras/TF reste répandu en production et dans certains environnements industriels (déploiement mobile via TFLite, écosystème TF Serving).

---

### 3.7 HuggingFace Transformers — NLP et fine-tuning de LLM

**Rôle** : bibliothèque qui démocratise l'accès aux modèles pré-entraînés (BERT, GPT, T5, LLaMA...) avec une API unifiée pour l'inférence et le fine-tuning.

```python
from transformers import pipeline

# Pipeline prêt à l'emploi pour l'analyse de sentiment
classifier = pipeline("sentiment-analysis")
result = classifier("I absolutely love this product!")
print(result)
# -> [{'label': 'POSITIVE', 'score': 0.999...}]

# Pipeline pour la génération de texte
generator = pipeline("text-generation", model="gpt2")
output = generator("The future of AI is", max_length=30, num_return_sequences=1)
print(output[0]["generated_text"])
```

---

### 3.8 LangChain — Orchestration de LLM

**Rôle** : framework pour construire des applications complexes autour des LLM (chaînage de prompts, agents, RAG, mémoire conversationnelle, intégration d'outils externes).

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

prompt = ChatPromptTemplate.from_template(
    "Résume ce texte en une phrase : {texte}"
)

# Chaînage : prompt -> LLM -> parsing de la sortie
chain = prompt | llm | StrOutputParser()

resultat = chain.invoke({"texte": "Un long article sur le changement climatique..."})
print(resultat)
```

---

### 3.9 OpenCV — Vision par ordinateur

**Rôle** : bibliothèque de référence pour le traitement d'images et de vidéos (lecture, transformation géométrique, filtres, détection de contours/visages, etc.), souvent utilisée en preprocessing avant un modèle DL.

```python
import cv2
import numpy as np

# Lecture d'une image
img = cv2.imread("photo.jpg")

# Conversion en niveaux de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Détection de contours (Canny)
edges = cv2.Canny(gray, threshold1=100, threshold2=200)

# Redimensionnement (préprocessing standard avant un CNN)
resized = cv2.resize(img, (224, 224))

# Détection de visages avec un modèle pré-entraîné Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
print(f"{len(faces)} visage(s) détecté(s)")

cv2.imwrite("contours.jpg", edges)
```

---

### 3.10 Tableau récapitulatif

| Bibliothèque | Domaine | Rôle principal |
|---|---|---|
| NumPy | Calcul numérique | Tableaux/matrices, opérations vectorisées |
| Pandas | Données tabulaires | Chargement, nettoyage, agrégation |
| Matplotlib/Seaborn | Visualisation | Graphiques exploratoires et de présentation |
| Scikit-learn | ML classique | Pipelines, modèles classiques, métriques |
| PyTorch | Deep Learning | Réseaux de neurones, autograd, recherche |
| TensorFlow/Keras | Deep Learning | Réseaux de neurones, production/mobile |
| HuggingFace Transformers | NLP/LLM | Modèles pré-entraînés, fine-tuning |
| LangChain | Orchestration LLM | Chaînes de prompts, agents, RAG |
| OpenCV | Vision par ordinateur | Traitement d'image classique, preprocessing |

### Référence
- Documentation officielle : [numpy.org](https://numpy.org), [pandas.pydata.org](https://pandas.pydata.org), [scikit-learn.org](https://scikit-learn.org), [pytorch.org](https://pytorch.org), [tensorflow.org](https://www.tensorflow.org), [huggingface.co/docs](https://huggingface.co/docs), [python.langchain.com](https://python.langchain.com), [opencv.org](https://opencv.org)
- Cours : [fast.ai](https://www.fast.ai) — "Practical Deep Learning for Coders", très orienté code et PyTorch.

---

## Questions d'entretien typiques — Partie 3

**Q1. Pourquoi NumPy est-il plus rapide qu'une boucle Python pure ?**
> Parce que NumPy délègue les calculs à du code C/Fortran compilé et vectorisé (utilisant SIMD), évitant l'overhead de l'interpréteur Python à chaque itération.

**Q2. Quand utiliser Pandas vs NumPy ?**
> NumPy pour des calculs numériques purs sur des tableaux homogènes (matrices, tenseurs). Pandas quand les données sont tabulaires, hétérogènes (colonnes de types différents), avec besoin d'indexation par nom de colonne, jointures, groupby, gestion de valeurs manquantes.

**Q3. Quel est l'intérêt d'un `Pipeline` scikit-learn par rapport à appliquer les transformations "à la main" ?**
> Le Pipeline évite le data leakage (le `fit` du preprocessing est fait uniquement sur les données d'entraînement lors de la cross-validation), simplifie la reproductibilité, et facilite le déploiement (un seul objet à sauvegarder/charger).

**Q4. Quelle est la différence majeure entre PyTorch et TensorFlow aujourd'hui ?**
> Historiquement, PyTorch utilisait un graphe de calcul dynamique (plus flexible, debug facile) contre un graphe statique pour TensorFlow 1.x. Depuis TF2 (eager execution) et Keras 3, les deux sont devenus assez proches en flexibilité, mais PyTorch reste dominant en recherche académique et dans l'écosystème HuggingFace, tandis que TF/Keras garde une forte présence en production et mobile (TFLite).

**Q5. À quoi sert LangChain par rapport à un simple appel d'API LLM ?**
> LangChain ajoute une couche d'orchestration : gestion de la mémoire conversationnelle, chaînage de plusieurs appels (prompt → LLM → parsing → action), intégration d'outils externes (recherche, bases de données), et frameworks prêts pour le RAG — utile dès que l'application dépasse un simple appel "prompt-réponse".
>
> # Guide de Préparation Entretien Data Scientist
## Partie 4 — Overfitting & Underfitting

### 4.1 Définitions

**Overfitting (sur-apprentissage)** : le modèle apprend "par cœur" les données d'entraînement, y compris leur bruit, et perd sa capacité à généraliser sur de nouvelles données. Symptôme : très bonne performance sur le train, mauvaise sur le test/validation.

**Underfitting (sous-apprentissage)** : le modèle est trop simple pour capturer les patterns sous-jacents, même dans les données d'entraînement. Symptôme : mauvaise performance sur train **et** test.

### 4.2 Courbes d'apprentissage (learning curves)

```
Loss
 │
 │  Underfitting          Bon équilibre         Overfitting
 │  ┌─────────┐            ┌─────────┐          ┌─────────┐
 │  │ Train ≈ │            │ Train ↓ │          │ Train ↓↓│
 │  │  Val    │            │ Val ↓   │          │ Val ↑   │
 │  │ (élevés)│            │ (proches│          │ (écart  │
 │  │         │            │  faibles│          │  grandit│
 │  └─────────┘            └─────────┘          └─────────┘
 └────────────────────────────────────────────────────────► Epochs
```

- **Underfitting** : les deux courbes (train et validation) restent élevées et proches.
- **Bon équilibre** : les deux courbes diminuent et convergent vers une faible valeur, avec un écart faible et stable.
- **Overfitting** : la loss train continue de diminuer, mais la loss validation **remonte** après un certain point — c'est le signal classique pour déclencher l'early stopping.

### 4.3 Le compromis Biais-Variance (Bias-Variance Tradeoff)

**Intuition** : imaginez un tireur à l'arc visant une cible.

- **Biais élevé (underfitting)** : toutes les flèches sont groupées, mais loin du centre — le modèle fait des hypothèses trop simplistes et rate systématiquement.
- **Variance élevée (overfitting)** : les flèches sont dispersées tout autour du centre — en moyenne correct, mais chaque tir individuel est imprévisible. Le modèle est trop sensible aux spécificités du jeu d'entraînement.
- **Bon modèle** : flèches groupées près du centre — biais ET variance faibles.

L'erreur totale se décompose ainsi :
```
Erreur totale = Biais² + Variance + Bruit irréductible
```

Augmenter la complexité du modèle réduit généralement le biais mais augmente la variance, et inversement — d'où le "tradeoff" (compromis).

### 4.4 Comment détecter overfitting/underfitting

| Symptôme | Train loss | Validation loss | Diagnostic |
|---|---|---|---|
| Les deux élevées et proches | Élevée | Élevée | **Underfitting** |
| Train très faible, val élevée, écart grandissant | Très faible | Élevée | **Overfitting** |
| Les deux faibles et proches | Faible | Faible | **Bon ajustement** |

### 4.5 Solutions contre l'overfitting

1. **Dropout** : désactive aléatoirement une fraction des neurones à chaque forward pass pendant l'entraînement, empêchant le réseau de "trop dépendre" de neurones spécifiques.
2. **Régularisation L1/L2** : ajoute une pénalité sur la magnitude des poids dans la fonction de perte (L1 favorise la sparsité, L2 réduit uniformément les poids).
3. **Data Augmentation** : génère artificiellement de nouvelles variantes des données d'entraînement (rotation, crop, bruit) pour augmenter la diversité.
4. **Early Stopping** : arrête l'entraînement quand la loss de validation commence à remonter.
5. **Ensembles** : combiner plusieurs modèles (bagging, boosting, voting) réduit la variance globale.

### 4.6 Solutions contre l'underfitting

1. Utiliser un modèle plus complexe (plus de couches/neurones, modèle non-linéaire au lieu de linéaire)
2. Ajouter plus de features pertinentes (feature engineering)
3. Réduire la régularisation (diminuer le coefficient L1/L2, réduire le dropout)
4. Entraîner plus longtemps (plus d'epochs)
5. Vérifier la qualité des données (bruit excessif, labels erronés)

### 4.7 Exemple PyTorch — avec et sans Dropout

```python
import torch
import torch.nn as nn

# Modèle SANS dropout (risque d'overfitting sur petit dataset)
class ModelSansDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


# Modèle AVEC dropout (régularisation)
class ModelAvecDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Dropout(p),       # désactive 50% des neurones aléatoirement
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

# Important : le dropout est actif uniquement en mode entraînement
model = ModelAvecDropout()
model.train()   # dropout ACTIF
# ... entraînement ...
model.eval()    # dropout DÉSACTIVÉ (toutes les unités sont utilisées)
# ... inférence ...
```

### Référence
- Goodfellow, I., Bengio, Y., Courville, A. — *Deep Learning* (MIT Press, 2016), Chapitre 5 "Machine Learning Basics" (biais-variance, régularisation).

---

## Questions d'entretien typiques — Partie 4

**Q1. Comment diagnostiquer un overfitting à partir des courbes d'entraînement ?**
> La loss d'entraînement continue de diminuer alors que la loss de validation stagne puis remonte — l'écart entre les deux courbes se creuse au fil des epochs.

**Q2. Le Dropout fonctionne-t-il pendant l'inférence ?**
> Non. Le dropout est désactivé en mode `eval()` — tous les neurones sont utilisés, généralement avec une mise à l'échelle implicite des poids pour compenser leur "absence" pendant l'entraînement (inverted dropout).

**Q3. Quelle est la différence entre régularisation L1 et L2 ?**
> L1 (Lasso) ajoute la somme des valeurs absolues des poids — elle favorise la sparsité (certains poids deviennent exactement zéro, sélection de features implicite). L2 (Ridge) ajoute la somme des poids au carré — elle réduit uniformément l'amplitude des poids sans les annuler.

**Q4. Un modèle a un train accuracy de 60% et un val accuracy de 58%. Que faire ?**
> C'est un cas d'underfitting (les deux scores sont faibles et proches). Solutions : augmenter la complexité du modèle, ajouter des features, réduire la régularisation, entraîner plus longtemps, vérifier la qualité des données.

**Q5. Pourquoi l'augmentation de données aide-t-elle contre l'overfitting ?**
> Elle augmente artificiellement la diversité et la taille effective du dataset d'entraînement, ce qui rend plus difficile pour le modèle de "mémoriser" des exemples spécifiques et l'encourage à apprendre des patterns plus généraux et invariants (ex : un chat reste un chat même tourné ou recadré).

---

## Partie 5 — Déséquilibre des classes (Imbalanced Data)

### 5.1 Pourquoi c'est un problème

**Exemple typique** : détection de fraude bancaire — sur 100 000 transactions, seulement 100 (0,1%) sont frauduleuses, soit 99,9% de transactions légitimes.

**Le piège** : un modèle qui prédit **toujours "non-fraude"** obtient 99,9% d'accuracy — un score impressionnant mais **totalement inutile**, puisqu'il ne détecte aucune fraude. L'accuracy est une métrique trompeuse dans ce contexte.

Le modèle, en minimisant la loss globale, est naturellement biaisé vers la classe majoritaire car elle domine numériquement le signal d'erreur moyen.

### 5.2 Solutions exhaustives

#### a) Resampling

- **RandomOverSampler** : duplique aléatoirement des exemples de la classe minoritaire
- **RandomUnderSampler** : supprime aléatoirement des exemples de la classe majoritaire
- **SMOTE** (Synthetic Minority Over-sampling Technique) : génère de **nouveaux** exemples synthétiques de la classe minoritaire par interpolation entre voisins existants (plus intelligent que la simple duplication)

```python
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import numpy as np

# Dataset déséquilibré simulé
X = np.random.rand(1000, 5)
y = np.array([0]*950 + [1]*50)  # 95% classe 0, 5% classe 1

print("Avant resampling:", Counter(y))

# SMOTE : génère des exemples synthétiques de la classe minoritaire
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)
print("Après SMOTE:", Counter(y_smote))

# Undersampling
under = RandomUnderSampler(random_state=42)
X_under, y_under = under.fit_resample(X, y)
print("Après undersampling:", Counter(y_under))
```

#### b) Algorithmes robustes : `class_weight='balanced'`

Plutôt que de modifier les données, on modifie la fonction de perte pour pénaliser davantage les erreurs sur la classe minoritaire.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# class_weight='balanced' ajuste automatiquement les poids
# inversement proportionnels aux fréquences des classes
model = RandomForestClassifier(class_weight="balanced", random_state=42)
model_lr = LogisticRegression(class_weight="balanced", max_iter=1000)
```

#### c) Métriques adaptées

Pour un dataset déséquilibré, **ne jamais se fier uniquement à l'accuracy**. Utiliser :
- **F1-score** : moyenne harmonique de précision et rappel
- **ROC-AUC** : capacité à distinguer les classes sur tous les seuils
- **Precision-Recall AUC** : particulièrement informatif quand la classe positive est rare

#### d) Threshold tuning (ajustement du seuil de décision)

Par défaut, un classifieur binaire utilise un seuil de 0.5 pour décider entre les deux classes. Sur des données déséquilibrées, ce seuil n'est souvent pas optimal.

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

# Supposons y_test (vrais labels) et y_proba (probas prédites par le modèle)
y_test = np.array([0,0,0,0,0,0,0,0,0,1,1])
y_proba = np.array([0.1,0.2,0.05,0.3,0.4,0.15,0.25,0.35,0.45,0.6,0.55])

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# Trouver le seuil qui maximise le F1-score
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores)
print(f"Meilleur seuil: {thresholds[best_idx]:.2f}, F1: {f1_scores[best_idx]:.2f}")
```

#### e) Cost-sensitive learning

Approche conceptuellement proche de `class_weight` mais appliquée plus largement : on définit explicitement une **matrice de coûts** reflétant l'impact métier réel de chaque type d'erreur (ex : un faux négatif sur une fraude peut coûter 1000€, un faux positif coûte juste un appel client de vérification — les coûts ne sont pas symétriques).

#### f) Génération synthétique avancée : CTGAN

Pour des données tabulaires complexes, **CTGAN** (Conditional Tabular GAN) génère des lignes synthétiques réalistes en apprenant la distribution conjointe des colonnes (y compris les corrélations), au-delà de la simple interpolation de SMOTE.

```python
# Nécessite : pip install ctgan
from ctgan import CTGAN
import pandas as pd

# data : DataFrame avec colonnes catégorielles et numériques
data = pd.DataFrame({
    "montant": np.random.rand(1000) * 1000,
    "type_transaction": np.random.choice(["achat", "retrait", "transfert"], 1000),
    "fraude": [0]*980 + [1]*20
})

ctgan = CTGAN(epochs=50)
ctgan.fit(data, discrete_columns=["type_transaction", "fraude"])

# Générer 500 nouvelles lignes synthétiques
synthetic_data = ctgan.sample(500)
```

### 5.3 Code complet avec imbalanced-learn (pipeline)

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

X = np.random.rand(2000, 8)
y = np.array([0]*1900 + [1]*100)
np.random.shuffle(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Pipeline imbalanced-learn : SMOTE est appliqué UNIQUEMENT sur le train,
# automatiquement géré pendant fit() — pas de leakage vers le test
pipeline = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
```

**Point critique** : SMOTE ou tout resampling doit être appliqué **uniquement sur les données d'entraînement**, à l'intérieur de la cross-validation/pipeline — jamais sur le test, sous peine de fuite d'information (data leakage) et de scores artificiellement optimistes.

### Référence
- He, H. & Garcia, E. A. (2009) — *"Learning from Imbalanced Data"*, IEEE Transactions on Knowledge and Data Engineering. Référence académique de synthèse sur les techniques de gestion du déséquilibre de classes.

---

## Questions d'entretien typiques — Partie 5

**Q1. Pourquoi l'accuracy est-elle une mauvaise métrique sur données déséquilibrées ?**
> Parce qu'un modèle naïf prédisant toujours la classe majoritaire obtient une accuracy très élevée (ex : 99%) sans avoir aucune capacité de détection de la classe minoritaire, ce qui est inutile pour le cas d'usage (ex : détection de fraude).

**Q2. Quelle est la différence entre SMOTE et RandomOverSampler ?**
> RandomOverSampler duplique des exemples existants de la classe minoritaire (risque de surapprentissage sur ces doublons exacts). SMOTE crée de **nouveaux** exemples synthétiques par interpolation entre des points minoritaires voisins, apportant plus de diversité.

**Q3. Où dans le pipeline doit-on appliquer SMOTE, et pourquoi ?**
> Uniquement sur les données d'entraînement, à l'intérieur de chaque fold de cross-validation. Si on l'applique avant le split, des exemples synthétiques dérivés du test peuvent influencer le train, causant un data leakage et une surestimation des performances.

**Q4. Que signifie `class_weight='balanced'` dans scikit-learn ?**
> Cela ajuste automatiquement le poids de chaque classe dans la fonction de perte, en proportion inverse de sa fréquence — les erreurs sur la classe minoritaire sont pénalisées davantage, sans modifier les données elles-mêmes.

**Q5. Citez deux métriques adaptées à l'évaluation sur données déséquilibrées et expliquez pourquoi.**
> F1-score (moyenne harmonique précision/rappel, sensible aux faux positifs ET faux négatifs sur la classe minoritaire) et PR-AUC (Precision-Recall AUC, qui se concentre sur la performance vis-à-vis de la classe positive rare, contrairement au ROC-AUC qui peut rester optimiste même avec beaucoup de faux positifs si la classe négative est énorme).

---

## Partie 6 — Split des données

### 6.1 Train / Validation / Test : rôles distincts

| Ensemble | Rôle | Utilisation |
|---|---|---|
| **Train** | Apprentissage des paramètres du modèle | Le modèle "voit" ces données et ajuste ses poids/coefficients |
| **Validation** | Réglage des hyperparamètres, sélection de modèle | Le modèle ne s'entraîne pas dessus, mais on l'utilise pour comparer des configurations et détecter l'overfitting |
| **Test** | Évaluation finale, non biaisée | Utilisé **une seule fois**, à la toute fin, pour estimer la performance réelle sur des données jamais vues |

**Analogie** : le train, c'est le cours et les exercices corrigés. La validation, ce sont les examens blancs (qui permettent d'ajuster la méthode de révision). Le test, c'est l'examen final — qu'on ne repasse pas pour "ajuster" sa préparation après coup, sinon le score n'est plus représentatif.

### 6.2 Pourquoi 80/20, 70/15/15 ou 60/20/20 ?

Il n'existe pas de règle universelle — le choix dépend de la **taille totale du dataset** :

- **Petit dataset (quelques milliers d'exemples)** : on privilégie souvent **80/20** (train/test) combiné à de la cross-validation sur le train pour la validation, car on ne peut pas "se permettre" de geler une grande partie des données pour la validation seule.
- **Dataset de taille moyenne** : **70/15/15** ou **60/20/20** — suffisamment de données dans chaque ensemble pour des estimations stables.
- **Très grand dataset (millions d'exemples, deep learning)** : même **98/1/1** peut suffire, car 1% de plusieurs millions d'exemples représente déjà des dizaines de milliers d'observations — largement suffisant pour une estimation fiable de la validation/test.

**Principe général** : plus le dataset est grand, plus on peut réduire la **proportion** allouée à validation/test sans réduire leur **valeur absolue** (nombre d'exemples).

### 6.3 K-Fold Cross Validation et Stratified K-Fold

**K-Fold CV** : on divise le train en K parts ("folds"). On entraîne K fois, chaque fois en utilisant K-1 folds pour l'entraînement et le fold restant pour la validation. On moyenne les K scores obtenus.

**Avantage** : chaque exemple est utilisé à la fois pour l'entraînement et la validation (sur des itérations différentes), donnant une estimation plus robuste et moins dépendante d'un split particulier.

**Stratified K-Fold** : variante qui préserve la **proportion des classes** dans chaque fold — crucial pour les données déséquilibrées (sans stratification, un fold pourrait par hasard ne contenir aucun exemple de la classe minoritaire).

```python
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X = np.random.rand(200, 5)
y = np.array([0]*180 + [1]*20)
np.random.shuffle(y)

model = RandomForestClassifier(random_state=42)

# K-Fold classique
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores_kf = cross_val_score(model, X, y, cv=kf, scoring="f1")
print("K-Fold F1 scores:", scores_kf)

# Stratified K-Fold (préserve la proportion des classes dans chaque fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_skf = cross_val_score(model, X, y, cv=skf, scoring="f1")
print("Stratified K-Fold F1 scores:", scores_skf)
print("Moyenne:", scores_skf.mean(), "± Écart-type:", scores_skf.std())
```

### 6.4 Time Series Split (éviter le data leakage temporel)

Pour les séries temporelles, un split aléatoire est **incorrect** : il permettrait au modèle d'apprendre à partir de données "du futur" pour prédire "le passé", ce qui ne reflète jamais la réalité du déploiement.

`TimeSeriesSplit` garantit que chaque fold de validation est **postérieur temporellement** aux données d'entraînement correspondantes.

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

X = np.arange(100).reshape(-1, 1)  # 100 points temporels ordonnés
y = np.arange(100)

tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold}: train jusqu'à index {train_idx[-1]}, "
          f"val de {val_idx[0]} à {val_idx[-1]}")
```

Sortie typique :
```
Fold 0: train jusqu'à index 19, val de 20 à 36
Fold 1: train jusqu'à index 36, val de 37 à 53
...
```
La fenêtre d'entraînement grandit progressivement, et la validation est toujours "dans le futur" par rapport au train — exactement comme en production réelle.

### 6.5 Danger du data leakage : exemple concret

**Scénario problématique** : un data scientist normalise (standardise) **toutes** les données (train + test ensemble) avec `StandardScaler` AVANT de faire le split.

```python
# ❌ MAUVAISE PRATIQUE — Data leakage
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = np.random.rand(1000, 5) * 100

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # fit sur TOUTES les données !

X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
# Le scaler "connaît" déjà la moyenne et l'écart-type du test set
# => fuite d'information du test vers le train
```

```python
# ✅ BONNE PRATIQUE — fit uniquement sur train
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit SEULEMENT sur train
X_test_scaled = scaler.transform(X_test)         # transform avec les stats du train
```

**Autres formes courantes de data leakage** :
- Feature engineering calculée sur l'ensemble complet (ex : moyenne globale d'une variable cible)
- Imputation de valeurs manquantes basée sur des statistiques globales (toutes données confondues)
- Doublons entre train et test (même client/transaction présent dans les deux ensembles)
- Pour les séries temporelles : utiliser des features qui incluent des informations "futures" par rapport au point prédit

### 6.6 Code scikit-learn complet (split + cross-validation + pipeline)

```python
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.random.rand(500, 6)
y = np.random.randint(0, 2, 500)

# 1) Split initial : isoler le test set (jamais touché avant l'évaluation finale)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2) Pipeline (preprocessing inclus, pour éviter le leakage)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

# 3) Cross-validation sur le train pour estimer la performance / régler les hyperparamètres
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
print("CV scores:", cv_scores, "Moyenne:", cv_scores.mean())

# 4) Entraînement final sur tout le train, évaluation UNIQUE sur le test
pipeline.fit(X_train, y_train)
test_score = pipeline.score(X_test, y_test)
print("Score test final:", test_score)
```

### Référence
- Hastie, T., Tibshirani, R., Friedman, J. — *The Elements of Statistical Learning* (Springer, 2009), chapitres sur l'évaluation et la sélection de modèle.

---

## Questions d'entretien typiques — Partie 6

**Q1. Pourquoi faut-il un ensemble de validation distinct du test ?**
> Si on utilise le test set pour ajuster les hyperparamètres, on "fuite" implicitement l'information du test dans le processus de modélisation — le score final sur ce même test n'est alors plus une estimation non biaisée de la performance en production.

**Q2. Quelle est la différence entre K-Fold et Stratified K-Fold ?**
> K-Fold divise les données en K parts sans tenir compte de la distribution des classes. Stratified K-Fold garantit que chaque fold conserve la même proportion de chaque classe que le dataset global — essentiel pour les données déséquilibrées ou les petits datasets.

**Q3. Pourquoi ne peut-on pas utiliser un split aléatoire classique pour des données temporelles ?**
> Parce que cela permettrait au modèle de s'entraîner sur des données "futures" par rapport à celles qu'il doit prédire en validation, créant un data leakage temporel qui surestime fortement la performance réelle en déploiement (où l'on ne dispose jamais du futur).

**Q4. Donnez un exemple concret de data leakage lié au preprocessing.**
> Appliquer `StandardScaler().fit_transform()` sur l'ensemble des données (train+test) avant le split : le scaler "apprend" la moyenne et l'écart-type incluant les données de test, ce qui introduit une information indirecte du test dans le train.

**Q5. Sur un dataset de 50 millions de lignes, pourquoi un split 98/1/1 peut-il être suffisant ?**
> Parce que 1% de 50 millions représente 500 000 exemples — largement suffisant pour obtenir des estimations statistiquement stables de la performance, tandis que maximiser la part du train (98%) profite directement à l'apprentissage de modèles complexes (deep learning).
>
> # Guide de Préparation Entretien Data Scientist
## Partie 7 — Entraînement des modèles

---

### 7.1 ML classique avec Scikit-learn

#### Pipeline complet : preprocessing → modèle → évaluation

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Données fictives avec valeurs manquantes
df = pd.DataFrame({
    "age": [25, 45, np.nan, 50, 23, 41, 38, 29, 33, 60],
    "salaire": [2500, 5000, 3800, np.nan, 2200, 4700, 3900, 2800, 3100, 7200],
    "ville": ["Paris", "Lyon", "Paris", "Lyon", "Paris", "Lyon", "Paris", "Lyon", "Nice", "Nice"],
    "achat": [0, 1, 0, 1, 0, 1, 1, 0, 0, 1]
})

X = df.drop("achat", axis=1)
y = df["achat"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

numeric_features = ["age", "salaire"]
categorical_features = ["ville"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

full_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", GradientBoostingClassifier(random_state=42))
])

full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)
y_proba = full_pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
```

#### Hyperparameter Tuning : GridSearchCV, RandomizedSearchCV, Optuna

**GridSearchCV** : teste **exhaustivement** toutes les combinaisons d'une grille d'hyperparamètres définie — garantit de trouver le meilleur point de la grille, mais coûteux si la grille est grande.

**RandomizedSearchCV** : échantillonne **aléatoirement** un nombre fixé de combinaisons — souvent presque aussi efficace que GridSearch pour une fraction du coût computationnel.

**Optuna** : optimisation bayésienne moderne, qui apprend des essais précédents pour explorer intelligemment l'espace des hyperparamètres (souvent plus efficace que Random Search pour un budget équivalent).

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

X = np.random.rand(300, 5)
y = np.random.randint(0, 2, 300)

# GridSearchCV : grille exhaustive
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)
grid_search.fit(X, y)
print("Meilleurs paramètres (Grid):", grid_search.best_params_)

# RandomizedSearchCV : échantillonnage aléatoire de distributions
param_dist = {
    "n_estimators": randint(50, 300),
    "max_depth": randint(2, 20),
    "min_samples_split": randint(2, 15)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring="f1",
    random_state=42,
    n_jobs=-1
)
random_search.fit(X, y)
print("Meilleurs paramètres (Random):", random_search.best_params_)
```

```python
# Optuna : optimisation bayésienne
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 15)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    score = cross_val_score(model, X, y, cv=3, scoring="f1").mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print("Meilleurs paramètres (Optuna):", study.best_params)
```

---

### 7.2 Deep Learning avec PyTorch

#### Vue d'ensemble des hyperparamètres

| Paramètre | Rôle | Effet d'une valeur trop élevée | Effet d'une valeur trop faible |
|---|---|---|---|
| **Learning rate (lr)** | Taille du pas de mise à jour des poids | Divergence, loss qui explose ou oscille | Convergence très lente, peut rester bloqué dans un minimum local |
| **Batch size** | Nombre d'exemples par mise à jour | Moins de bruit dans le gradient mais généralisation parfois moins bonne, plus de mémoire GPU | Gradient plus bruité (parfois bénéfique pour échapper aux minima locaux), entraînement plus lent |
| **Epochs** | Nombre de passages complets sur le dataset | Risque d'overfitting | Underfitting (modèle pas assez entraîné) |
| **Optimizer** | Algorithme de mise à jour des poids | — | — |
| **Loss function** | Mesure l'écart entre prédiction et vérité | Mauvais choix → signal d'apprentissage incohérent avec la tâche | idem |
| **Scheduler** | Ajuste le learning rate pendant l'entraînement | — | — |

#### Comment jouer avec chaque paramètre

- **Learning rate** : c'est l'hyperparamètre le plus impactant. Trop élevé → la loss diverge ou oscille violemment ; trop faible → convergence extrêmement lente. On le règle généralement en premier via un *LR finder* (voir ci-dessous).
- **Batch size** : des batchs plus grands stabilisent le gradient mais nécessitent souvent d'augmenter proportionnellement le learning rate (règle empirique de mise à l'échelle linéaire). Des batchs plus petits introduisent du bruit qui peut aider à généraliser mais ralentissent l'entraînement (moins de parallélisme GPU).
- **Epochs** : à combiner systématiquement avec l'**early stopping** plutôt que de fixer un nombre arbitraire.
- **Optimizer** : `SGD` (avec momentum) généralise parfois mieux mais converge plus lentement ; `Adam`/`AdamW` convergent plus vite et sont le choix par défaut moderne, surtout pour les Transformers (`AdamW` corrige un bug de couplage entre weight decay et learning rate présent dans `Adam`).
- **Loss function** : doit correspondre à la tâche — `CrossEntropyLoss` pour la classification multi-classe, `BCEWithLogitsLoss` pour la classification binaire/multi-label, `MSELoss`/`L1Loss` pour la régression.
- **Scheduler** : ajuste dynamiquement le learning rate pendant l'entraînement (souvent en le diminuant progressivement pour affiner la convergence vers la fin).

#### Learning Rate Finder, Warm-up, Cosine Annealing

**Learning Rate Finder** (popularisé par fast.ai) : on entraîne le modèle sur quelques itérations en augmentant exponentiellement le learning rate (de très faible à très élevé), et on trace la loss en fonction du LR. On choisit un LR juste avant le point où la loss commence à exploser.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
criterion = nn.MSELoss()

lrs, losses = [], []
lr = 1e-6
optimizer = optim.Adam(model.parameters(), lr=lr)

X = torch.randn(64, 10)
y = torch.randn(64, 1)

for i in range(100):
    optimizer.param_groups[0]["lr"] = lr
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    lrs.append(lr)
    losses.append(loss.item())
    lr *= 1.1  # augmentation exponentielle du LR

plt.plot(lrs, losses)
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("LR Finder")
plt.savefig("lr_finder.png")
```

**Warm-up** : démarrer l'entraînement avec un learning rate très faible, puis l'augmenter progressivement jusqu'à la valeur cible sur les premières itérations/epochs. Cela stabilise l'entraînement au début, lorsque les poids sont encore aléatoires et les gradients potentiellement très instables — particulièrement important pour les Transformers.

**Cosine Annealing** : fait décroître le learning rate selon une courbe en cosinus, de la valeur initiale vers (presque) zéro, souvent avec des "redémarrages" périodiques (cosine annealing with warm restarts) pour aider à explorer différents minima.

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import torch.optim as optim

model = nn.Linear(10, 1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Cosine annealing : LR décroît en cosinus sur 50 epochs
scheduler = CosineAnnealingLR(optimizer, T_max=50)

# Warm-up linéaire sur les 5 premières epochs, combiné manuellement
def warmup_then_cosine(epoch, warmup_epochs=5, total_epochs=50):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

scheduler_warmup = LambdaLR(optimizer, lr_lambda=lambda epoch: warmup_then_cosine(epoch))

for epoch in range(50):
    # ... boucle d'entraînement ...
    scheduler.step()
```

#### Différence fondamentale entre entraîner un ML classique vs un DL

| Aspect | ML classique (ex : Random Forest) | Deep Learning |
|---|---|---|
| **Optimisation** | Souvent pas de gradient (ex : arbres construits par règles de split), ou optimisation convexe simple (régression linéaire/logistique) | Descente de gradient **stochastique** (SGD/Adam) sur une fonction de perte fortement non-convexe |
| **Mécanisme d'apprentissage** | Pas de rétropropagation | **Backpropagation** : calcul des gradients de la loss par rapport à chaque poids via la règle de chaîne, couche par couche, de la sortie vers l'entrée |
| **Features** | Souvent conçues manuellement (feature engineering) | Apprises automatiquement, couche par couche (représentations hiérarchiques) |
| **Données nécessaires** | Peut bien fonctionner avec peu de données | Nécessite généralement beaucoup de données pour bien généraliser |
| **Matériel** | CPU souvent suffisant | GPU/TPU quasi indispensables pour des tailles raisonnables |

**Gradient Descent & Backpropagation — intuition** : imaginez que vous êtes dans le brouillard sur une montagne et que vous voulez descendre vers la vallée (minimiser la loss). À chaque pas, vous regardez la pente sous vos pieds (le gradient) et faites un petit pas dans la direction la plus pentue vers le bas. La **backpropagation** est la méthode efficace pour calculer "la pente" (le gradient) par rapport à **chaque poids du réseau**, en propageant l'erreur de la sortie vers l'entrée via la règle de dérivation en chaîne (chain rule).

#### Exemple complet : entraînement sur MNIST

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Données
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. Modèle
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 classes (chiffres 0-9)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MNISTNet().to(device)

# 3. Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

# 4. Boucle d'entraînement avec early stopping
best_val_loss = float("inf")
patience, patience_counter = 3, 0

for epoch in range(10):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()

    # Évaluation
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(test_loader)
    accuracy = correct / total

    print(f"Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}, "
          f"val_loss={val_loss:.4f}, val_acc={accuracy:.4f}, "
          f"lr={scheduler.get_last_lr()[0]:.6f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping déclenché.")
            break
```

### Référence
- [fast.ai](https://www.fast.ai) — *"Practical Deep Learning for Coders"*, particulièrement les leçons sur le learning rate finder (issu de Leslie Smith, *"Cyclical Learning Rates for Training Neural Networks"*, ArXiv 1506.01186).
- Documentation officielle : [pytorch.org/docs](https://pytorch.org/docs), [scikit-learn.org](https://scikit-learn.org), [optuna.org](https://optuna.org)

---

## Questions d'entretien typiques — Partie 7

**Q1. Quelle est la différence entre GridSearchCV et RandomizedSearchCV ?**
> GridSearchCV teste toutes les combinaisons possibles d'une grille définie (exhaustif mais coûteux). RandomizedSearchCV échantillonne un nombre fixe de combinaisons aléatoirement dans des distributions définies — souvent presque aussi performant pour une fraction du coût, surtout quand certains hyperparamètres ont peu d'impact.

**Q2. Pourquoi utilise-t-on `AdamW` plutôt que `Adam` pour entraîner des Transformers ?**
> `AdamW` découple correctement la régularisation weight decay de la mise à jour adaptative du gradient, corrigeant un défaut d'`Adam` original où le weight decay interagissait incorrectement avec les moments adaptatifs — ce qui améliore la généralisation, particulièrement importante pour les grands modèles.

**Q3. Qu'est-ce que le "warm-up" du learning rate et pourquoi est-il utile ?**
> C'est une phase initiale où le learning rate augmente progressivement depuis une valeur très faible jusqu'à sa valeur cible. Elle évite des mises à jour de poids trop violentes au tout début de l'entraînement, quand les gradients peuvent être très instables (poids initialisés aléatoirement) — particulièrement critique pour les architectures Transformer profondes.

**Q4. Expliquez intuitivement la backpropagation.**
> C'est l'application répétée de la règle de dérivation en chaîne pour calculer, de manière efficace, la dérivée de la loss par rapport à chaque poids du réseau. L'erreur calculée à la sortie est "propagée en arrière" couche par couche, chaque couche utilisant le gradient reçu de la couche suivante pour calculer le sien, sans recalculer chaque dérivée depuis zéro.

**Q5. Que se passe-t-il si le learning rate est trop élevé ? Trop faible ?**
> Trop élevé : les mises à jour de poids "dépassent" le minimum à chaque étape, la loss peut osciller violemment ou diverger (devenir NaN). Trop faible : la convergence est extrêmement lente, et le modèle peut rester bloqué dans un minimum local ou un plateau pendant un budget d'entraînement limité.
