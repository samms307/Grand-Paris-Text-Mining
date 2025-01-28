# **Classification des Propositions pour la Métropole du Grand Paris**

## **Contexte**
Dans le cadre de la campagne *"Construisons la Métropole du Grand Paris"*, la **Direction de la Démocratie, des Citoyens et des Territoires (DDCT)** de la Ville de Paris a sollicité la participation des citoyens et collectivités. Ces derniers ont `soumis des idées innovantes` pour transformer la métropole en un espace plus connecté durable et inclusif. 

Ces propositions d'idées ont éte regroupées en **5 grandes thématiques**:

1. **Transition écologique et Mobilités**
2. **Culture et Identité**
3. **Logement et Aménagement**
4. **Rayonnement**
5. **Lutte contre les inégalités**

### **Jeu de données**
Les propositions ont été collectées dans un jeu de données comprenant **362 propositions** et **21 attributs**. Parmi ces attributs on trouve :
- **Date de publication**
- **Objectif** (résultats attendus)
- **Description textuelle** (détails des proposition


# **Objectifs du Projet**
L’objectif est d’appliquer une **classification automatique** sur le contenu des idées afin d’identifier les termes les plus représentatifs pour chaque thématique. 


## **Outils utilisés**
- **PySpark** est une interface Python d'Apache Spark un framework open-source conçu pour le traitement distribué de grandes quantités de données. **Notre objectifs** :  
   - Manipuler et analyser des données à l'aide de Spark SQL et des DataFrames.  
   - Appliquer des algorithmes de machine learning via la bibliothèque MLlib.  

- **Spark NLP**  
Bibliothèque de traitement automatique du langage naturel (NLP) basée sur Spark et pour optimisée pour les pipelines scalables et le traitement parallèle.  **Notre objectifs** :  
   - Prétraiter et analyser des textes (tokenisation, lemmatisation, annotation).  
   - Construire des pipelines NLP performants et adaptés aux grandes volumétries.  
   - Exploiter des modèles NLP pré-entraînés pour des tâches spécifiques.  

---------------------


# 📈 **Étapes Clés du Projet**

### **1️⃣ Prétraitement des données textuelles**
Pour préparer les données textuelles à la vectorisation nous avons utilisé la bibliothèque **Spark NLP** développée par **John Snow Labs**. Nous avons construit un `pipeline NLP composé de plusieurs Annotators` dont le principal objectif est de nettoyer la variable texte avant la vectorisation.  
**Les étapes principales incluent** :  

- **Normalisation et Nettoyage** : Uniformiser le texte en transformant les mots en minuscules en supprimant les accents ainsi la ponctuation et les caractères spéciaux non pertinents pour l'analyse.  
- **Tokenisation** : Découpage du texte en unités de base de mots dans notre cas.  
- **Lemmatisation** : Réduction des mots à leur forme canonique ou racine (par exemple, "mangeant" devient "manger").  
- **Correction orthographique (SpellChecker)** : Correction des fautes d'orthographe dans les textes. Nous avons enrichi cette étape avec un fichier texte personnalisé nommé `correction_mots` contenant des mots spécifiques à corriger.  
- **Suppression des Stop Words** : Élimination des mots courants qui n'apportent pas de valeur sémantique significative. Cette étape a également été enrichie avec un fichier texte nommé `french`, contenant des mots supplémentaires à supprimer.

Ces étapes de prétraitement sont essentielles pour améliorer la qualité des données textuelles avant de les soumettre aux modèles de vectorisation.

--------------
### 2️⃣ Représentation Vectorielle

L'objectif ici est de transformer les propositions d'idées en données numériques afin de pouvoir les représenter graphiquement ou les analyser avec des modèles de machine learning. 
Pour ce faire nous utilisons `les transformeurs` de **Spark NLP** et **Spark MLlib**. Ces outils prennent les données annotées (nettoyées) vues précédemment et appliquent des transformations pour générer des représentations vectorielles à l'aide de modèles tels que :

**Modèle de Sac de Mots Statistique : TF-IDF avec et sans LSA**
Le modèle **TF-IDF** a été utilisé pour pondérer les termes des propositions, mettant en avant les mots significatifs et réduisant l'impact des termes fréquents. 
Nous avons également testé **LSA (Latent Semantic Analysis)**, qui permet de réduire la dimensionnalité tout en capturant les relations sémantiques entre les termes.

**Modèles Basés sur les Embeddings et Réseaux de Neurones Profonds**
Pour ces modèles, nous avons utilisé des **modèles pré-entraînés** spécifiquement conçus pour le français. Nous avons exploré deux approches principales :

**Word Embeddings :**
- **Word2Vec** : Utilisé pour générer des représentations vectorielles des mots, capturant leurs relations de similarité.
- **BERT** : Produit des **embeddings contextuels**, où la représentation des mots varie selon leur contexte dans une phrase.

###### **Document/Sentence Embeddings :**
- **Agrégation par la moyenne des word embeddings** : Cette méthode calcule la moyenne des vecteurs des mots d'une proposition entière, offrant une représentation globale.
- **USE (Universal Sentence Encoder)** : Génère des embeddings pour des phrases entières, capturant ainsi le sens global d'une proposition.

------------------------------
-----------------------------

##  **Étapes Méthodologiques**  

## 1. Exploration des techniques de vectorisation
Nous avons exploré différentes méthodes pour transformer les propositions textuelles en vecteurs numériques exploitables pour le **clustering**. Parmi les techniques étudiées :

#### **1.1 Modèle de sac de mots statistique : TF-IDF avec et sans LSA**  
Le modèle **TF-IDF** a été utilisé pour pondérer les termes des propositions, mettant en avant les mots significatifs et réduisant l'impact des termes fréquents. Nous avons également testé **LSA**, qui permet de réduire la dimensionnalité tout en capturant les relations sémantiques entre les termes.

### **1.2 Modèles basés sur les embeddings et réseaux de neurones profonds**  
Pour ces modèles, nous avons utilisé des **modèles pré-entraînés** spécifiquement conçus pour le français. Nous avons exploré deux approches principales :  

#### **Word Embeddings**  
- **Word2Vec** : Utilisé pour générer des représentations vectorielles des mots, capturant leurs relations de similarité.  
- **BERT** : Produit des **embeddings contextuels**, où la représentation des mots varie selon leur contexte dans une phrase.

#### **Document/Sentence Embeddings**  
- **Agrégation par la moyenne des word embeddings** : Cette méthode calcule la moyenne des vecteurs des mots d'une proposition entière, offrant une représentation globale.  
- **USE (Universal Sentence Encoder)** : Génére des embeddings pour des phrases entières, capturant ainsi le sens global d'une proposition.


## 2️⃣ Représentation Vectorielle

L'objectif ici est de transformer les propositions d'idées en données numériques, afin de pouvoir les représenter graphiquement ou les analyser avec des modèles de machine learning. Pour ce faire, nous utilisons les transformeurs de **Spark NLP** et **Spark MLlib**. Ces outils prennent les données annotées (nettoyées) vues précédemment et appliquent des transformations pour générer des représentations vectorielles à l'aide de modèles tels que :

### **Modèle de Sac de Mots Statistique : TF-IDF avec et sans LSA**

Le modèle **TF-IDF** a été utilisé pour pondérer les termes des propositions, mettant en avant les mots significatifs et réduisant l'impact des termes fréquents. Nous avons également testé **LSA (Latent Semantic Analysis)**, qui permet de réduire la dimensionnalité tout en capturant les relations sémantiques entre les termes.

### **Modèles Basés sur les Embeddings et Réseaux de Neurones Profonds**

Pour ces modèles, nous avons utilisé des **modèles pré-entraînés** spécifiquement conçus pour le français. Nous avons exploré deux approches principales :

#### **Word Embeddings :**
- **Word2Vec** : Utilisé pour générer des représentations vectorielles des mots, capturant leurs relations de similarité.
- **BERT** : Produit des **embeddings contextuels**, où la représentation des mots varie selon leur contexte dans une phrase.

#### **Document/Sentence Embeddings :**
- **Agrégation par la moyenne des word embeddings** : Cette méthode calcule la moyenne des vecteurs des mots d'une proposition entière, offrant une représentation globale.
- **USE (Universal Sentence Encoder)** : Génère des embeddings pour des phrases entières, capturant ainsi le sens global d'une proposition.






Contrairement aux annotateurs, les transformateurs sont souvent utilisés pour appliquer ou enregistrer des modèles entraînés ou pour combiner plusieurs étapes de traitement.
Ils opèrent sur des DataFrames et produisent également des DataFrames enrichis.
Exemples de transformateurs :
PipelineModel : Un pipeline complet contenant plusieurs étapes (annotateurs et transformateurs).
EmbeddingsFinisher : Convertit des embeddings NLP en colonnes exploitables (par exemple, pour des modèles de machine learning).
LightPipeline : Permet une exécution rapide et optimisée d'un pipeline complet sur des petits jeux de données ou des chaînes brutes.


Après le prétraitement les données textuelles sont prêtes à être transformées en **vecteurs numériques**. Ces vecteurs permettent aux modèles de machine learning de traiter les textes sous une forme numérique.
Nous avons exploré différentes méthodes pour transformer les propositions textuelles en vecteurs numériques exploitables pour le **clustering**. Parmi les techniques étudiées :

Le deuxième pipeline consiste à faire plusieurs représentations vectorielles des documents 
utilisées en traitement de langage naturel pour représenter les documents sous forme de 
vecteurs numériques. Plus précisément ces traitements se feront à l’aide de la bibliothèque
SPARK NLP qui contient des fonctions de transformateurs afin de représenter notre variable 
texte sous forme de vecteur numérique.



Les techniques de représentation vectorielle utilisées sont les suivantes :

- **TF-IDF** : Pondération des mots dans les propositions en fonction de leur fréquence et de leur importance relative dans le corpus.
- **Word Embeddings** : Utilisation de modèles pré-entraînés comme **Word2Vec** et **BERT** pour représenter les mots sous forme de vecteurs dans un espace continu.
- **Sentence Embeddings** : Utilisation de modèles comme **USE (Universal Sentence Encoder)** pour obtenir des représentations vectorielles des phrases entières.





### 2️⃣ **Modélisation**  
Après avoir prétraité les données et éliminé tout risque de data leakage, le notebook suivant se concentre sur la construction et l’évaluation des modèles. Voici les points clés abordés dans cette phase :

- **Sélection du modèle :** Nous avons comparé plusieurs modèles, à commencer par la régression logistique et en incluant des modèles plus complexes comme Random Forest et LightGBM, adaptés aux données déséquilibrées. Les performances ont été évaluées via la métrique AUC-ROC, en calculant la moyenne des scores et l'écart-type pour mesurer la stabilité. Pour assurer une évaluation robuste, nous avons utilisé une validation croisée stratifiée, garantissant que la proportion des classes reste constante dans chaque pli, ce qui évite tout biais dans l'entraînement et permet au modèle de mieux généraliser.

- **Optimisation du seuil de décision :**  L’ajustement du seuil de probabilité a été effectué pour optimiser la classification des défauts de paiement, en tenant compte des erreurs de classification (faux négatifs et faux positifs), qui peuvent avoir un impact financier significatif pour l'entreprise.  
  Deux approches ont été explorées :  
  1. **Maximisation de la sensibilité et de la spécificité :** Trouver un seuil qui équilibre les faux positifs et faux négatifs pour améliorer la performance globale et minimiser les pertes.  
  2. **Optimisation de la précision et du rappel :** Prioriser la détection des clients risqués, en particulier pour minimiser les faux négatifs, afin de réduire les risques financiers et améliorer la gestion du crédit.

  
- **Explicabilité du modèle :**  Le modèle **LightGBM** a été utilisé pour les prédictions, et pour en comprendre les décisions, nous avons appliqué **LIME** pour expliquer chaque prédiction (ex : refus de prêt). L'**importance des caractéristiques** a permis d'identifier les variables influentes globalement (comme le revenu et l'historique de crédit). Ces méthodes assurent que les décisions du modèle sont compréhensibles et justifiables, répondant ainsi aux exigences réglementaires.


- **Détection du Data Drift :**
Pour assurer la performance continue du modèle en production, nous avons surveillé l'évolution des données avec la `bibliothèque Evidently`. Evidently permet de détecter le Data Drift en comparant les distributions des données d'entrée en production avec celles des données d’entraînement. Cette surveillance du drift des données aide à identifier des écarts significatifs dans les caractéristiques des données, garantissant ainsi que le modèle continue à fournir des prédictions fiables même lorsque les données changent avec le temps.
[Pour plus de détails sur la détection du data drift, vous pouvez consulter le rapport d'analyse de dérive des données ici](https://github.com/samms307/scoring_client_api/blob/main/report_datadrift.html)



👉 **[Voir le notebook de modélisation pour l'interprétation des résultats](https://github.com/samms307/scoring_client_api/blob/main/Final_Mod%C3%A9lisation.ipynb)**











-------------------------------------------------------------------------------------------






a moyenne des word embding uilisé precedament
    - USE qui donne des phrases

Nous avons également exploré des modèles d'embeddings statiques et dynamiques comme **Word2Vec**, **BERT**, et **USE**. Ces modèles ont permis de capturer des relations sémantiques plus profondes entre les mots et les propositions. Les performances de ces modèles sont comparées dans le [notebook](./notebook/comparaison_modeles.ipynb).


1. **Exploration des techniques de vectorisation**  
Nous avons examiné plusieurs méthodes pour représenter les propositions d'idées sous forme vectorielle afin d'utiliser nos algorithmes d’apprentissage automatique :

1.1 Modèle de sac de mots statistique : TF-IDF (avec et sans réduction par LSA)

Pour analyser et classifier les propositions d’idées, nous explorons différentes méthodes de représentation textuelle. Ces méthodes transforment les descriptions textuelles en vecteurs, permettant leur traitement par des algorithmes d’apprentissage automatique. 
Tout les modeles pre entrainé ont ete choisi pour la langue française 

1.1 Modèle de sac de mots statistique : TF-IDF (avec et sans réduction par LSA)
1.2 Modèles basés sur les embeddings : Nous utilisons le modèle pré-entraîné **"w2v_cc_300d"** concus pour la langue francaise, qui génère des embeddings de 300 dimensions.
Nous allons explorer et comparer différentes méthodes pour transformer du texte en représentations vectorielles adaptées à des tâches NLP afin de trouver la meilleur represnttion pour un clustering afin d'extraire automatique les termes

Méthodes couvertes :

2 méthodes de représentation :
 Nous examinerons plusieurs méthodes pour représenter les propositions d'idées sous forme vectorielle notamment :  
• le word embedding
(ou « plongement de mots »)
• le doc embedding
(ou « plongement de documents »)

TF-IDF (avec et sans réduction par LSA)
Word2Vec (pré-entraîné, moyenne des vecteurs)
BERT (pré-entraîné, agrégation des embeddings)
USE (Universal Sentence Encoder via TensorFlow Hub)
  


   ### **Phase 1 : Extraction des Termes Pertinents**

L’objectif principal de cette phase est d’appliquer une **classification automatique** sur le contenu des idées afin d’identifier les termes les plus pertinents et représentatifs pour chaque thématique. En d'autres termes, il s'agit de repérer automatiquement les mots ou expressions qui **caractérisent** efficacement chaque thématique, en s'appuyant sur les descriptions textuelles des propositions.

2. **Combinaison des approches non supervisées et supervisées**  
   - **Non supervisé (exploration) :**  
     Les techniques comme le clustering ou l’analyse thématique (exemple : LDA) seront utilisées pour identifier des regroupements naturels dans les données et extraire les termes fréquents ou récurrents dans chaque groupe. Cette étape permet de fournir une vue d’ensemble des thématiques et de détecter les termes généraux liés à chaque catégorie.
    Elle offre une analyse globale des données en regroupant les idées similaires, mais produit des résultats parfois bruts ou trop larges. Par exemple, certains termes fréquents dans un cluster peuvent être peu spécifiques ou apparaître dans plusieurs thématiques.

   - **Supervisé (affinage) :**  
     Les modèles supervisés, comme les SVM, les réseaux de neurones ou la régression logistique, exploiteront les données étiquetées (les thématiques connues) pour attribuer un score d'importance aux termes. Ces modèles permettront d’identifier les mots ou expressions les plus discriminants, c’est-à-dire ceux qui différencient efficacement une thématique d’une autre.

----------------
Dans un premier temps, nous allons extraire automatiquement les termes clés et les concepts les plus pertinents de chaque proposition en fonction de sa thématique. Nous avons choisi d'utiliser **PySpark** et **Spark NLP** plutôt que des méthodes plus simples comme **TF-IDF**, **Word2Vec**, **GloVe** ou **BERT** pour plusieurs raisons techniques :

1. **Problèmes avec TF-IDF** :  
   Le modèle **TF-IDF** calcule l'importance des termes en fonction de leur fréquence dans chaque document par rapport à l'ensemble du corpus. Cependant, il ne prend pas en compte le contexte sémantique des mots ni leur relation avec la thématique de chaque proposition. Cela pourrait conduire à l'extraction de termes peu pertinents ou non représentatifs des thématiques principales (par exemple, "énergie" dans le cadre de la mobilité).

2. **Limitations de Word2Vec, GloVe et BERT** :  
   - **Word2Vec** et **GloVe** génèrent des vecteurs de mots en fonction de leur contexte global, mais ne captent pas toujours les nuances spécifiques d'un texte lié à une thématique particulière. De plus, ils ne sont pas directement conçus pour extraire les termes les plus pertinents pour un thème donné, ce qui rend leur utilisation moins efficace dans ce contexte.
   - **BERT** est très performant pour des tâches contextuelles complexes, mais son utilisation pour l'extraction de termes clés nécessite un fine-tuning ou un prétraitement supplémentaire. De plus, il peut être coûteux en termes de ressources computationnelles, ce qui n'est pas optimal pour un projet à grande échelle comme celui-ci.

3. **Pourquoi PySpark et Spark NLP ?**  
   Ces outils sont spécifiquement conçus pour traiter de grandes quantités de données tout en permettant une analyse sémantique plus fine. Par exemple, **Spark NLP** offre des fonctionnalités comme la **Reconnaissance d'Entités Nommées (NER)** et l'analyse syntaxique, qui permettent de capturer non seulement les termes pertinents mais aussi les entités et les relations spécifiques à chaque thématique. De plus, **PySpark** permet d'exécuter ces analyses à grande échelle, ce qui est essentiel pour traiter les 362 propositions du jeu de données.

Cette approche garantit une extraction plus précise et adaptée aux thématiques spécifiques des propositions, tout en optimisant les ressources et le temps de traitement.


----------

### **Phase 1 : Extraction des Termes Pertinents**
Dans un premier temps, nous allons extraire automatiquement les termes clés et les concepts les plus pertinents de chaque proposition en fonction de sa thématique. Pour ce faire, nous utiliserons **PySpark** et **Spark NLP**, deux outils puissants permettant de traiter efficacement des volumes de données importants et d'exécuter des analyses textuelles de manière distribuée.

### **Phase 2 : Résumé Automatique des Propositions**
Dans un second temps, nous allons appliquer des techniques de résumé automatique pour condenser les descriptions textuelles en résumés courts. Deux méthodes seront explorées :
1. **Méthode extractive** : Sélection des phrases les plus représentatives des propositions.
2. **Méthode abstractive** : Génération de résumés en reformulant les idées principales de manière plus concise.


---------------------
### **Structure des Données**
Le jeu de données comprend **362 propositions**, chacune décrite par **21 attributs**. Ces attributs incluent notamment :
- **Date de publication** : Pour situer chronologiquement les idées.  
- **Objectif** : Une indication des résultats attendus.  
- **Description textuelle** : Une explication détaillée de chaque proposition.  

## **Objectif**
Ce jeu de données constitue une opportunité d’appliquer des techniques avancées d’analyse textuelle et dapprendre de nouvelle outil :

Dans un premier temps va etre de cherchez les termes issus de ces descriptions textuelles qui sont les plus pertinents pour chaque thématique de maniere automatique Cela se fera a l'aide pypsark et spark nlp 
Ensuite dans un second temps va etre de faire des résumer f'information selon deux méthodes 

- Extraire des mots-clés et tendances émergentes.  
- Classifier les propositions selon les thématiques principales.  
- Fournir une vision globale et exploitable pour guider les prises de décision.

## **Analyse des Contributions pour le Grand Paris**

### **Problématique**
Le volume et la diversité des données textuelles provenant des propositions soumises par les citoyens et collectivités rendent leur analyse manuelle difficile et inefficace. Cependant, ces données contiennent des **insights** stratégiques essentiels, tels que :
- **Les tendances émergentes** dans les thématiques principales (par exemple, l'augmentation des idées sur la transition écologique),
- **Les besoins spécifiques** exprimés par les citoyens ou collectivités (tels que des demandes de logements sociaux ou d'écoquartiers),
- **Les priorités stratégiques** pour orienter les politiques publiques (comme la nécessité de solutions de mobilité durable).


L'enjeu est donc de traiter efficacement ces données pour extraire des informations pertinentes et organiser les idées de manière à répondre aux objectifs du projet.

### **Objectifs de l'Analyse**
L’analyse des contributions vise à :
1. **Extraire des informations clés** des propositions, telles que les mots-clés, thèmes récurrents ou tendances émergentes.
   - Exemple : Identifier que les propositions liées au logement mentionnent souvent des concepts comme "écoquartiers" ou "logement social".
   
2. **Organiser les propositions** en les classifiant ou en les regroupant selon les thématiques définies : 
   - Transition écologique, Mobilité, Logement, Rayonnement, et Lutte contre les inégalités.
   - Exemple : Catégoriser automatiquement une proposition comme relevant de la "Transition écologique" lorsqu'elle mentionne des énergies renouvelables ou des pratiques durables.

3. **Interpréter les données** pour mettre en évidence les priorités et besoins exprimés par les participants.
   - Exemple : Identifier une forte demande pour des solutions de "mobilité douce" dans la thématique "Transition écologique et Mobilité".

4. **Fournir des résultats exploitables** pour orienter les décisions stratégiques des décideurs publics.
   - Exemple : Identifier que l’"aménagement urbain durable" est une préoccupation majeure pour de nombreux participants.

### **Méthodes et Techniques utilisées**

1. **Extraction des mots-clés et des tendances émergentes :**
   - **Traitement du langage naturel (NLP)** avec des techniques comme **TF-IDF** ou **Word2Vec** pour extraire des termes significatifs et identifier les thématiques dominantes.
   - Utilisation des **embeddings** pour capturer les relations sémantiques entre les termes et détecter des tendances basées sur la fréquence et le contexte des mots.

2. **Classification des propositions selon les thématiques :**
   - **Classification supervisée** : Modèles comme **SVM**, **Random Forests**, ou **réseaux de neurones** pour assigner chaque proposition à l’une des cinq thématiques principales en fonction du texte.
   - **Méthodes non supervisées** : 
     - **Clustering (K-means, DBSCAN)** pour regrouper les propositions similaires sans préétiquetage.
     - **LDA (Latent Dirichlet Allocation)** pour découvrir des thèmes latents dans les données, en groupant les propositions selon leur contenu.

3. **Fournir des visualisations interactives pour les décisions stratégiques :**
   - Création de **visualisations** avec **matplotlib**, **seaborn**, ou **Plotly** pour permettre aux décideurs de visualiser les tendances émergentes, ainsi que les priorités exprimées par les citoyens.
   - Développement de **tableaux de bord dynamiques** avec des outils comme **Dash** ou **Streamlit**, pour offrir une exploration intuitive des données et faciliter la prise de décisions stratégiques basées sur les résultats de l’analyse.

---

## 1. Extraction des Informations Clés (Mots-clés, Thèmes, Tendances Émergentes)

Cette étape est cruciale pour structurer les données et en tirer des insights exploitables. Voici les différentes approches que vous pouvez adopter :

### Mots-clés :
- Utilisation de techniques comme **TF-IDF** ou **Word2Vec** pour identifier les termes les plus significatifs dans les descriptions de propositions. 
- Ces mots-clés peuvent refléter les préoccupations principales, telles que "énergies renouvelables", "logement social", ou "mobilité durable".
- Ces mots-clés peuvent ensuite être utilisés pour classer ou regrouper les propositions selon les thématiques, ou pour détecter des tendances émergentes.

### Thèmes Récurrents :
- Une méthode comme **LDA** (Latent Dirichlet Allocation) peut aider à identifier des thèmes latents dans les textes.
- Par exemple, **LDA** pourrait révéler que plusieurs propositions se concentrent sur des thèmes tels que la transition écologique, l'inclusion sociale, ou l'urbanisme durable.

### Tendances Émergentes :
- Une analyse de fréquence des mots-clés au fil du temps pourrait mettre en évidence des tendances, comme une montée d’intérêt pour certains sujets (par exemple, un intérêt accru pour la "mobilité douce" ou les "zones piétonnes").
- Vous pouvez également observer des **co-occurrences** de termes pour détecter des relations entre différentes thématiques (par exemple, "logement" et "biodiversité" dans un même contexte).

## 2. Résumé Automatique des Textes

Après avoir extrait des informations clés, il peut être utile de générer des résumés des propositions pour faciliter la compréhension globale et la prise de décision. Cela est particulièrement important quand il y a un grand volume de données.

### Résumé Extractif :
- Cette approche consiste à extraire les phrases ou parties les plus pertinentes des textes originaux pour créer un résumé.
- Il peut être utile si vous souhaitez garder la structure des propositions, mais en réduisant leur longueur.
- **Techniques :** TF-IDF, TextRank, ou BERT pour extraire les phrases les plus significatives du texte.

### Résumé Abstratif :
- Cette méthode génère un résumé en reformulant ou en paraphrasant le texte original, ce qui permet de créer un résumé plus fluide et plus cohérent.
- Cela peut être particulièrement utile pour des textes plus longs et complexes.
- **Techniques :** Transformers comme BERT ou GPT peuvent être utilisés pour générer des résumés plus naturels et moins mécaniques.

### Complémentarité des Deux Méthodes :
Les deux méthodes peuvent être complémentaires et se renforcer :
- L'**extraction des informations clés** peut servir à réduire le bruit et à identifier les éléments les plus pertinents dans chaque proposition (mots-clés, thèmes, etc.).
- Ensuite, les **résumés de texte** peuvent être utilisés pour condenser les propositions tout en conservant leur essence, facilitant ainsi l'analyse globale.

**Exemple :**
- Vous pourriez commencer par extraire des mots-clés pour chaque proposition afin de comprendre rapidement de quoi elle parle.
- Ensuite, vous pourriez générer un résumé pour chaque proposition afin de la rendre plus lisible tout en gardant les informations stratégiques intactes.

Ces deux approches — **extraction des informations clés** et **résumé de texte** — sont très pertinentes pour votre projet. L'extraction des informations clés vous aide à organiser et structurer les données, tandis que le résumé de texte améliore la lisibilité et la compréhension générale des propositions. Ensemble, elles vous permettront d’offrir une analyse complète et efficace des contributions pour le projet "Grand Paris", tout en facilitant la prise de décisions basées sur des données textuelles volumineuses.


--------------------------------

### 1. Choix des métriques pour KMeans avec étiquettes connues

Pour évaluer un modèle KMeans sur des données textuelles où vous avez des étiquettes connues, je recommanderais de choisir les deux métriques suivantes :

- **Adjusted Rand Index (ARI)** :
  - **Pourquoi** : L'ARI mesure la similarité entre les clusters prédits et les étiquettes réelles, en tenant compte des regroupements aléatoires. C'est une métrique robuste qui donne une bonne indication de la qualité du clustering par rapport à la vérité terrain.

- **Adjusted Mutual Information (AMI)** :
  - **Pourquoi** : L'AMI quantifie l'accord entre les clusters et les vérités de classe, tout en tenant compte de la taille des classes. Elle est également insensible à l'échelle et est particulièrement utile lorsque le nombre de classes varie.

Ces deux métriques sont complémentaires et fournissent une évaluation complète de la performance du clustering par rapport aux étiquettes connues.

### 2. Comparaison de plusieurs représentations vectorielles avec KMeans

Lorsque vous souhaitez comparer plusieurs représentations vectorielles en utilisant KMeans, il est judicieux d'utiliser :

- **Silhouette Score** :
  - **Pourquoi** : Cette métrique vous permet d'évaluer la qualité de chaque cluster en fonction de la distance entre les points d'un même cluster et ceux des autres clusters. Un score élevé indique que les clusters sont bien séparés et compacts, ce qui est essentiel pour comparer différentes représentations.

- **Davies-Bouldin Index** :
  - **Pourquoi** : Cette métrique évalue la séparation entre les clusters par rapport à leur compacité. Un indice plus bas indique une meilleure qualité de clustering. Cela peut être particulièrement utile pour comparer différentes représentations vectorielles, car il prend en compte à la fois la distance inter-cluster et intra-cluster.

En résumé, pour évaluer un modèle KMeans avec des étiquettes connues, l'ARI et l'AMI sont des choix pertinents. Pour comparer différentes représentations vectorielles, le Silhouette Score et le Davies-Bouldin Index sont recommandés pour obtenir une vue d'ensemble complète de la qualité du clustering.

Citations:
[1] https://blog.octo.com/sous-le-capot-des-bases-de-donnees-vectorielles-(vector-databases)
[2] https://cedric.cnam.fr/vertigo/Cours/ml/coursClassificationAutomatique.html
[3] https://www.data-transitionnumerique.com/k-means/
[4] https://mrmint.fr/algorithme-k-means
[5] https://www.codeandcortex.fr/analyse-textuelle-kmeans/
[6] https://delladata.fr/kmeans/
[7] https://fr.wikipedia.org/wiki/K-moyennes
[8] https://www.ibm.com/docs/fr/db2/12.1?topic=building-k-means-clustering




