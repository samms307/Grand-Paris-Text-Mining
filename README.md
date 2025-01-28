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

Le modèle **TF-IDF** a été utilisé pour pondérer les termes des propositions mettant en avant les mots significatifs et réduisant l'impact des termes fréquents. 
Nous avons également testé **LSA (Latent Semantic Analysis)** qui permet de réduire la dimensionnalité tout en capturant les relations sémantiques entre les termes.

**Modèles Basés sur les Embeddings et Réseaux de Neurones Profonds**

Pour ces modèles nous avons utilisé des **modèles pré-entraînés** spécifiquement conçus pour le français. Nous avons exploré deux approches principales :

**Word Embeddings :**
- **Word2Vec** : Utilisé pour générer des représentations vectorielles des mots capturant leurs relations de similarité.
- **BERT** : Produit des `embeddings contextuels` où la représentation des mots varie selon leur contexte dans une phrase.

**Document/Sentence Embeddings :**
- **Agrégation par la moyenne des word embeddings** : Cette méthode calcule la moyenne des vecteurs `des mots d'une proposition entière` offrant une représentation globale.
- **USE (Universal Sentence Encoder)** : Génère des embeddings pour des phrases entières capturant ainsi le sens global d'une proposition.

------------------------------

### 3️⃣ Application d'une classification automatique et Évaluation des Représentations Vectorielles

PySpark propose plusieurs méthodes de classification automatique supervisées et non supervisées (clustering). Dans ce projet nous avons choisi l’algorithme K-means pour sa simplicité d’implémentation et son efficacité. De plus PySpark utilise une version `parallèle de K-means' optimisée pour exploiter pleinement les capacités de traitement distribué. 

**L’objectif principal de cette partie est d’évaluer quelle représentation vectorielle produit les clusters les plus cohérents et significatifs. Pour ce faire** :  

**3.1 Application du Modèle K-means || :** Nous appliquons ce modèle sur les différentes représentations vectorielles (mots ou documents) afin de regrouper les propositions d'idées en clusters homogènes. Chaque cluster représente idéalement un groupe d’idées similaires correspondant à une thématique déjà connue.
Le nombre de clusters a été fixé à **k = 5** en cohérence avec le nombre de thématiques identifiées dans nos données.  


**3.2 Sélection de la Meilleure Représentation Vectorielle avec K-means**: Nous utilisons deux métriques pour évaluer la qualité des clusters formés par K-means :  
- **Silhouette Score** : Cette métrique mesure la **compacité** des clusters et la **distance entre eux**.  
- **Davies-Bouldin Index (DBI)** : Cet indice évalue la **séparation** et la **compacité** des clusters (inter-cluster et intra-cluster).


**3.3 Validation des Clusters et Analyse Thématique**: Une fois la meilleure représentation vectorielle identifiée à l’aide des métriques précédentes, nous avons effectué une validation des clusters en analysant leur correspondance avec les thématiques connues :  

 - **Tableau de Contingence** : Nous avons construit un tableau de contingence entre les clusters générés par K-means et les thématiques préexistantes dans les données. Cet outil permet de visualiser la correspondance entre les clusters et les thématiques révélant leur cohérence et homogénéité.
 - **Homogeneity Score** : Cette métrique mesure dans quelle mesure chaque cluster ne contient que des données appartenant à une seule thématique. Un score d'homogénéité élevé indique que les clusters sont bien séparés en termes de thématiques.
 - **Adjusted Rand Index (ARI)** : Cette métrique évalue la similarité entre les clusters générés et les thématiques attendues en tenant compte des coïncidences dues au hasard. Un ARI élevé indique une meilleure correspondance entre les clusters et les thématiques.

Ces analyses complémentaires ont permis de confirmer la qualité des clusters et de valider la pertinence de la représentation vectorielle choisie.

----------------------

### 4️⃣ Sélection des Mots ou des Documents Représentatifs pour Chaque Cluster

Lors de l'analyse précédente nous avons constaté que les clusters générés par K-means présentent une forte similarité. De plus quelle que soit la représentation vectorielle utilisée,les propositions d'idées issues des différentes thématiques montrent un chevauchement important.

Pour finaliser le projet et extraire les termes (mots ou phrases) les plus représentatifs après le clustering nous avons adopté deux approches complémentaires :

**4.1 Sélection des Mots Représentatifs**

Nous avons utilisé Word Embedding avec Word2Vec et combiné deux méthodes pour identifier les termes représentatifs :

- **Cosine Similarity** : Cette méthode mesure la proximité des mots avec le centroïde de chaque cluster garantissant qu'ils représentent bien le thème principal.
                      Cependant, lorsque les clusters sont très similaires certains mots peuvent se chevaucher entre plusieurs clusters.
 - **TF-IDF** : Cette technique pondère les mots en fonction de leur fréquence dans un cluster et de leur rareté dans l'ensemble des données.
               Cela permet de gérer les termes redondants mais n'intègre pas directement leur proximité avec le centroïde.
 - **Combinaison des deux** : En combinant TF-IDF et Cosine Similarity nous maximisons la séparation entre les clusters. Cette approche permet de sélectionner des termes spécifiques et alignés avec le centroïde tout en réduisant le chevauchement. Cela améliore la précision et la pertinence des termes représentatifs pour chaque cluster.


**4.2 Sélection des documents Représentatifs + extraction de termes pertiant**

Pour identifier les documents les plus représentatifs de chaque cluster et en extraire les informations clés nous avons utilisé un sentence embedding avec USE (Universal Sentence Encoder). Deux techniques complémentaires ont été mises en œuvre :

**4.2.1 Identification des Documents Représentatifs** : 
 - Nous avons utilisé la distance cosinus pour mesurer la similarité entre chaque document et le centroïde du cluster.
 - Les 5 documents les plus proches de chaque centroïde ont été sélectionnés comme les plus pertinents reflétant au mieux la thématique du cluster.

**4.2.1 Extraction des Mots-Clés et Résumé Automatique**
 - **Extraction des mots-clés** : Pour extraire les mots clés nous avons utilisé la **bibliothèque RAKE** (Rapid Automatic Keyword Extraction). Cette méthode permet d'identifier les termes les plus significatifs dans les documents sélectionnés pour chaque cluster. RAKE est particulièrement efficace pour extraire des mots clés pertinents en se basant sur la fréquence et la distribution des termes dans le texte.
 - **Résumé automatique** : Pour produire un résumé de deux lignes par document nous avons employé la **bibliothèque Sumy** en s'appuyant sur l'algorithme de résumé LSA (Latent Semantic Analysis). Cet algorithme identifie les phrases les plus importantes en fonction des relations sémantiques entre elles et offrant une synthèse concise et pertinente.




