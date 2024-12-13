# **Presentation des données**

Dans le cadre de la campagne *"Construisons la Métropole du Grand Paris"*, menée par la **Direction de la Démocratie, des Citoyens et des Territoires (DDCT)** de la Ville de Paris, des citoyens et des collectivités ont soumis des idées pour construire une métropole plus connectée, durable et inclusive.

Ces idées, réparties en **5 grandes thématiques** :
- Transition écologique et Mobilités  
- Culture et Identité  
- Logement et Aménagement  
- Rayonnement  
- Lutte contre les inégalités  

Le jeu de données correspondant contient 362 propositions, chacune décrite par 21 attributs, incluant la date de publication, l'objectif, et une description textuelle détaillée.


# **Présentation des Données**

Dans le cadre de la campagne *"Construisons la Métropole du Grand Paris"*, organisée par la **Direction de la Démocratie, des Citoyens et des Territoires (DDCT)** de la Ville de Paris, des citoyens et des collectivités ont soumis des idées innovantes pour transformer la métropole en un espace plus connecté, durable et inclusif.

Les propositions idées sont regroupées en **5 grandes thématiques** :
1. **Transition écologique et Mobilités** : Solutions pour réduire l'empreinte écologique et améliorer les transports.  
2. **Culture et Identité** : Initiatives visant à promouvoir le patrimoine culturel et l’identité locale.  
3. **Logement et Aménagement** : Projets pour améliorer l’accès au logement et repenser l’espace urbain.  
4. **Rayonnement** : Idées pour accroître l’attractivité et le rayonnement international de la métropole.  
5. **Lutte contre les inégalités** : Actions pour réduire les disparités économiques et sociales.

### **Structure des Données**
Le jeu de données comprend **362 propositions**, chacune décrite par **21 attributs**. Ces attributs incluent notamment :
- **Date de publication** : Pour situer chronologiquement les idées.  
- **Objectif** : Une indication des résultats attendus.  
- **Description textuelle** : Une explication détaillée de chaque proposition.  

### **Enjeu pour la Data Science**
Ce jeu de données constitue une opportunité d’appliquer des techniques avancées d’analyse textuelle pour :
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




