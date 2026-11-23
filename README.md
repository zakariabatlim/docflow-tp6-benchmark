# DocFlow Pro · Benchmark TP-6

Dossier autonome du dépôt public **`zakariabatlim/docflow-tp6-benchmark`**, sur la branche `main`. Le dépôt sert à comparer la géométrie du détecteur OpenCV de DocFlow Pro à une variante expérimentale qui emploie SLANet_plus. Le dossier ne requiert ni Django, ni PostgreSQL, ni configuration du projet DocFlow Pro.

## Fichiers à publier

| Chemin | Contenu |
| --- | --- |
| `.github/workflows/benchmark.yml` | Workflow manuel pour le dépôt public dédié |
| `detector/` et `provenance-code.json` | Extrait autonome du détecteur OpenCV et empreintes des sources et de l’extrait |
| `scripts/` | Comparaison et supervision, adaptateur ONNX, contrôle des téléchargements, export des mesures et générateur du corpus |
| `corpus/validation/` | 48 scènes synthétiques gelées et leurs annotations (36 tableaux et 12 cas à rejeter) |
| `corpus/smoke/` | 3 contrôles synthétiques hors de la comparaison |
| `requirements-cpu.lock` et `provenance-telechargements.json` | Versions, URL officielles, tailles, licences, révision de modèle et SHA-256 |
| `licenses/Apache-2.0.txt`, `LICENCES.md`, `README.md` | Licence applicable et guide |
| `.gitignore` et `.gitattributes` | Exclusion des binaires et sorties temporaires, conservation des octets annotés |

Les fichiers versionnés proviennent d’une liste explicite. Le ZIP portable, les roues, les poids, les rapports, le venv, les bases, les fichiers `.env`, les documents privés et les secrets restent en dehors des fichiers à publier. Le workflow utilise deux actions officielles épinglées par commit, un jeton limité à `contents: read` et un checkout qui ne conserve pas le jeton.

## Lancer le benchmark

Avec un droit d’écriture, ouvrir **Actions → Benchmark TP-6 sous Windows → Run workflow**, sélectionner **main** et confirmer le lancement manuel après validation. Le job est réservé au dépôt public `zakariabatlim/docflow-tp6-benchmark` et à cette branche. Le groupe de concurrence autorise au plus un job de benchmark actif.

GitHub indique que le calcul sur runner standard est gratuit pour les dépôts publics. Les journaux d’exécution sont publics. Le workflow limite les artefacts aux mesures et journaux du corpus synthétique.

[Facturation GitHub Actions](https://docs.github.com/en/billing/concepts/product-billing/github-actions) · [Déclenchement manuel](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)

## Exécution et protections

- `windows-2025` standard x64 dans un dépôt public, annoncé à **4 vCPU, 16 Go de RAM et 14 Go de SSD**. La version de correctif CPython **3.13 x64** la plus récente disponible dans le cache du runner est utilisée et consignée. Son absence arrête le job. [Caractéristiques des runners](https://docs.github.com/en/actions/how-tos/write-workflows/choose-where-workflows-run/choose-the-runner-for-a-job)
- RAM et disque relevés dans le job avant téléchargement, avant installation et avant benchmark. Seuil de **2 Gio de RAM disponible** et **1 Gio de marge disque**. Le budget du dossier TP-6 est de **1 Gio**. L’estimation de décompression est vérifiée puis l’occupation réelle est relevée après les passes.
- Téléchargement limité à **77 630 593 octets** de binaires approuvés, via HTTPS, auprès de PaddlePaddle/Hugging Face et de PyPI. L’empreinte et la taille de chacun des neuf fichiers, y compris la configuration du modèle, doivent correspondre. Installation des sept roues hors ligne avec `pip --no-index --no-deps --require-hashes` dans `.venv-tp6`, puis `pip check`.
- ONNX Runtime CPU, deux threads d’inférence au maximum, OpenCV à un thread et un seul document à la fois. Un Job Object Windows limite la mémoire privée engagée du processus de benchmark à **1 Gio**. Le superviseur contrôle la RAM résidente toutes les 50 ms et arrête l’essai au-delà de **1 Gio** ou si la RAM disponible du système chute sous **0,5 Gio**. Le benchmark dispose d’un délai global de **660 secondes** au sein du job de **20 minutes**.
- Les sorties autorisées sont les JSON, CSV et journaux des mesures dans `rapports/`. GitHub les conserve **7 jours** sous `docflow-tp6-resultats`. Si un précontrôle échoue, l’artefact peut contenir uniquement les états et messages d’arrêt. Un délai maximal GitHub ou une annulation reste susceptible d’interrompre la publication des artefacts.

## Lire les mesures

La comparaison principale traite, dans le même ordre, les **48 scènes** et les mêmes annotations. La proposition de zone OpenCV est utilisée en amont de SLANet_plus lorsqu’elle est disponible. Sur d’autres cas, SLANet_plus reçoit la page entière. La comparaison mesure donc deux pipelines de détection. La passe complémentaire utilise la zone annotée des **36 tableaux** pour les deux moteurs, afin d’isoler l’effet du découpage de la page. Les trois contrôles de développement restent hors des scores.

`synthese.json` et `synthese.csv` donnent les dimensions et topologies exactes, la précision et le rappel des cellules appariées par position et fusion avec IoU ≥ 0,80, les fusions, les rejets, la latence médiane et P95, le démarrage, les pics de RAM et le CPU utilisateur/noyau. Le P95 est le rang supérieur de 95 % dans la série triée. `resultats-par-document.csv` conserve le détail et les rapports JSON de passes incluent les catégories et les prédictions. Les unités et les conditions de collecte figurent dans les noms des champs. Aucun résultat n’est imputé aux passes interrompues. Le benchmark porte sur la géométrie. L’OCR et le temps de correction humaine ne sont pas mesurés par ce workflow.

Le corpus synthétique permet de décider s’il faut poursuivre. Avant toute décision d’intégration, prévoir un corpus professionnel indépendant dont les droits d’usage sont vérifiés, figer ses annotations et seuils avant la mesure, couvrir les tableaux sans bordures, les fusions, les cas complexes et les rejets, puis mesurer séparément les effets sur l’OCR. Garder les documents et les sorties détaillées sur un environnement explicitement autorisé. Comparer les limites de ressources du poste cible et vérifier les régressions de DocFlow Pro avant de demander un GO.
