📊 Crime Data Analysis (2020 - Present) | Big Data Management
Αυτό το project πραγματοποιεί ανάλυση δεδομένων εγκληματικότητας στο Λος Άντζελες από το 2020 μέχρι σήμερα. Περιλαμβάνει προκαταρκτική ανάλυση, οπτικοποιήσεις, clustering και PCA για εξαγωγή συμπερασμάτων.

🛠️ Τεχνολογίες
Python 3.x
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn (KMeans, PCA)

🚀 Εκτέλεση
python exampleBigData1.py
Αρχικά βεβαιώσου πως έχεις το αρχείο Crime_Data_from_2020_to_Present.csv στον ίδιο φάκελο με το script.

📁 Φάκελοι
stats/: αποθηκεύονται όλες οι εικόνες και γραφήματα
results_csv/: αποθηκεύονται οι αναλύσεις σε CSV μορφή

📊 Αποτελέσματα (CSV)
Όλα τα στατιστικά εξάγονται και σε .csv μορφή στον φάκελο results_csv/:

results_crime_types.csv
results_area_stats.csv
results_time_trends.csv
results_hourly_stats.csv
results_time_of_day_stats.csv
results_day_stats.csv
results_crime_time_analysis.csv
results_clusters_analysis.csv
🔒 Περιορισμός δεδομένων
Για λόγους ταχύτητας και αποδοτικότητας, γίνεται sampling στα:

PCA: έως 20.000 εγγραφές
Clustering: έως 20.000 σημεία
Ολικό dataset: έως 100.000 εγγραφές
✍️ Δημιουργήθηκε από
Παπαγεωργίου Φίλιππος
Θωμάς Νικόλαος
Μπούμπας Ταξιάρχης
Εργασία για το μάθημα Big Data Management, Τμήμα Μηχανικών Πληροφορικής & Υπολογιστών / Πανεπιστήμιο Δυτικής Αττικής
