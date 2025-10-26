import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#from sklearn.metrics import silhouette_score
#import folium
from matplotlib.colors import LinearSegmentedColormap

# Ορισμός του στυλ για τα γραφήματα
plt.style.use('ggplot')
pd.set_option('display.max_columns', 200)

def load_data(csv_path):
    """Φόρτωση των δεδομένων από αρχείο CSV"""
    start_time = time.time()
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Χρόνος φόρτωσης δεδομένων: {time.time() - start_time:.2f} δευτερόλεπτα")
        return df
    else:
        raise FileNotFoundError(f"Σφάλμα: Το αρχείο '{csv_path}' δεν βρέθηκε.")

def inspect_dataframe(df):
    """Εξέταση της δομής και των χαρακτηριστικών του DataFrame"""
    start_time = time.time()
    
    # Βασικά στοιχεία για το DataFrame
    rows, columns = df.shape
    print(f"Το DataFrame έχει {rows} γραμμές και {columns} στήλες\n")
    print("Πρώτες 5 γραμμές:")
    print(df.head())
    print("\nΣτήλες:")
    print(df.columns)
    print("\nΤύποι δεδομένων:")
    print(df.dtypes)
    
    print("\nΕλλείπουσες τιμές ανά στήλη:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])  # Εμφάνιση μόνο των στηλών με ελλείπουσες τιμές
    
    print("\nΒασικά στατιστικά:")
    print(df.describe())
    
    print(f"Χρόνος επιθεώρησης δεδομένων: {time.time() - start_time:.2f} δευτερόλεπτα")

def preprocess_data(df):
    """Προεπεξεργασία του DataFrame."""
    start_time = time.time()
    
    # Δημιουργία αντιγράφου για να μην τροποποιηθεί το αρχικό DataFrame
    processed_df = df.copy()
    
    # Μετατροπή στηλών ημερομηνίας σε datetime
    processed_df['Date Rptd'] = pd.to_datetime(processed_df['Date Rptd'])
    processed_df['DATE OCC'] = pd.to_datetime(processed_df['DATE OCC'])
    
    # Εξαγωγή χρήσιμων χαρακτηριστικών από τις ημερομηνίες
    processed_df['Year'] = processed_df['DATE OCC'].dt.year
    processed_df['Month'] = processed_df['DATE OCC'].dt.month
    processed_df['Day'] = processed_df['DATE OCC'].dt.day
    processed_df['Day_of_Week'] = processed_df['DATE OCC'].dt.day_name()
    
    # Μετατροπή της στήλης TIME OCC σε πιο κατανοητή μορφή
    processed_df['Hour'] = processed_df['TIME OCC'] // 100
    processed_df['Minute'] = processed_df['TIME OCC'] % 100
    
    # Αφαίρεση των αρχικών στηλών ημερομηνίας αφού έχουμε μεταφέρει την πληροφορία
    processed_df = processed_df.drop(['TIME OCC', 'Date Rptd', 'DATE OCC'], axis=1)
    
    # Δημιουργία κατηγορίας ώρας της ημέρας
    bins = [0, 6, 12, 18, 24]
    labels = ['Νύχτα', 'Πρωί', 'Απόγευμα', 'Βράδυ']
    processed_df['Time_of_Day'] = pd.cut(processed_df['Hour'], bins=bins, labels=labels, right=False)
    
    # Διαχείριση ελλειπουσών τιμών
    # Υποθέτουμε ότι θέλουμε να διατηρήσουμε όλες τις γραμμές
    # Για αριθμητικές στήλες, συμπληρώνουμε με τη μέση τιμή
    numeric_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if processed_df[col].isnull().sum() > 0:
            processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
    
    # Για κατηγορικές στήλες, συμπληρώνουμε με την πιο συχνή τιμή
    categorical_cols = processed_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if processed_df[col].isnull().sum() > 0:
            processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
    
    print(f"Χρόνος προεπεξεργασίας δεδομένων: {time.time() - start_time:.2f} δευτερόλεπτα")
    return processed_df

def analyze_crime_types(df):
    """Ανάλυση των τύπων εγκλημάτων."""
    start_time = time.time()
    
    # Μέτρηση κάθε τύπου εγκλήματος
    crime_counts = df['Crm Cd Desc'].value_counts().reset_index()
    crime_counts.columns = ['Τύπος Εγκλήματος', 'Πλήθος']
    
    # Εύρεση των 10 πιο συχνών εγκλημάτων
    top_crimes = crime_counts.head(10)
    
    # Δημιουργία γραφήματος
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Πλήθος', y='Τύπος Εγκλήματος', data=top_crimes)
    plt.title('Οι 10 Πιο Συχνοί Τύποι Εγκλημάτων')
    plt.tight_layout()
    plt.savefig('stats/top_crimes.png', dpi=300)
    plt.close()
    
    print(f"Χρόνος ανάλυσης τύπων εγκλημάτων: {time.time() - start_time:.2f} δευτερόλεπτα")
    return crime_counts

def analyze_crime_by_area(df):
    """Ανάλυση κατανομής εγκλημάτων ανά περιοχή."""
    start_time = time.time()
    
    area_counts = df['AREA NAME'].value_counts().reset_index()
    area_counts.columns = ['Περιοχή', 'Πλήθος']
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Πλήθος', y='Περιοχή', data=area_counts)
    plt.title('Πλήθος Εγκλημάτων ανά Περιοχή')
    plt.tight_layout()
    plt.savefig('stats/crime_by_area.png', dpi=300)
    plt.close()
    
    print(f"Χρόνος ανάλυσης εγκλημάτων ανά περιοχή: {time.time() - start_time:.2f} δευτερόλεπτα")
    return area_counts

def analyze_crime_over_time(df):
    """Ανάλυση τάσεων εγκληματικότητας με την πάροδο του χρόνου."""
    start_time = time.time()
    
    # Ομαδοποίηση ανά μήνα και έτος, καταμέτρηση εγκλημάτων
    time_series = df.groupby(['Year', 'Month']).size().reset_index(name='Πλήθος')
    
    # Δημιουργία στήλης datetime για σωστή απεικόνιση
    time_series['Ημερομηνία'] = pd.to_datetime(time_series[['Year', 'Month']].assign(Day=1))
    
    plt.figure(figsize=(14, 6))
    plt.plot(time_series['Ημερομηνία'], time_series['Πλήθος'])
    plt.title('Τάσεις Εγκληματικότητας με την Πάροδο του Χρόνου')
    plt.xlabel('Ημερομηνία')
    plt.ylabel('Αριθμός Εγκλημάτων')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('stats/crime_over_time.png', dpi=300)
    plt.close()
    
    print(f"Χρόνος ανάλυσης εγκλημάτων με την πάροδο του χρόνου: {time.time() - start_time:.2f} δευτερόλεπτα")
    return time_series

def analyze_crime_by_time_of_day(df):
    """Ανάλυση εγκλημάτων ανά ώρα της ημέρας."""
    start_time = time.time()
    
    # Καταμέτρηση εγκλημάτων ανά ώρα
    hourly_counts = df.groupby('Hour').size().reset_index(name='Πλήθος')
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Hour', y='Πλήθος', data=hourly_counts, marker='o')
    plt.title('Κατανομή Εγκλημάτων ανά Ώρα της Ημέρας')
    plt.xlabel('Ώρα')
    plt.ylabel('Αριθμός Εγκλημάτων')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('stats/crime_by_hour.png', dpi=300)
    plt.close()
    
    # Καταμέτρηση εγκλημάτων ανά κατηγορία ώρας
    time_of_day_counts = df['Time_of_Day'].value_counts().reset_index()
    time_of_day_counts.columns = ['Περίοδος Ημέρας', 'Πλήθος']
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Περίοδος Ημέρας', y='Πλήθος', data=time_of_day_counts, 
                order=['Νύχτα', 'Πρωί', 'Απόγευμα', 'Βράδυ'])
    plt.title('Κατανομή Εγκλημάτων ανά Περίοδο της Ημέρας')
    plt.tight_layout()
    plt.savefig('stats/crime_by_time_of_day.png', dpi=300)
    plt.close()
    
    print(f"Χρόνος ανάλυσης εγκλημάτων ανά ώρα της ημέρας: {time.time() - start_time:.2f} δευτερόλεπτα")
    return hourly_counts, time_of_day_counts

def analyze_crime_by_day_of_week(df):
    """Ανάλυση εγκλημάτων ανά ημέρα της εβδομάδας."""
    start_time = time.time()
    
    # Καταμέτρηση εγκλημάτων ανά ημέρα της εβδομάδας
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['Day_of_Week'].value_counts().reset_index()
    day_counts.columns = ['Ημέρα', 'Πλήθος']
    
    # Ταξινόμηση των ημερών
    day_counts['Ημέρα'] = pd.Categorical(day_counts['Ημέρα'], categories=day_order, ordered=True)
    day_counts = day_counts.sort_values('Ημέρα')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Ημέρα', y='Πλήθος', data=day_counts)
    plt.title('Κατανομή Εγκλημάτων ανά Ημέρα της Εβδομάδας')
    plt.tight_layout()
    plt.savefig('stats/crime_by_day_of_week.png', dpi=300)
    plt.close()
    
    print(f"Χρόνος ανάλυσης εγκλημάτων ανά ημέρα της εβδομάδας: {time.time() - start_time:.2f} δευτερόλεπτα")
    return day_counts

def cluster_crime_locations(df, n_clusters=5):
    """Ομαδοποίηση (clustering) των τοποθεσιών εγκλημάτων με δειγματοληψία."""
    start_time = time.time()
    
    # Επιλογή δεδομένων για clustering
    geo_data = df[['LAT', 'LON']].dropna()
    
    # Δειγματοληψία για τη μείωση του υπολογιστικού κόστους
    # Χρησιμοποιούμε μέγιστο 20.000 σημεία για τη συσταδοποίηση
    sample_size = min(20000, len(geo_data))
    sampled_geo_data = geo_data.sample(sample_size, random_state=42)
    
    print(f"Χρησιμοποιούνται {sample_size} σημεία για την ομαδοποίηση από σύνολο {len(geo_data)} σημείων")
    
    # Κανονικοποίηση δεδομένων
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(sampled_geo_data)
    
    # Εφαρμογή K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Προσθήκη των ομάδων στα δεδομένα
    sampled_geo_data_with_clusters = sampled_geo_data.copy()
    sampled_geo_data_with_clusters['Cluster'] = clusters
    
    # Είχαμε σκοπό να υπολογίσουμε το Silhouette Score αλλά είναι υπολογιστικά πολύ ακριβό 
    # Εναλλακτικά, υπολογίζουμε την αδράνεια (inertia) που είναι πιο απλή μετρική
    print(f"Αδράνεια (inertia) του clustering: {kmeans.inertia_:.2f}")
    
    # Δημιουργία απλού scatter plot αντί για χάρτη folium (ταχύτερο)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        sampled_geo_data_with_clusters['LON'], 
        sampled_geo_data_with_clusters['LAT'], 
        c=sampled_geo_data_with_clusters['Cluster'], 
        s=3, 
        cmap='viridis', 
        alpha=0.7
    )
    plt.colorbar(scatter, label='Ομάδα')
    plt.title('Ομαδοποίηση Τοποθεσιών Εγκλημάτων')
    plt.xlabel('Γεωγραφικό Μήκος')
    plt.ylabel('Γεωγραφικό Πλάτος')
    plt.tight_layout()
    plt.savefig('stats/crime_clusters.png', dpi=300)
    plt.close()
    
    # Ανάλυση των clusters
    cluster_analysis = sampled_geo_data_with_clusters.groupby('Cluster').agg({
        'LAT': ['mean', 'count'],
        'LON': 'mean'
    })
    cluster_analysis.columns = ['Μέσο Γεωγραφικό Πλάτος', 'Πλήθος Εγκλημάτων', 'Μέσο Γεωγραφικό Μήκος']
    cluster_analysis = cluster_analysis.reset_index()
    
    print(f"Χρόνος ομαδοποίησης τοποθεσιών: {time.time() - start_time:.2f} δευτερόλεπτα")
    return cluster_analysis

def analyze_crime_types_by_time(df):
    """Ανάλυση των τύπων εγκλημάτων ανά περίοδο της ημέρας."""
    start_time = time.time()
    
    # Επιλογή των 5 πιο συχνών τύπων εγκλημάτων
    top_crimes = df['Crm Cd Desc'].value_counts().nlargest(5).index
    
    # Δημιουργία DataFrame μόνο με τους επιλεγμένους τύπους εγκλημάτων
    filtered_df = df[df['Crm Cd Desc'].isin(top_crimes)]
    
    # Δημιουργία πίνακα συνάφειας (contingency table)
    crime_time_table = pd.crosstab(filtered_df['Crm Cd Desc'], filtered_df['Time_of_Day'])
    
    # Μετατροπή των απόλυτων αριθμών σε ποσοστά ανά τύπο εγκλήματος
    crime_time_pct = crime_time_table.div(crime_time_table.sum(axis=1), axis=0) * 100
    
    # Δημιουργία heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(crime_time_pct, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Ποσοστό (%)'})
    plt.title('Κατανομή των 5 Πιο Συχνών Τύπων Εγκλημάτων ανά Περίοδο της Ημέρας')
    plt.tight_layout()
    plt.savefig('stats/crime_types_by_time.png', dpi=300)
    plt.close()
    
    print(f"Χρόνος ανάλυσης τύπων εγκλημάτων ανά περίοδο: {time.time() - start_time:.2f} δευτερόλεπτα")
    return crime_time_pct

def analyze_victim_demographics(df):
    """Ανάλυση δημογραφικών χαρακτηριστικών θυμάτων"""
    start_time = time.time()
    
    # Κατανομή ηλικιών θυμάτων
    age_data = df[df['Vict Age'] > 0]['Vict Age']
    
    plt.figure(figsize=(12, 6))
    plt.hist(age_data, bins=20, edgecolor='black')
    plt.xlabel('Ηλικία')
    plt.ylabel('Πλήθος')
    plt.title('Κατανομή Ηλικιών Θυμάτων')
    plt.savefig('stats/victim_age_distribution.png', dpi=300)
    plt.close()
    
    # Κατανομή φύλου θυμάτων
    sex_counts = df['Vict Sex'].value_counts()
    
    plt.figure(figsize=(8, 8))
    plt.pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%')
    plt.title('Κατανομή Φύλου Θυμάτων')
    plt.savefig('stats/victim_sex_distribution.png', dpi=300)
    plt.close()
    
    print(f"Χρόνος ανάλυσης δημογραφικών: {time.time() - start_time:.2f} δευτερόλεπτα")

def predict_future_crimes(df):
    """Πρόβλεψη μελλοντικών εγκλημάτων βασισμένη σε ιστορικά δεδομένα."""
    start_time = time.time()
    
    # Δημιουργία μηνιαίων συνολικών εγκλημάτων
    monthly_data = df.groupby(['Year', 'Month']).size().reset_index(name='Crime_Count')
    monthly_data['Date'] = pd.to_datetime(monthly_data[['Year', 'Month']].assign(Day=1))
    monthly_data = monthly_data.sort_values('Date')
    
    # Χρήση απλού γραμμικού μοντέλου για πρόβλεψη
    from sklearn.linear_model import LinearRegression
    
    # Προετοιμασία δεδομένων για το μοντέλο
    X = np.arange(len(monthly_data)).reshape(-1, 1)  # Χρονικός δείκτης
    y = monthly_data['Crime_Count'].values
    
    # Εκπαίδευση του μοντέλου
    model = LinearRegression()
    model.fit(X, y)
    
    # Πρόβλεψη για τους επόμενους 12 μήνες
    future_months = 12
    last_date = monthly_data['Date'].max()
    future_X = np.arange(len(monthly_data), len(monthly_data) + future_months).reshape(-1, 1)
    future_predictions = model.predict(future_X)
    
    # Δημιουργία ημερομηνιών για τις προβλέψεις
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_months, freq='MS')
    
    # Δημιουργία DataFrame με τις προβλέψεις
    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Crime_Count': np.maximum(future_predictions, 0)  # Αποφυγή αρνητικών προβλέψεων
    })
    
    # Υπολογισμός R² score για αξιολόγηση του μοντέλου
    # Το R² δείχνει πόσο καλά προσαρμόζεται το μοντέλο στα δεδομένα. Τιμή > 0.7 θεωρείται καλή.
    from sklearn.metrics import r2_score
    historical_predictions = model.predict(X)
    r2 = r2_score(y, historical_predictions)
    
    # Δημιουργία γραφήματος με ιστορικά δεδομένα και προβλέψεις
    plt.figure(figsize=(14, 8))
    
    # Ιστορικά δεδομένα
    plt.plot(monthly_data['Date'], monthly_data['Crime_Count'], 
             label='Ιστορικά Δεδομένα', color='blue', linewidth=2)
    
    # Προβλέψεις
    plt.plot(predictions_df['Date'], predictions_df['Predicted_Crime_Count'], 
             label='Προβλέψεις', color='red', linestyle='--', linewidth=2, marker='o')
    
    # Γραμμή τάσης για ιστορικά δεδομένα
    plt.plot(monthly_data['Date'], historical_predictions, 
             label='Γραμμή Τάσης', color='green', alpha=0.7)
    
    plt.title('Πρόβλεψη Μηνιαίων Εγκλημάτων για τους Επόμενους 12 Μήνες')
    plt.xlabel('Ημερομηνία')
    plt.ylabel('Αριθμός Εγκλημάτων')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('stats/crime_prediction.png', dpi=300)
    plt.close()
    
    # Ανάλυση εποχικότητας - πρόβλεψη ανά μήνα του έτους
    monthly_avg = df.groupby('Month')['Month'].count().reset_index(name='Avg_Crime_Count')
    monthly_avg.columns = ['Month', 'Average_Crimes']
    
    plt.figure(figsize=(12, 6))
    plt.bar(monthly_avg['Month'], monthly_avg['Average_Crimes'], 
            color=['skyblue' if i != monthly_avg['Average_Crimes'].idxmax() else 'red' 
                   for i in range(len(monthly_avg))])
    plt.title('Μέσος Αριθμός Εγκλημάτων ανά Μήνα (Εποχικότητα)')
    plt.xlabel('Μήνας')
    plt.ylabel('Μέσος Αριθμός Εγκλημάτων')
    plt.xticks(range(1, 13), ['Ιαν', 'Φεβ', 'Μαρ', 'Απρ', 'Μαι', 'Ιουν', 
                              'Ιουλ', 'Αυγ', 'Σεπ', 'Οκτ', 'Νοε', 'Δεκ'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('stats/monthly_seasonality.png', dpi=300)
    plt.close()
    
    # Στατιστικά πρόβλεψης
    avg_prediction = predictions_df['Predicted_Crime_Count'].mean()
    max_prediction = predictions_df['Predicted_Crime_Count'].max()
    min_prediction = predictions_df['Predicted_Crime_Count'].min()
    
    print(f"Αξιολόγηση Μοντέλου - R² Score: {r2:.3f}")
    print(f"Μέσος προβλεπόμενος αριθμός εγκλημάτων/μήνα: {avg_prediction:.0f}")
    print(f"Μέγιστος προβλεπόμενος αριθμός: {max_prediction:.0f}")
    print(f"Ελάχιστος προβλεπόμενος αριθμός: {min_prediction:.0f}")
    
    # Εύρεση του μήνα με τη μεγαλύτερη εποχικότητα
    peak_month = monthly_avg.loc[monthly_avg['Average_Crimes'].idxmax(), 'Month']
    month_names = {1: 'Ιανουάριος', 2: 'Φεβρουάριος', 3: 'Μάρτιος', 4: 'Απρίλιος',
                   5: 'Μάιος', 6: 'Ιούνιος', 7: 'Ιούλιος', 8: 'Αύγουστος',
                   9: 'Σεπτέμβριος', 10: 'Οκτώβριος', 11: 'Νοέμβριος', 12: 'Δεκέμβριος'}
    
    print(f"Μήνας με υψηλότερη εγκληματικότητα: {month_names[peak_month]}")
    
    print(f"Χρόνος πρόβλεψης: {time.time() - start_time:.2f} δευτερόλεπτα")
    
    return predictions_df, monthly_avg, r2

def perform_dimensionality_reduction(df):
    """Εκτέλεση μείωσης διαστατικότητας με PCA σε δείγμα των δεδομένων."""
    start_time = time.time()
    
    # Επιλογή αριθμητικών χαρακτηριστικών για PCA
    numeric_cols = ['LAT', 'LON', 'Hour', 'Year', 'Month', 'Day']
    
    # Αφαίρεση γραμμών με ελλείπουσες τιμές
    pca_data = df[numeric_cols].dropna()
    
    # Δειγματοληψία για μείωση του υπολογιστικού κόστους
    sample_size = min(20000, len(pca_data))
    sampled_pca_data = pca_data.sample(sample_size, random_state=42)
    
    print(f"Χρησιμοποιούνται {sample_size} σημεία για το PCA από σύνολο {len(pca_data)} σημείων")
    
    # Κανονικοποίηση δεδομένων
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(sampled_pca_data)
    
    # Εφαρμογή PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # Δημιουργία DataFrame με τα αποτελέσματα της PCA
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    
    # Δημιουργία scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5, s=3)
    plt.title('PCA των Χαρακτηριστικών των Εγκλημάτων')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} διακύμανσης)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} διακύμανσης)')
    plt.tight_layout()
    plt.savefig('stats/crime_pca.png', dpi=300)
    plt.close()
    
    # Ανάλυση των συστατικών του PCA
    components_df = pd.DataFrame(
        pca.components_,
        columns=numeric_cols,
        index=['PC1', 'PC2']
    )
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(components_df, annot=True, cmap='coolwarm', cbar_kws={'label': 'Συνεισφορά'})
    plt.title('Συνεισφορά των Χαρακτηριστικών στις Κύριες Συνιστώσες')
    plt.tight_layout()
    plt.savefig('stats/pca_components.png', dpi=300)
    plt.close()
    
    print(f"Χρόνος μείωσης διαστατικότητας: {time.time() - start_time:.2f} δευτερόλεπτα")
    return pca, components_df


def limit_data_size(df, max_rows=100000):
    """Περιορίζει το μέγεθος του DataFrame για καλύτερη απόδοση."""
    if len(df) > max_rows:
        print(f"Το αρχικό DataFrame έχει {len(df)} γραμμές. Γίνεται περιορισμός σε {max_rows} γραμμές.")
        return df.sample(max_rows, random_state=42)
    return df

def main():
    csv_path = "Crime_Data_from_2020_to_Present.csv"
    total_start_time = time.time()
    try:
        os.mkdir('stats')
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        os.mkdir('results_csv')
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        # Βήμα 1: Φόρτωση και επιθεώρηση δεδομένων
        print("\n=== ΦΟΡΤΩΣΗ ΚΑΙ ΕΠΙΘΕΩΡΗΣΗ ΔΕΔΟΜΕΝΩΝ ===")
        df = load_data(csv_path)
        # Περιορισμός του μεγέθους του DataFrame για καλύτερη απόδοση
        df = limit_data_size(df, max_rows=100000)
        inspect_dataframe(df)
        
        # Βήμα 2: Προεπεξεργασία δεδομένων
        print("\n=== ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ ΔΕΔΟΜΕΝΩΝ ===")
        processed_df = preprocess_data(df)
        print("Επιθεώρηση προεπεξεργασμένων δεδομένων:")
        print(processed_df.head())
        print(processed_df.dtypes)
        
        # Βήμα 3: Ανάλυση δεδομένων
        print("\n=== ΑΝΑΛΥΣΗ ΔΕΔΟΜΕΝΩΝ ===")
        
        # Ανάλυση τύπων εγκλημάτων
        print("\n--- Ανάλυση Τύπων Εγκλημάτων ---")
        crime_types = analyze_crime_types(processed_df)
        print(f"Συνολικά βρέθηκαν {len(crime_types)} διαφορετικοί τύποι εγκλημάτων")
        print("Οι 10 πιο συχνοί τύποι εγκλημάτων:")
        print(crime_types.head(10))
        
        # Ανάλυση τύπων εγκλημάτων
        print("\n--- Ανάλυση  δημογραφικών χαρακτηριστικών θυμάτων  ---")
        victim_demographics = analyze_victim_demographics(processed_df)
        
        # Πρόβλεψη μελλοντικών εγκλημάτων
        print("\n--- Πρόβλεψη Μελλοντικών Εγκλημάτων ---")
        predictions, seasonality, model_score = predict_future_crimes(processed_df)
        print("Προβλέψεις για τους επόμενους 12 μήνες:")
        print(predictions.head())
        print("Εποχικότητα εγκλημάτων ανά μήνα:")
        print(seasonality)
        
        # Ανάλυση εγκλημάτων ανά περιοχή
        print("\n--- Ανάλυση Εγκλημάτων ανά Περιοχή ---")
        area_stats = analyze_crime_by_area(processed_df)
        print("Στατιστικά εγκλημάτων ανά περιοχή:")
        print(area_stats.head(10))
        
        # Ανάλυση εγκλημάτων με την πάροδο του χρόνου
        print("\n--- Ανάλυση Εγκλημάτων με την Πάροδο του Χρόνου ---")
        time_trends = analyze_crime_over_time(processed_df)
        print("Μηνιαία στατιστικά εγκλημάτων:")
        print(time_trends.head())
        
        # Ανάλυση εγκλημάτων ανά ώρα της ημέρας
        print("\n--- Ανάλυση Εγκλημάτων ανά Ώρα της Ημέρας ---")
        hourly_stats, time_of_day_stats = analyze_crime_by_time_of_day(processed_df)
        print("Στατιστικά εγκλημάτων ανά περίοδο της ημέρας:")
        print(time_of_day_stats)
        
        # Ανάλυση εγκλημάτων ανά ημέρα της εβδομάδας
        print("\n--- Ανάλυση Εγκλημάτων ανά Ημέρα της Εβδομάδας ---")
        day_stats = analyze_crime_by_day_of_week(processed_df)
        print("Στατιστικά εγκλημάτων ανά ημέρα της εβδομάδας:")
        print(day_stats)
        
        # Ανάλυση των τύπων εγκλημάτων ανά περίοδο της ημέρας
        print("\n--- Ανάλυση Τύπων Εγκλημάτων ανά Περίοδο της Ημέρας ---")
        crime_time_analysis = analyze_crime_types_by_time(processed_df)
        print("Κατανομή των 5 πιο συχνών τύπων εγκλημάτων ανά περίοδο της ημέρας (%):")
        print(crime_time_analysis)
        
        # Βήμα 4: Προχωρημένες αναλύσεις
        print("\n=== ΠΡΟΧΩΡΗΜΕΝΕΣ ΑΝΑΛΥΣΕΙΣ ===")
        
        # Ομαδοποίηση τοποθεσιών εγκλημάτων
        print("\n--- Ομαδοποίηση Τοποθεσιών Εγκλημάτων ---")
        clusters_analysis = cluster_crime_locations(processed_df, n_clusters=5)
        print("Ανάλυση των ομάδων τοποθεσιών εγκλημάτων:")
        print(clusters_analysis)
        
        # Μείωση διαστατικότητας με PCA
        print("\n--- Μείωση Διαστατικότητας με PCA ---")
        pca_model, pca_components = perform_dimensionality_reduction(processed_df)
        print("Ποσοστό εξηγούμενης διακύμανσης από τις πρώτες 2 συνιστώσες:")
        print(f"PC1: {pca_model.explained_variance_ratio_[0]:.2%}")
        print(f"PC2: {pca_model.explained_variance_ratio_[1]:.2%}")
        print("Συνεισφορά των χαρακτηριστικών στις κύριες συνιστώσες:")
        print(pca_components)
        
        # Συνολικός χρόνος εκτέλεσης
        total_time = time.time() - total_start_time
        print(f"\nΣυνολικός χρόνος εκτέλεσης: {total_time:.2f} δευτερόλεπτα")
        
        # Αποθήκευση των αποτελεσμάτων σε αρχείο CSV για μελλοντική χρήση
        results = {
            'crime_types': crime_types,
            'area_stats': area_stats,
            'time_trends': time_trends,
            'hourly_stats': hourly_stats,
            'time_of_day_stats': time_of_day_stats,
            'day_stats': day_stats,
            'crime_time_analysis': crime_time_analysis,
            'clusters_analysis': clusters_analysis,
            'victim_demographics' : victim_demographics,
            'predictions': predictions,
            'seasonality': seasonality
        }
        
        # Αποθήκευση των αποτελεσμάτων σε ξεχωριστά αρχεία CSV
        for name, result in results.items():
            result.to_csv(f'results_csv/results_{name}.csv', index=False)
        
        print("\nΌλα τα αποτελέσματα έχουν αποθηκευτεί σε αρχεία CSV.")
        
    except Exception as e:
        print(f"Σφάλμα: {e}")

if __name__ == "__main__":
    main()