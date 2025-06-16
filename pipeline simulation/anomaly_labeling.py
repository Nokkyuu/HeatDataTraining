import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class ImprovedAnomalyLabeler:
    """
    This script is used for initial anomaly labeling as a starting point for training the models for the data pipeline.
    It can be extended by more careful feature engineering, expanding it on forward flow data, changing models, etc.
    """
    
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        
    def create_features(self, df):
        features = df.copy()
        features = features.sort_values('timestamp').reset_index(drop=True)

        # time based features
        dt = pd.to_datetime(df['timestamp'])
        features['hour'] = dt.dt.hour
        features['day_of_week'] = dt.dt.dayofweek
        features['month'] = dt.dt.month
        features['is_weekend'] = features['day_of_week'].isin([5, 6])
    
        if 'nighttime' in df.columns:
            features['nighttime'] = df['nighttime']
        
        # dummies for type - add ID for multiple systems
        categorical_cols = ['type']
        
        for col in categorical_cols:
            if col in df.columns:
                # Erstelle Dummy-Variablen
                dummies = pd.get_dummies(
                    df[col], 
                    prefix=col,
                    drop_first=False,
                    dummy_na=True
                )
                features = pd.concat([features, dummies], axis=1)
        
        # rolling features
        self._add_time_based_rolling_features(features)
        # time diff features
        self._add_time_diff_features(features)
        # rate based features (changes over time)
        self._add_rate_features(features)
        # temperature specific features
        self._add_temperature_features(features)
        # exclusion of original columns
        exclude_cols = ['timestamp', 'type', 'ID', 'datetime']
        feature_cols = [col for col in features.columns if col not in exclude_cols]
        
        final_features = features[feature_cols].copy()
        
        return final_features.ffill().fillna(0)
    
    def _add_temperature_features(self, df):
        """temperature specificfeatures (differences to avg, stability etc.)"""
        if 'avg' in df.columns and 'min_v' in df.columns and 'max_v' in df.columns:
            df['avg_to_max_ratio'] = df['avg'] / (df['max_v'] + 1e-8)
            df['avg_to_min_ratio'] = df['avg'] / (df['min_v'] + 1e-8)
            df['range_to_avg_ratio'] = df['T_diff'] / (df['avg'] + 1e-8)
            df['temp_range_normalized'] = df['T_diff'] / (df['max_v'] + 1e-8)
            df['temp_stability'] = 1 / (df['T_diff'] + 1e-8)
    
    def _add_time_based_rolling_features(self, df):
        """rolling means and rolling stds for time based features"""
        df['datetime'] = pd.to_datetime(df['timestamp'])
        
        numeric_cols = ['avg', 'min_v', 'max_v', 'T_diff']
        
        for col in numeric_cols:
            if col in df.columns:
                for hours in [1, 6, 24]:
                    df[f'{col}_rolling_mean_{hours}h'] = self._time_based_rolling(
                        df, col, 'mean', hours
                    )
                    df[f'{col}_rolling_std_{hours}h'] = self._time_based_rolling(
                        df, col, 'std', hours
                    )
                df[f'{col}_weighted_change'] = self._weighted_change(df, col)
    
    def _time_based_rolling(self, df, column, operation, hours):
        """rolling mean/std over a time window"""
        result = []
        for i in range(len(df)):
            current_time = df['datetime'].iloc[i]
            time_window_start = current_time - pd.Timedelta(hours=hours)
            
            mask = (df['datetime'] <= current_time) & (df['datetime'] >= time_window_start)
            window_data = df.loc[mask, column]
            
            if len(window_data) > 0:
                if operation == 'mean':
                    result.append(window_data.mean())
                elif operation == 'std':
                    result.append(window_data.std() if len(window_data) > 1 else 0)
            else:
                result.append(df[column].iloc[i])
                
        return result
    
    def _weighted_change(self, df, column):
        """weighted change of a column based on time differences"""
        changes = []
        for i in range(len(df)):
            if i == 0:
                changes.append(0)
            else:
                value_change = df[column].iloc[i] - df[column].iloc[i-1]
                time_diff = df['time_diff_min'].iloc[i] if 'time_diff_min' in df.columns else 1
                
                if time_diff > 0:
                    weighted_change = value_change / time_diff
                    changes.append(weighted_change)
                else:
                    changes.append(0)
        return changes
    
    def _add_time_diff_features(self, df):
        """adds time difference features"""
        if 'time_diff_min' in df.columns:
            median_time_diff = df['time_diff_min'].median()
            mean_time_diff = df['time_diff_min'].mean()
            std_time_diff = df['time_diff_min'].std()
            
            df['time_diff_normalized'] = df['time_diff_min'] / median_time_diff
            df['time_diff_deviation'] = np.abs(df['time_diff_min'] - median_time_diff)
            df['time_diff_z_score'] = np.abs((df['time_diff_min'] - mean_time_diff) / (std_time_diff + 1e-8))
    
    def _add_rate_features(self, df):
        """adds change rates per minute for numeric columns"""
        numeric_cols = ['avg', 'min_v', 'max_v', 'T_diff']
        
        for col in numeric_cols:
            if col in df.columns and 'time_diff_min' in df.columns:
                rates = []
                for i in range(len(df)):
                    if i == 0:
                        rates.append(0)
                    else:
                        value_diff = df[col].iloc[i] - df[col].iloc[i-1]
                        time_diff = df['time_diff_min'].iloc[i]
                        
                        if time_diff > 0:
                            rate = value_diff / time_diff
                            rates.append(rate)
                        else:
                            rates.append(0)
                
                df[f'{col}_rate_per_min'] = rates
    
    def detect_anomalies(self, df, system_type='domestic'):
        """Anomaly detection using Isolation Forest as unsupervised ML model and additional methods"""
        if system_type and 'type' in df.columns:
            df_filtered = df[df['type'] == system_type].copy().reset_index(drop=True)
        else:
            df_filtered = df.copy().reset_index(drop=True)
            
        print(f"Analysiere {len(df_filtered)} Datenpunkte für Typ: {system_type}")
        
        # Zeitdifferenz-Statistiken (nur zur Info)
        if 'time_diff_min' in df_filtered.columns:
            time_stats = df_filtered['time_diff_min'].describe()
            print(f"\nTime difference in minutes:")
            print(f"  Median: {time_stats['50%']:.1f}")
            print(f"  Mean: {time_stats['mean']:.1f}")
            print(f"  Min: {time_stats['min']:.1f}")
            print(f"  Max: {time_stats['max']:.1f}")
        
        # feature engineering
        features = self.create_features(df_filtered)
        print(f"\nCreate {features.shape[1]} Features")
        
        # scaling
        features_scaled = self.scaler.fit_transform(features)
        
        # Isolation Forest
        anomaly_scores = self.isolation_forest.fit_predict(features_scaled)
        anomaly_scores_proba = self.isolation_forest.score_samples(features_scaled)
        
        # anomaly detection
        statistical_anomalies = self._detect_statistical_anomalies(df_filtered)
        domain_anomalies = self._detect_domain_anomalies(df_filtered)
        
        # combination
        combined_anomalies = (
            (anomaly_scores == -1) | 
            statistical_anomalies | 
            domain_anomalies
        )
        
        results = df_filtered.copy()
        results['is_anomaly'] = combined_anomalies
        results['anomaly_score'] = anomaly_scores_proba
        results['isolation_forest_anomaly'] = anomaly_scores == -1
        results['statistical_anomaly'] = statistical_anomalies
        results['domain_anomaly'] = domain_anomalies
        
        return results
    
    def _detect_statistical_anomalies(self, df):
        """detect anomalies using statistical methods (Z-Score)"""
        anomalies = np.zeros(len(df), dtype=bool)
        
        numeric_cols = ['avg', 'min_v', 'max_v', 'T_diff']  # time_diff_min entfernt
        for col in numeric_cols:
            if col in df.columns:
                z_scores = np.abs(stats.zscore(df[col].fillna(df[col].median())))
                anomalies |= z_scores > 3
                
        return anomalies
    
    def _detect_domain_anomalies(self, df):
        """detect domain-specific anomalies based on known constraints"""
        anomalies = np.zeros(len(df), dtype=bool)
        
        # Impossible temperatures
        if 'avg' in df.columns:
            anomalies |= (df['avg'] < -10) | (df['avg'] > 100)
            
        if 'min_v' in df.columns and 'max_v' in df.columns:
            anomalies |= df['min_v'] > df['max_v']
            
        if 'T_diff' in df.columns:
            anomalies |= (df['T_diff'] < 0) | (df['T_diff'] > 50)
            
        return anomalies
    
    def visualize_anomalies(self, results):
        """visualizes the detected anomalies in a comprehensive way"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        results['datetime'] = pd.to_datetime(results['timestamp'])
        
        
        numeric_cols = ['avg', 'min_v', 'max_v', 'T_diff', 'time_diff_min']
        
        for i, col in enumerate(numeric_cols):
            if col in results.columns and i < len(axes):
                ax = axes[i]
                
                # Scatter plot for each numeric column
                normal_data = results[~results['is_anomaly']]
                ax.scatter(normal_data['datetime'], normal_data[col], 
                          alpha=0.6, label='Normal', s=1, color='blue')
                
                # anomalies by type
                for anomaly_type, color in [
                    ('isolation_forest_anomaly', 'red'),
                    ('statistical_anomaly', 'orange'),
                    ('domain_anomaly', 'purple')
                ]:
                    if anomaly_type in results.columns:
                        anomaly_data = results[results[anomaly_type]]
                        if len(anomaly_data) > 0:
                            ax.scatter(anomaly_data['datetime'], anomaly_data[col], 
                                     color=color, label=anomaly_type.replace('_', ' '), 
                                     s=15, alpha=0.8)
                
                ax.set_title(f'{col} über Zeit')
                ax.set_xlabel('Zeit')
                ax.set_ylabel(col)
                ax.legend()
                ax.tick_params(axis='x', rotation=45)
        
        # Anomalie-Score distribution
        if len(axes) > 5:
            ax = axes[5]
            ax.hist(results[~results['is_anomaly']]['anomaly_score'], 
                   bins=50, alpha=0.7, label='Normal', color='blue')
            ax.hist(results[results['is_anomaly']]['anomaly_score'], 
                   bins=50, alpha=0.7, label='Anomaly', color='red')
            ax.set_title('Anomalie Score Verteilung')
            ax.set_xlabel('Anomalie Score')
            ax.set_ylabel('Häufigkeit')
            ax.legend()
                
        plt.tight_layout()
        plt.show()
        
        self._print_detailed_stats(results)
    
    def _print_detailed_stats(self, results):
        """prints anomaly statistics and breakdowns"""
        print(f"\n=== ANOMALIE ANALYSE (ohne Time-Anomalien) ===")
        print(f"Gesamtanzahl Datenpunkte: {len(results)}")
        print(f"Erkannte Anomalien: {results['is_anomaly'].sum()}")
        print(f"Anomalie-Rate: {results['is_anomaly'].mean():.2%}")
        
        print(f"\nAufschlüsselung nach Typ:")
        for anomaly_type in ['isolation_forest_anomaly', 'statistical_anomaly', 'domain_anomaly']:
            if anomaly_type in results.columns:
                count = results[anomaly_type].sum()
                rate = results[anomaly_type].mean()
                print(f"- {anomaly_type.replace('_', ' ').title()}: {count} ({rate:.2%})")
        
        # category-wise anomaly analysis
        if 'type' in results.columns:
            print(f"\n=== ANOMALIEN NACH KATEGORIEN ===")
            anomaly_by_type = results.groupby('type').agg({
                'is_anomaly': ['sum', 'count', 'mean']
            }).round(3)
            print(anomaly_by_type)


if __name__ == "__main__":
    df = pd.read_csv('data/combined_data.csv')
    
    print("=== Anomaly Labeling - Unsupervised ===")
    print("Existing Columns:", df.columns.tolist())
    print(f"Data Points: {len(df)}")
    
    if 'type' in df.columns:
        print(f"Numbers per type:\n{df['type'].value_counts()}")
    
    # anomaly detection and labeling
    labeler = ImprovedAnomalyLabeler(contamination=0.05)
    labeled_data = labeler.detect_anomalies(df, system_type='domestic') #concentration on domestic hot water for ease of use
    
    # visualisation
    labeler.visualize_anomalies(labeled_data)
    
    # Speichern der bereinigten gelabelten Daten
    labeled_data.to_csv('data/labeled_training_data_cleaned.csv', index=False)
    print("\n✅ cleaned and labeled data saved to '../data/labeled_training_data_cleaned.csv'")