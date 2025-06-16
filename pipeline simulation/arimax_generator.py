import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class ARIMAXHeatingGenerator:
    """
    ARIMAX-basierter Generator für Domestic Hot Water Daten - Vereinfacht und mit Model Persistence
    """
    
    def __init__(self, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24)):
        """
        Vereinfachte Konfiguration für schnelles Training
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.scaler_target = StandardScaler()
        self.scaler_exog = StandardScaler()
        self.exog_columns = None
        self.model_path = 'models/arimax_heating_model.pkl'
        
    def prepare_exogenous_variables(self, df):
        """Bereitet minimale externe Variablen vor - stark vereinfacht"""
        df = df.copy()
        
        # Nur die wichtigsten zeitbasierten Features
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        
        # Nur einfache zyklische Features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Boolean Features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = df['nighttime'].astype(int) if 'nighttime' in df.columns else 0
        
        # Nur ein Lag Feature (24h)
        df['temp_lag_24h'] = df['avg'].shift(24)
        
        # Nur ein Rolling Feature
        df['temp_ma_24h'] = df['avg'].rolling(24, min_periods=1).mean()
        
        # Peak Hours (vereinfacht)
        df['is_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                        (df['hour'] >= 19) & (df['hour'] <= 21)).astype(int)
        
        # Minimale exogenous variables
        exog_cols = [
            'hour_sin', 'hour_cos', 'is_weekend', 'is_night', 
            'is_peak', 'temp_lag_24h', 'temp_ma_24h'
        ]
        
        self.exog_columns = exog_cols
        return df[exog_cols].ffill().fillna(0)
    
    def save_model(self):
        """Speichert das trainierte Modell"""
        if self.fitted_model is None:
            print("❌ Kein Modell zum Speichern vorhanden!")
            return False
            
        os.makedirs('models', exist_ok=True)
        
        model_data = {
            'fitted_model': self.fitted_model,
            'scaler_target': self.scaler_target,
            'scaler_exog': self.scaler_exog,
            'exog_columns': self.exog_columns,
            'order': self.order,
            'seasonal_order': self.seasonal_order
        }
        
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"💾 Modell gespeichert: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Fehler beim Speichern: {e}")
            return False
    
    def load_model(self):
        """Lädt ein gespeichertes Modell"""
        if not os.path.exists(self.model_path):
            print(f"📂 Kein gespeichertes Modell gefunden: {self.model_path}")
            return False
            
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.fitted_model = model_data['fitted_model']
            self.scaler_target = model_data['scaler_target']
            self.scaler_exog = model_data['scaler_exog']
            self.exog_columns = model_data['exog_columns']
            self.order = model_data['order']
            self.seasonal_order = model_data['seasonal_order']
            
            print(f"📂 Modell geladen: {self.model_path}")
            print(f"   Order: {self.order}")
            print(f"   Exog Columns: {self.exog_columns}")
            return True
            
        except Exception as e:
            print(f"❌ Fehler beim Laden: {e}")
            return False
    
    def fit(self, df, target_column='avg', force_retrain=False):
        """Trainiert ARIMAX Modell (nur wenn nötig oder force_retrain=True)"""
        
        # Prüfe ob Modell bereits existiert
        if not force_retrain and self.load_model():
            print("✅ Verwende bereits trainiertes Modell!")
            return self.fitted_model
        
        print(f"🚀 Trainiere neues ARIMAX Modell (vereinfacht)...")
        print(f"   Order: {self.order}")
        print(f"   Seasonal Order: {self.seasonal_order}")
        
        # Sortiere nach Zeit
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # Prepare data
        target = df_sorted[target_column].values
        exog = self.prepare_exogenous_variables(df_sorted)
        
        print(f"   Target Shape: {target.shape}")
        print(f"   Exogenous Shape: {exog.shape}")
        print(f"   Exogenous Variables: {list(exog.columns)}")
        
        # Normalisierung
        target_scaled = self.scaler_target.fit_transform(target.reshape(-1, 1)).flatten()
        exog_scaled = self.scaler_exog.fit_transform(exog)
        
        # Verwende einfaches ARIMA mit wenigen exogenen Variablen
        try:
            print(f"🏋️ Erstelle vereinfachtes ARIMAX Modell...")
            
            # Verwende nur die wichtigsten exogenen Variablen für Geschwindigkeit
            key_exog = exog_scaled[:, :4]  # Nur erste 4 Features
            
            self.model = SARIMAX(
                endog=target_scaled,
                exog=key_exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            print(f"🏋️ Fitting Modell (maxiter=50 für Geschwindigkeit)...")
            fitted_result = self.model.fit(disp=False, maxiter=50, method='lbfgs')
            
            self.fitted_model = fitted_result
            
            # Model summary
            print(f"✅ Modell erfolgreich trainiert!")
            print(f"   AIC: {self.fitted_model.aic:.2f}") #type: ignore
            
            # Speichere Modell
            self.save_model()
            
            return self.fitted_model
            
        except Exception as e:
            print(f"❌ Fehler beim Training: {e}")
            print("🔄 Verwende ultra-einfaches ARIMA...")
            
            # Ultra-Fallback: Nur ARIMA ohne exogene Variablen
            try:
                simple_model = ARIMA(endog=target_scaled, order=self.order)
                self.fitted_model = simple_model.fit(maxiter=30) #type: ignore
                print(f"✅ Einfaches ARIMA Modell trainiert!")
                print(f"   AIC: {self.fitted_model.aic:.2f}")
                
                # Anpassung für exog-freies Modell
                self.exog_columns = []
                self.save_model()
                
                return self.fitted_model
                
            except Exception as e2:
                print(f"❌ Auch einfaches ARIMA fehlgeschlagen: {e2}")
                return None
    
    def generate_synthetic_data(self, n_hours=1000, start_date='2024-01-01'):
        """Generiert synthetische Daten mit geladenem/trainiertem Modell"""
        if self.fitted_model is None:
            print("❌ Kein Modell verfügbar! Bitte zuerst trainieren.")
            return None
        
        print(f"🎲 Generiere {n_hours} Stunden synthetische Daten...")
        
        # Create time index
        timestamps = pd.date_range(start=start_date, periods=n_hours, freq='H')
        
        # Create base DataFrame
        df_synthetic = pd.DataFrame({'timestamp': timestamps})
        df_synthetic['datetime'] = df_synthetic['timestamp']
        df_synthetic['avg'] = 0  # Placeholder
        df_synthetic['nighttime'] = (
            (df_synthetic['datetime'].dt.hour >= 23) | 
            (df_synthetic['datetime'].dt.hour <= 5)
        )
        
        try:
            # Versuche Forecast mit exogenen Variablen
            if self.exog_columns and len(self.exog_columns) > 0:
                exog_synthetic = self.prepare_exogenous_variables(df_synthetic)
                # Verwende nur die ersten 4 Features (wie beim Training)
                exog_synthetic_scaled = self.scaler_exog.transform(exog_synthetic)
                key_exog = exog_synthetic_scaled[:, :4]
                
                forecast_result = self.fitted_model.forecast( #type: ignore
                    steps=n_hours,
                    exog=key_exog
                )
            else:
                # Einfacher Forecast ohne exogene Variablen
                forecast_result = self.fitted_model.forecast(steps=n_hours) #type: ignore
            
            # Extrahiere Werte
            if hasattr(forecast_result, 'values'):
                forecast_values = forecast_result.values
            else:
                forecast_values = np.array(forecast_result)
                
        except Exception as e:
            print(f"⚠️ Forecast fehlgeschlagen: {e}")
            print("🔄 Verwende einfachen Forecast...")
            
            forecast_result = self.fitted_model.forecast(steps=n_hours) #type: ignore
            if hasattr(forecast_result, 'values'):
                forecast_values = forecast_result.values
            else:
                forecast_values = np.array(forecast_result)
        
        # Denormalize
        forecast_values = forecast_values.reshape(-1, 1)
        synthetic_temps = self.scaler_target.inverse_transform(forecast_values).flatten()
        
        # Create final DataFrame
        result_df = pd.DataFrame({
            'timestamp': timestamps,
            'avg': synthetic_temps,
            'type': 'domestic',
            'ID': 'arimax_synthetic',
            'nighttime': df_synthetic['nighttime']
        })
        
        # Add derived features
        #np.random.seed()
        result_df['min_v'] = result_df['avg'] - np.random.uniform(3, 8, len(result_df))
        result_df['max_v'] = result_df['avg'] + np.random.uniform(3, 10, len(result_df))
        result_df['T_diff'] = result_df['max_v'] - result_df['min_v']
        
        # Inject simple anomalies
        result_df = self._inject_simple_anomalies(result_df)
        
        print(f"✅ {len(result_df)} synthetische Datenpunkte generiert!")
        return result_df
    
    def _inject_simple_anomalies(self, df, anomaly_rate=0.05):
        """Injiziert einfache Anomalien"""
        df = df.copy()
        n_anomalies = int(len(df) * anomaly_rate)
        
        df['is_anomaly'] = False
        df['anomaly_type'] = 'normal'
        
        if n_anomalies > 0:
            anomaly_indices = np.random.choice(len(df), n_anomalies, replace=False)
            
            for idx in anomaly_indices:
                anomaly_type = np.random.choice(['overheat', 'underheat', 'sensor_error'])
                
                if anomaly_type == 'overheat':
                    df.loc[idx, 'avg'] += np.random.uniform(15, 25)
                elif anomaly_type == 'underheat':
                    df.loc[idx, 'avg'] -= np.random.uniform(10, 20)
                elif anomaly_type == 'sensor_error':
                    df.loc[idx, 'avg'] *= np.random.uniform(0.5, 1.5)
                
                # Update derived values
                df.loc[idx, 'min_v'] = df.loc[idx, 'avg'] - np.random.uniform(3, 8)
                df.loc[idx, 'max_v'] = df.loc[idx, 'avg'] + np.random.uniform(3, 10)
                df.loc[idx, 'T_diff'] = df.loc[idx, 'max_v'] - df.loc[idx, 'min_v']
                df.loc[idx, 'is_anomaly'] = True
                df.loc[idx, 'anomaly_type'] = anomaly_type
        
        return df
    
    def quick_comparison_plot(self, real_df, synthetic_df):
        """Schneller Vergleich zwischen echten und synthetischen Daten"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Temperature distribution
        axes[0].hist(real_df['avg'], bins=30, alpha=0.7, label='Real', density=True, color='blue')
        axes[0].hist(synthetic_df['avg'], bins=30, alpha=0.7, label='Synthetic', density=True, color='red')
        axes[0].set_title('Temperature Distribution')
        axes[0].legend()
        
        # 2. Hourly patterns
        real_hourly = real_df.groupby(pd.to_datetime(real_df['timestamp']).dt.hour)['avg'].mean()
        synth_hourly = synthetic_df.groupby(pd.to_datetime(synthetic_df['timestamp']).dt.hour)['avg'].mean()
        
        axes[1].plot(real_hourly.index, real_hourly.values, 'b-', label='Real', linewidth=2)
        axes[1].plot(synth_hourly.index, synth_hourly.values, 'r--', label='Synthetic', linewidth=2)
        axes[1].set_title('Hourly Pattern')
        axes[1].legend()
        
        # 3. Time series sample (first 3 days)
        sample_real = real_df.head(72)
        sample_synth = synthetic_df.head(72)
        
        axes[2].plot(sample_real['avg'], 'b-', alpha=0.8, label='Real')
        axes[2].plot(sample_synth['avg'], 'r--', alpha=0.8, label='Synthetic')
        axes[2].set_title('3-Day Sample')
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()

def main(retrain=False, generate_hours=1000):
    """
    Hauptfunktion mit Optionen
    
    Args:
        retrain (bool): Erzwingt Neutraining des Modells
        generate_hours (int): Anzahl Stunden für Synthese
    """
    
    # Load data
    try:
        df = pd.read_csv('data/labeled_training_data_cleaned.csv')
        print("✅ Bereinigte Daten geladen")
    except FileNotFoundError:
        try:
            df = pd.read_csv('data/labeled_training_data.csv')
            print("✅ Original gelabelte Daten geladen")
        except FileNotFoundError:
            print("❌ Keine Trainingsdaten gefunden!")
            return None
    
    print(f"Geladene Daten: {df.shape}")
    
    # Filter für domestic hot water
    if 'type' in df.columns:
        df_domestic = df[df['type'] == 'domestic'].copy()
        print(f"Domestic Hot Water Daten: {df_domestic.shape}")
    else:
        df_domestic = df.copy()
    
    # Initialize generator (vereinfacht)
    generator = ARIMAXHeatingGenerator(
        order=(1, 1, 1),           # Einfach
        seasonal_order=(0, 0, 0, 0) # Keine Saisonalität
    )
    
    # Train oder load model
    print(f"\n=== MODELL TRAINING/LOADING ===")
    if retrain:
        print("🔄 Erzwinge Neutraining...")
    
    fitted_model = generator.fit(df_domestic, target_column='avg', force_retrain=retrain)
    
    if fitted_model is None:
        print("❌ Training fehlgeschlagen!")
        return None
    
    # Generate synthetic data
    print(f"\n=== DATENGENERIERUNG ===")
    synthetic_df = generator.generate_synthetic_data(
        n_hours=generate_hours,
        start_date='2024-06-01'
    )
    
    if synthetic_df is None:
        print("❌ Datengenerierung fehlgeschlagen!")
        return None
    
    # Quick comparison
    generator.quick_comparison_plot(df_domestic, synthetic_df)
    
    # Save synthetic data
    os.makedirs('data/synthetic', exist_ok=True)
    output_path = 'data/synthetic/arimax_synthetic_heating_data.csv'
    synthetic_df.to_csv(output_path, index=False)
    
    print(f"\n✅ ARIMAX Generator abgeschlossen!")
    print(f"📁 Synthetische Daten: {synthetic_df.shape}")
    print(f"💾 Gespeichert: {output_path}")
    
    # Anomalie-Statistiken
    if 'is_anomaly' in synthetic_df.columns:
        anomaly_count = synthetic_df['is_anomaly'].sum()
        anomaly_rate = synthetic_df['is_anomaly'].mean()
        print(f"🚨 Generierte Anomalien: {anomaly_count} ({anomaly_rate:.1%})")
    
    return synthetic_df

if __name__ == "__main__":
    # Standard-Ausführung (verwendet gespeichertes Modell falls vorhanden)
    #synthetic_data = main(retrain=False, generate_hours=1000)
    
    # Für Neutraining verwenden Sie:
     synthetic_data = main(retrain=True, generate_hours=1000)