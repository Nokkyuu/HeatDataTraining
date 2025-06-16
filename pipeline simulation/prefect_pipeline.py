import pandas as pd
import numpy as np
from prefect import flow, task, serve
from typing import Dict, Tuple, Any
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from arimax_generator import ARIMAXHeatingGenerator 
from ensemble_detector import EnsembleAnomalyDetector, IsolationForestDetector, StatisticalDetector

@task(name="generate_heating_batch")
def generate_heating_batch(batch_size: int = 1000, anomaly_rate: float = 0.05) -> pd.DataFrame:
    """
    Generiert einen Batch synthetischer Heizungsdaten mit Anomalie-Labels
    """
    print(f"🏭 Generiere {batch_size} Datenpunkte mit {anomaly_rate:.1%} Anomalie-Rate...")
    
    # KORRIGIERT: Verwende echte Klasse und Methoden
    generator = ARIMAXHeatingGenerator( #type: ignore
        order=(1, 1, 1),           # Einfach wie im Original
        seasonal_order=(0, 0, 0, 0) # Keine Saisonalität
    )
    
    # KORRIGIERT: Lade oder trainiere das Modell erst
    print("🤖 Lade/Trainiere ARIMAX Modell...")
    try:
        # Versuche echte Trainingsdaten zu laden
        try:
            df_real = pd.read_csv('data/labeled_training_data_cleaned.csv')
        except FileNotFoundError:
            df_real = pd.read_csv('data/labeled_training_data.csv')
        
        # Filter domestic
        if 'type' in df_real.columns:
            df_real = df_real[df_real['type'] == 'domestic'].copy()
        
        # Trainiere/Lade Modell
        fitted_model = generator.fit(df_real, target_column='avg', force_retrain=False)
        
        if fitted_model is None:
            raise Exception("ARIMAX Training fehlgeschlagen")
            
    except Exception as e:
        print(f"⚠️ ARIMAX Modell-Problem: {e}")
        print("🔄 Verwende Dummy-Daten für Training...")
        
        # Fallback: Erstelle minimale Dummy-Daten
        dummy_timestamps = pd.date_range('2024-01-01', periods=500, freq='H')
        dummy_df = pd.DataFrame({
            'timestamp': dummy_timestamps,
            'avg': np.random.normal(50, 10, 500),
            'type': 'domestic'
        })
        
        fitted_model = generator.fit(dummy_df, target_column='avg', force_retrain=True)
    
    # KORRIGIERT: Verwende echte Methode mit korrekten Parametern
    df_batch = generator.generate_synthetic_data(
        n_hours=batch_size,  # ✅ Korrekte Parameter
        start_date='2024-06-01'
    )
    
    if df_batch is None:
        raise Exception("Datengenerierung fehlgeschlagen!")
    
    # KORRIGIERT: Anomalie-Rate anpassen falls nötig
    current_anomaly_rate = df_batch['is_anomaly'].mean() if 'is_anomaly' in df_batch.columns else 0
    print(f"   Generierte Anomalie-Rate: {current_anomaly_rate:.1%} (Ziel: {anomaly_rate:.1%})")
    
    # Falls Anomalie-Rate stark abweicht, re-inject
    if abs(current_anomaly_rate - anomaly_rate) > 0.02:  # Mehr als 2% Abweichung
        print(f"🔄 Passe Anomalie-Rate an...")
        df_batch = generator._inject_simple_anomalies(df_batch, anomaly_rate=anomaly_rate)
    
    # Nehme nur die ersten batch_size Zeilen
    df_batch = df_batch.head(batch_size).copy()
    
    print(f"✅ Batch generiert: {len(df_batch)} Zeilen")
    print(f"   Finale Anomalien: {df_batch['is_anomaly'].sum()} ({df_batch['is_anomaly'].mean():.1%})")
    
    return df_batch

@task(name="clean_data_for_production")
def clean_data_for_production(df_with_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Entfernt Anomalie-Injection Spalten für Production-Simulation
    """
    print("🧹 Bereite Daten für Production vor (entferne Anomalie-Labels)...")
    
    # KORRIGIERT: Spalten basierend auf ARIMAX Output
    production_columns = [
        'timestamp', 'avg', 'min_v', 'max_v', 'T_diff', 'type', 'ID', 'nighttime'
    ]
    
    # Nur verfügbare Spalten verwenden
    available_columns = [col for col in production_columns if col in df_with_labels.columns]
    
    clean_data = df_with_labels[available_columns].copy()
    
    print(f"✅ Daten bereinigt: {len(clean_data)} Zeilen, {len(available_columns)} Spalten")
    print(f"   Verfügbare Spalten: {available_columns}")
    print(f"   Entfernte Spalten: is_anomaly, anomaly_type")
    
    return clean_data

@task(name="load_ensemble_model")
def load_ensemble_model() -> EnsembleAnomalyDetector: #type: ignore
    """
    Lädt das trainierte Ensemble Model
    """
    print("🤖 Lade Ensemble Model...")
    
    ensemble = EnsembleAnomalyDetector()#type: ignore
    
    # Füge Detektoren hinzu (ohne LSTM wie optimiert)
    ensemble.add_detector(IsolationForestDetector(contamination=0.1), weight=1.0)#type: ignore
    ensemble.add_detector(StatisticalDetector(z_threshold=2.5), weight=1.0)#type: ignore
    
    # Lade trainierte Models
    try:
        # Versuche echte Trainingsdaten zu laden
        try:
            df_real = pd.read_csv('data/labeled_training_data_cleaned.csv')
        except FileNotFoundError:
            df_real = pd.read_csv('data/labeled_training_data.csv')
        
        if 'type' in df_real.columns:
            df_real = df_real[df_real['type'] == 'domestic'].copy()
        
        X_train = df_real['avg'].values
        y_train = df_real['is_anomaly'].astype(int).values if 'is_anomaly' in df_real.columns else None
        
        ensemble.fit(X_train, y_train, force_retrain=False)  # Versuche zu laden #type: ignore
        
        print("✅ Ensemble Model geladen")
        return ensemble
        
    except Exception as e:
        print(f"❌ Fehler beim Laden: {e}")
        print("🔄 Verwende Dummy-Training...")
        
        # Fallback: Dummy-Training
        dummy_X = np.random.normal(50, 10, 1000)
        dummy_y = np.random.choice([0, 1], 1000, p=[0.95, 0.05])
        
        ensemble.fit(dummy_X, dummy_y, force_retrain=True)
        print("✅ Ensemble Model mit Dummy-Daten trainiert")
        return ensemble

@task(name="detect_anomalies_ensemble")
def detect_anomalies_ensemble(clean_data: pd.DataFrame, ensemble: EnsembleAnomalyDetector) -> Dict[str, np.ndarray]:#type: ignore
    """
    Führt Anomalie-Detection mit dem Ensemble durch
    """
    print("🔍 Führe Anomalie-Detection durch...")
    
    # Verwende 'avg' Spalte für Detection
    X = clean_data['avg'].values
    
    # Ensemble Prediction
    predictions = ensemble.predict(X)#type: ignore
    
    detected_count = predictions['ensemble'].sum()
    detection_rate = predictions['ensemble'].mean()
    
    print(f"✅ Anomalie-Detection abgeschlossen")
    print(f"   Erkannte Anomalien: {detected_count} ({detection_rate:.1%})")
    
    # KORRIGIERT: Detector-Namen anpassen
    print(f"   Isolation Forest: {predictions.get('Isolation_Forest', np.array([])).sum()} erkannt")
    print(f"   Statistical: {predictions.get('Statistical', np.array([])).sum()} erkannt")
    
    return predictions

@task(name="extract_ground_truth")
def extract_ground_truth(df_with_labels: pd.DataFrame) -> Dict[str, Any]:
    """
    Extrahiert Ground Truth Labels aus den ursprünglichen Daten
    """
    print("📋 Extrahiere Ground Truth Labels...")
    
    # KORRIGIERT: Sichere Extraktion
    labels = df_with_labels['is_anomaly'].astype(int).values if 'is_anomaly' in df_with_labels.columns else np.zeros(len(df_with_labels))
    anomaly_types = df_with_labels['anomaly_type'].values if 'anomaly_type' in df_with_labels.columns else np.array(['unknown'] * len(df_with_labels))
    
    ground_truth = {
        'labels': labels,
        'anomaly_types': anomaly_types,
        'total_anomalies': labels.sum(), #type: ignore
        'anomaly_rate': labels.mean() #type: ignore
    }
    
    print(f"✅ Ground Truth extrahiert")
    print(f"   Tatsächliche Anomalien: {ground_truth['total_anomalies']} ({ground_truth['anomaly_rate']:.1%})")
    
    # Anomalie-Typ Verteilung
    if 'anomaly_type' in df_with_labels.columns:
        anomaly_type_counts = df_with_labels[df_with_labels['is_anomaly'] == True]['anomaly_type'].value_counts()
        print(f"   Anomalie-Typen: {dict(anomaly_type_counts)}")
    
    return ground_truth

@task(name="validate_detection_quality")
def validate_detection_quality(predictions: Dict[str, np.ndarray], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validiert die Detection-Qualität gegen Ground Truth
    """
    print("📊 Validiere Detection-Qualität...")
    
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    
    y_true = ground_truth['labels']
    y_pred = predictions['ensemble']
    
    # Metriken berechnen
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Detaillierte Analyse
    correct_detections = np.intersect1d(
        np.where(y_true == 1)[0], 
        np.where(y_pred == 1)[0]
    )
    missed_anomalies = np.setdiff1d(
        np.where(y_true == 1)[0], 
        np.where(y_pred == 1)[0]
    )
    false_alarms = np.setdiff1d(
        np.where(y_pred == 1)[0], 
        np.where(y_true == 1)[0]
    )
    
    validation_results = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'correct_detections': len(correct_detections),
        'missed_anomalies': len(missed_anomalies),
        'false_alarms': len(false_alarms),
        'validation_passed': recall >= 0.7  # Prefect Threshold
    }
    
    print(f"✅ Validation abgeschlossen")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1-Score: {f1:.3f}")
    print(f"   Validation Status: {'✅ PASSED' if validation_results['validation_passed'] else '❌ FAILED'}")
    
    # KORRIGIERT: Einzelne Detector-Analyse mit korrekten Namen
    print(f"\n🔍 EINZELNE DETECTOR PERFORMANCE:")
    for detector_name in ['Isolation_Forest', 'Statistical']:
        if detector_name in predictions:
            detector_pred = predictions[detector_name]
            detector_recall = recall_score(y_true, detector_pred, zero_division=0)
            detector_precision = precision_score(y_true, detector_pred, zero_division=0)
            
            print(f"   {detector_name}:")
            print(f"     Precision: {detector_precision:.3f}, Recall: {detector_recall:.3f}")
            print(f"     Erkannte: {detector_pred.sum()}, Davon korrekt: {np.sum((detector_pred == 1) & (y_true == 1))}")
    
    return validation_results

@task(name="generate_validation_report")
def generate_validation_report(
    validation_results: Dict[str, Any],
    clean_data: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict[str, Any],
    save_report: bool = True
) -> Dict[str, Any]:
    """
    Generiert detaillierten Validation Report mit Visualisierung
    """
    print("📊 Generiere Validation Report...")
    
    # Report Daten sammeln
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'batch_size': len(clean_data),
        'metrics': validation_results,
        'summary': {
            'injected_anomalies': ground_truth['total_anomalies'],
            'detected_anomalies': predictions['ensemble'].sum(),
            'correct_detections': validation_results['correct_detections'],
            'missed_anomalies': validation_results['missed_anomalies'],
            'false_alarms': validation_results['false_alarms']
        }
    }
    
    # Visualisierung erstellen
    if save_report:
        create_validation_visualization(clean_data, predictions, ground_truth, validation_results)
    
    print(f"✅ Validation Report generiert")
    print(f"   Timestamp: {report['timestamp']}")
    print(f"   Batch Size: {report['batch_size']}")
    print(f"   Overall Status: {'SUCCESS' if validation_results['validation_passed'] else 'FAILED'}")
    
    return report

def create_validation_visualization(
    clean_data: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict[str, Any],
    validation_results: Dict[str, Any]
):
    """
    Erstellt Plotly Visualisierung der Validation Ergebnisse
    """
    
    # Sample für Visualisierung (erste 200 Punkte)
    sample_size = min(1000, len(clean_data))
    X_sample = clean_data['avg'].values[:sample_size]
    y_true_sample = ground_truth['labels'][:sample_size]
    y_pred_sample = predictions['ensemble'][:sample_size]
    
    # Subplots erstellen
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Temperature mit Anomalien (ARIMAX Daten)',
            'Detection Status über Zeit',
            'Confusion Matrix',
            'Performance Metriken'
        ],
        specs=[[{"colspan": 2}, None],
               [{"type": "xy"}, {"type": "xy"}]]
    )
    
    # Plot 1: Temperature Zeitreihe
    fig.add_trace(
        go.Scatter(x=list(range(sample_size)), y=X_sample, 
                  mode='lines', name='Temperature', 
                  line=dict(color='blue', width=1)),
        row=1, col=1
    )
    
    # Echte Anomalien
    true_anomaly_indices = np.where(y_true_sample == 1)[0]
    if len(true_anomaly_indices) > 0:
        fig.add_trace(
            go.Scatter(x=true_anomaly_indices, y=X_sample[true_anomaly_indices],
                      mode='markers', name='ARIMAX Injizierte Anomalien',
                      marker=dict(color='red', size=8, symbol='circle')),
            row=1, col=1
        )
    
    # Erkannte Anomalien
    detected_indices = np.where(y_pred_sample == 1)[0]
    if len(detected_indices) > 0:
        fig.add_trace(
            go.Scatter(x=detected_indices, y=X_sample[detected_indices],
                      mode='markers', name='Ensemble Erkannte Anomalien',
                      marker=dict(color='orange', size=6, symbol='triangle-up')),
            row=1, col=1
        )
    
    # Korrekt erkannte (grün)
    correct_indices = np.intersect1d(true_anomaly_indices, detected_indices)
    if len(correct_indices) > 0:
        fig.add_trace(
            go.Scatter(x=correct_indices, y=X_sample[correct_indices],
                      mode='markers', name='Korrekt erkannt',
                      marker=dict(color='green', size=10, symbol='star')),
            row=1, col=1
        )
    
    # Plot 2: Confusion Matrix
    cm_data = [[validation_results['true_negatives'], validation_results['false_positives']],
               [validation_results['false_negatives'], validation_results['true_positives']]]
    
    fig.add_trace(
        go.Heatmap(z=cm_data, 
                  x=['Predicted Normal', 'Predicted Anomaly'],
                  y=['Actual Normal', 'Actual Anomaly'],
                  colorscale='Blues',
                  showscale=False,
                  text=cm_data, texttemplate="%{text}"),
        row=2, col=1
    )
    
    # Plot 3: Metriken Bar Chart
    metrics_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    metrics_values = [
        validation_results['precision'],
        validation_results['recall'],
        validation_results['f1_score'],
        validation_results['accuracy']
    ]
    
    fig.add_trace(
        go.Bar(x=metrics_names, y=metrics_values,
               name='Performance Metriken',
               marker_color=['green' if v >= 0.7 else 'orange' if v >= 0.5 else 'red' for v in metrics_values]),
        row=2, col=2
    )
    
    # Layout anpassen
    fig.update_layout(
        height=800,
        title_text=f"Prefect Pipeline Validation Report (ARIMAX + Ensemble) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        showlegend=True
    )
    
    # Speichere als HTML
    os.makedirs('reports', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = f'reports/arimax_ensemble_validation_{timestamp}.html'
    fig.write_html(filepath)
    
    print(f"📊 Visualisierung gespeichert: {filepath}")

@flow(name="arimax_ensemble_validation_pipeline")
def arimax_ensemble_validation_pipeline(
    batch_size: int = 1000,
    anomaly_rate: float = 0.05,
    save_report: bool = True
) -> Dict[str, Any]:
    """
    KORRIGIERTE Pipeline: ARIMAX Generator + Ensemble Detection + Validation
    """
    print("🚀 === PREFECT ARIMAX + ENSEMBLE VALIDATION PIPELINE ===")
    print(f"Batch Size: {batch_size}, Anomaly Rate: {anomaly_rate:.1%}")
    
    # Task 1: Generiere Daten mit ARIMAX
    raw_data_with_labels = generate_heating_batch(batch_size, anomaly_rate)
    
    # Task 2: Bereite Production-Daten vor
    clean_data = clean_data_for_production(raw_data_with_labels)
    
    # Task 3: Lade Ensemble Model
    ensemble = load_ensemble_model()
    
    # Task 4: Führe Anomalie-Detection durch
    predictions = detect_anomalies_ensemble(clean_data, ensemble)
    
    # Task 5: Extrahiere Ground Truth
    ground_truth = extract_ground_truth(raw_data_with_labels)
    
    # Task 6: Validiere Detection-Qualität
    validation_results = validate_detection_quality(predictions, ground_truth)
    
    # Task 7: Generiere Report
    report = generate_validation_report(
        validation_results, clean_data, predictions, ground_truth, save_report
    )
    
    # Final Status
    status = "SUCCESS" if validation_results['validation_passed'] else "FAILED"
    print(f"\n🏁 ARIMAX + Ensemble Pipeline beendet: {status}")
    print(f"🎯 Final Metrics: P={validation_results['precision']:.3f}, R={validation_results['recall']:.3f}, F1={validation_results['f1_score']:.3f}")
    
    return report

@flow(name="scheduled_arimax_ensemble_validation")
def scheduled_validation_flow(
    batch_size: int = 1000,
    anomaly_rate: float = 0.05,
    save_report: bool = True
):
    """
    Scheduled wrapper für die Pipeline
    """
    print(f"🕒 Scheduled Run gestartet: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    result = arimax_ensemble_validation_pipeline(
        batch_size=batch_size,
        anomaly_rate=anomaly_rate, 
        save_report=save_report
    )
    
    status = "SUCCESS" if result['metrics']['validation_passed'] else "FAILED"
    print(f"🏁 Scheduled Run beendet: {status}")
    
    return result

class ContinuousPipelineRunner:
    """
    Einfacher kontinuierlicher Pipeline Runner
    """
    def __init__(self, interval_minutes: int = 120):
        self.interval_minutes = interval_minutes
        self.running = True
        self.run_count = 0
        
        # Graceful shutdown
        import signal
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        print(f"\n🛑 Shutdown Signal empfangen...")
        self.running = False
    
    def run_continuous(self):
        """
        Läuft kontinuierlich mit definiertem Intervall
        """
        print(f"🚀 === KONTINUIERLICHE ANOMALIE-VALIDIERUNG ===")
        print(f"⏰ Intervall: {self.interval_minutes} Minuten")
        print(f"🛑 Stoppen mit Ctrl+C")
        print("="*60)
        
        while self.running:
            self.run_count += 1
            
            print(f"\n🔄 === RUN #{self.run_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            
            try:
                # Pipeline ausführen
                result = arimax_ensemble_validation_pipeline(
                    batch_size=1000,
                    anomaly_rate=0.05,
                    save_report=True
                )
                
                # Status anzeigen
                if result['metrics']['validation_passed']:
                    print(f"✅ Run #{self.run_count} ERFOLGREICH")
                    print(f"   🎯 Metrics: P={result['metrics']['precision']:.3f}, R={result['metrics']['recall']:.3f}, F1={result['metrics']['f1_score']:.3f}")
                    print(f"   📊 Detected: {result['summary']['detected_anomalies']}/{result['summary']['injected_anomalies']} Anomalien")
                else:
                    print(f"❌ Run #{self.run_count} FEHLGESCHLAGEN")
                    print(f"   ⚠️ Recall {result['metrics']['recall']:.3f} < 0.7 Threshold")
                    print(f"   📊 Detected: {result['summary']['detected_anomalies']}/{result['summary']['injected_anomalies']} Anomalien")
                
                print(f"   📁 Report: reports/arimax_ensemble_validation_{result['timestamp'].replace(' ', '_').replace(':', '')}.html")
                
            except Exception as e:
                print(f"❌ Run #{self.run_count} FEHLER: {e}")
                # Kurzer Traceback für Debug
                import traceback
                print("🔍 Kurzer Error-Trace:")
                traceback.print_exc(limit=3)
            
            if self.running:
                print(f"\n💤 Warte {self.interval_minutes} Minuten bis zum nächsten Run...")
                print(f"🛑 Jederzeit stoppen mit Ctrl+C")
                
                # Sleep mit Interrupt-Check (alle 10 Sekunden prüfen)
                total_seconds = self.interval_minutes * 60
                for i in range(0, total_seconds, 10):
                    if not self.running:
                        break
                    
                    remaining_minutes = (total_seconds - i) // 60
                    if i % 60 == 0 and remaining_minutes > 0:  # Jede Minute update
                        print(f"⏳ Noch {remaining_minutes} Minuten...")
                    
                    time.sleep(min(10, total_seconds - i))
        
        print(f"\n🏁 Pipeline gestoppt nach {self.run_count} Runs")

def run_pipeline_once():
    """
    Führt die Pipeline einmal aus (für Testing)
    """
    print("🚀 === SINGLE PIPELINE RUN ===")
    
    try:
        result = arimax_ensemble_validation_pipeline(
            batch_size=1000,
            anomaly_rate=0.05,
            save_report=True
        )
        
        print(f"\n✅ Pipeline erfolgreich abgeschlossen!")
        print(f"📊 Report Timestamp: {result['timestamp']}")
        print(f"🎯 Validation Status: {'PASSED' if result['metrics']['validation_passed'] else 'FAILED'}")
        print(f"📈 Metrics: P={result['metrics']['precision']:.3f}, R={result['metrics']['recall']:.3f}, F1={result['metrics']['f1_score']:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Pipeline Fehler: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_continuous_pipeline(interval_minutes: int = 120):
    """
    Startet kontinuierliche Pipeline
    """
    runner = ContinuousPipelineRunner(interval_minutes=interval_minutes)
    runner.run_continuous()

def interactive_menu():
    """
    Interaktives Menü für Pipeline Control
    """
    print("🎛️ === PREFECT PIPELINE CONTROL ===")
    
    while True:
        print(f"\nOptionen:")
        print("1. Single Run (Pipeline einmal ausführen)")
        print("2. Continuous Run (Kontinuierlich ausführen)")
        print("3. Exit")
        
        choice = input("\nWähle Option (1-3): ").strip()
        
        if choice == "1":
            print("\n" + "="*50)
            run_pipeline_once()
            
        elif choice == "2":
            interval_input = input("Intervall in Minuten (default: 120): ").strip()
            
            try:
                interval = int(interval_input) if interval_input else 120
                if interval < 1:
                    print("❌ Intervall muss mindestens 1 Minute sein!")
                    continue
                    
                print(f"\n🚀 Starte kontinuierliche Pipeline mit {interval} Min Intervall...")
                print("="*50)
                run_continuous_pipeline(interval_minutes=interval)
                
            except ValueError:
                print("❌ Ungültiges Intervall! Bitte Zahl eingeben.")
                
        elif choice == "3":
            print("👋 Auf Wiedersehen!")
            break
            
        else:
            print("❌ Ungültige Option! Bitte 1, 2 oder 3 wählen.")


if __name__ == "__main__":
    import sys
    
    # Command line arguments prüfen
    if len(sys.argv) > 1:
        if sys.argv[1] == "continuous":
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 120
            print(f"🚀 Starte kontinuierliche Pipeline (alle {interval} Min)...")
            run_continuous_pipeline(interval_minutes=interval)
            
        elif sys.argv[1] == "once":
            print("🚀 Führe Pipeline einmal aus...")
            run_pipeline_once()
            
        else:
            print("❌ Unbekanntes Argument. Verwende 'once' oder 'continuous [interval]'")
            sys.exit(1)
    else:
        # Interaktives Menü als default
        interactive_menu()