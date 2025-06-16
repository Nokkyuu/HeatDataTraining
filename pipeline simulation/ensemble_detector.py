import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
import pickle
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class BaseDetector(ABC):
    """
    Basis-Klasse für alle Anomalie-Detektoren
    """
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.scaler = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BaseDetector':
        """Trainiert den Detector"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vorhersage: 0=normal, 1=anomaly"""
        pass
    
    @abstractmethod
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Anomalie-Score (höher = anomaler)"""
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """Speichert den trainierten Detector"""
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """Lädt einen trainierten Detector"""
        pass

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder für Zeitreihen-Anomalie-Erkennung
    """
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, 
                 num_layers: int = 2, sequence_length: int = 24):
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Encoder
        encoded, (hidden, cell) = self.encoder_lstm(x)
        
        # Decoder
        # Repeat the last hidden state for sequence_length
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        
        # Output layer
        output = self.output_layer(decoded)
        
        return output

class LSTMDetector(BaseDetector):
    """
    LSTM Autoencoder Anomalie-Detector mit vollständiger Persistierung
    """
    
    def __init__(self, sequence_length: int = 24, hidden_dim: int = 64):
        super().__init__("LSTM_Autoencoder")
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Erstellt überlappende Sequenzen"""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
        return np.array(sequences)
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'LSTMDetector':
        """
        Trainiert LSTM Autoencoder nur auf normalen Daten
        """
        print(f"🚀 Trainiere {self.name}...")
        
        # Filter nur normale Daten für Training
        if y is not None:
            normal_data = X[y == 0]  # 0 = normal
            print(f"   Verwende {len(normal_data)} normale Samples von {len(X)} total")
        else:
            normal_data = X
            print(f"   Verwende alle {len(X)} Samples (unüberwacht)")
        
        # Normalisierung
        normal_data_scaled = self.scaler.fit_transform(normal_data.reshape(-1, 1)).flatten()
        
        # Erstelle Sequenzen
        sequences = self._create_sequences(normal_data_scaled)
        print(f"   Erstellt {len(sequences)} Sequenzen der Länge {self.sequence_length}")
        
        if len(sequences) == 0:
            raise ValueError("Nicht genügend Daten für Sequenz-Erstellung!")
        
        # PyTorch Setup
        sequences_tensor = torch.FloatTensor(sequences).unsqueeze(-1).to(self.device)
        
        # Model
        self.model = LSTMAutoencoder(
            input_dim=1,
            hidden_dim=self.hidden_dim,
            sequence_length=self.sequence_length
        ).to(self.device)
        
        # Training
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        dataset = TensorDataset(sequences_tensor, sequences_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.model.train()
        epochs = 50
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"   Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Bestimme Threshold auf Trainingsdaten
        self.model.eval()
        with torch.no_grad():
            train_reconstruction_errors = []
            for batch_x, _ in dataloader:
                reconstructed = self.model(batch_x)
                mse = nn.MSELoss(reduction='none')(reconstructed, batch_x)
                mse = mse.mean(dim=[1, 2])  # Mean über sequence und features
                train_reconstruction_errors.extend(mse.cpu().numpy())
        
        # Threshold als 95. Percentile der Trainingsfehler
        self.threshold = np.percentile(train_reconstruction_errors, 95)
        print(f"   Threshold gesetzt auf: {self.threshold:.4f}")
        
        self.is_fitted = True
        print(f"✅ {self.name} Training abgeschlossen!")
        return self
    
    def _calculate_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Berechnet Rekonstruktionsfehler"""
        if not self.is_fitted:
            raise ValueError("Model muss erst trainiert werden!")
        
        # Normalisierung
        X_scaled = self.scaler.transform(X.reshape(-1, 1)).flatten()
        
        # Erstelle Sequenzen
        sequences = self._create_sequences(X_scaled)
        
        if len(sequences) == 0:
            # Fallback für zu kurze Sequenzen
            return np.full(len(X), self.threshold * 2)  # type: ignore # Markiere als anomal
        
        # PyTorch prediction
        sequences_tensor = torch.FloatTensor(sequences).unsqueeze(-1).to(self.device)
        
        self.model.eval() # type: ignore
        with torch.no_grad():
            reconstructed = self.model(sequences_tensor) # type: ignore
            mse = nn.MSELoss(reduction='none')(reconstructed, sequences_tensor)
            reconstruction_errors = mse.mean(dim=[1, 2]).cpu().numpy()
        
        # Erweitere auf ursprüngliche Länge
        full_errors = np.full(len(X), reconstruction_errors[-1])  # Letzter Wert für padding
        full_errors[self.sequence_length-1:self.sequence_length-1+len(reconstruction_errors)] = reconstruction_errors
        
        return full_errors
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Anomalie-Score (höher = anomaler)"""
        return self._calculate_reconstruction_error(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vorhersage: 0=normal, 1=anomaly"""
        scores = self.decision_function(X)
        return (scores > self.threshold).astype(int)
    
    def save(self, filepath: str):
        """Speichert den trainierten Detector"""
        if not self.is_fitted:
            raise ValueError("Model muss erst trainiert werden!")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(), # type: ignore
            'scaler': self.scaler,
            'threshold': self.threshold,
            'sequence_length': self.sequence_length,
            'hidden_dim': self.hidden_dim
        }
        
        torch.save(save_dict, filepath)
        print(f"💾 {self.name} gespeichert: {filepath}")
    
    def load(self, filepath: str):
        """Lädt einen trainierten Detector - PyTorch 2.6+ Compatible"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model nicht gefunden: {filepath}")
        
        # Sicherer Import und Referenz
        import torch as torch_module
        
        save_dict = None
        error_messages = []
        
        # Strategie 1: weights_only=False (einfachster Fix)
        try:
            save_dict = torch_module.load(filepath, map_location=self.device, weights_only=False)
            print(f"   ✅ Geladen mit weights_only=False")
            
        except Exception as e1:
            error_messages.append(f"weights_only=False: {str(e1)}")
            
            # Strategie 2: Legacy load (keine weights_only Parameter)
            try:
                save_dict = torch_module.load(filepath, map_location=self.device)
                print(f"   ✅ Geladen mit Legacy-Modus")
                
            except Exception as e2:
                error_messages.append(f"legacy: {str(e2)}")
                
                # Strategie 3: Manueller safe_globals
                try:
                    # Direkte Allowlist ohne add_safe_globals
                    save_dict = torch_module.load(
                        filepath, 
                        map_location=self.device, 
                        weights_only=True,
                        pickle_module=pickle  # Expliziter pickle
                    )
                    print(f"   ✅ Geladen mit explizitem pickle")
                    
                except Exception as e3:
                    error_messages.append(f"explicit_pickle: {str(e3)}")
                    
                    # Wenn alles fehlschlägt, gib detaillierte Fehlermeldung
                    raise Exception(f"Alle Load-Strategien fehlgeschlagen:\n" + 
                                  "\n".join([f"  - {msg}" for msg in error_messages]))
        
        # Validiere save_dict
        if save_dict is None:
            raise Exception("save_dict ist None - unerwarteter Fehler")
        
        required_keys = ['model_state_dict', 'scaler', 'threshold', 'sequence_length', 'hidden_dim']
        missing_keys = [key for key in required_keys if key not in save_dict]
        if missing_keys:
            raise Exception(f"Fehlende Keys in save_dict: {missing_keys}")
        
        # Recreate model
        self.model = LSTMAutoencoder(
            input_dim=1,
            hidden_dim=save_dict['hidden_dim'],
            sequence_length=save_dict['sequence_length']
        ).to(self.device)
        
        self.model.load_state_dict(save_dict['model_state_dict'])
        self.scaler = save_dict['scaler']
        self.threshold = save_dict['threshold']
        self.sequence_length = save_dict['sequence_length']
        self.hidden_dim = save_dict['hidden_dim']
        
        self.model.eval()
        self.is_fitted = True
        print(f"📂 {self.name} erfolgreich geladen: {filepath}")

class IsolationForestDetector(BaseDetector):
    """
    Wrapper für Isolation Forest mit vollständiger Persistierung
    """
    
    def __init__(self, contamination: float = 0.1):
        super().__init__("Isolation_Forest")
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'IsolationForestDetector':
        print(f"🚀 Trainiere {self.name}...")
        
        # Normalisierung
        X_scaled = self.scaler.fit_transform(X.reshape(-1, 1))
        
        # Training
        self.model = IsolationForest(
            contamination=self.contamination,
            #random_state=42,
            n_estimators=100
        )
        self.model.fit(X_scaled)
        
        self.is_fitted = True
        print(f"✅ {self.name} Training abgeschlossen!")
        return self
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model muss erst trainiert werden!")
        
        X_scaled = self.scaler.transform(X.reshape(-1, 1))
        # Isolation Forest gibt negative Scores für Anomalien
        # Invertieren für Konsistenz (höher = anomaler)
        return -self.model.decision_function(X_scaled) # type: ignore
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model muss erst trainiert werden!")
        
        X_scaled = self.scaler.transform(X.reshape(-1, 1))
        predictions = self.model.predict(X_scaled) # type: ignore
        # Isolation Forest: -1=anomaly, 1=normal
        # Konvertiere zu: 1=anomaly, 0=normal
        return (predictions == -1).astype(int)
    
    def save(self, filepath: str):
        """Speichert Isolation Forest Model"""
        if not self.is_fitted:
            raise ValueError("Model muss erst trainiert werden!")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'contamination': self.contamination,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"💾 {self.name} gespeichert: {filepath}")
    
    def load(self, filepath: str):
        """Lädt Isolation Forest Model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model nicht gefunden: {filepath}")
        
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)      
        self.model = save_dict['model']
        self.scaler = save_dict['scaler']
        self.contamination = save_dict['contamination']
        self.is_fitted = save_dict['is_fitted']
        
        print(f"📂 {self.name} geladen: {filepath}")

class StatisticalDetector(BaseDetector):
    """
    Statistischer Anomalie-Detector mit vollständiger Persistierung
    """
    
    def __init__(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        super().__init__("Statistical")
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.mean = None
        self.std = None
        self.q1 = None
        self.q3 = None
        self.iqr = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'StatisticalDetector':
        print(f"🚀 Trainiere {self.name}...")
        
        # Berechne Statistiken nur auf normalen Daten
        if y is not None:
            normal_data = X[y == 0]
            print(f"   Verwende {len(normal_data)} normale Samples von {len(X)} total")
        else:
            normal_data = X
            print(f"   Verwende alle {len(X)} Samples")
        
        self.mean = np.mean(normal_data)
        self.std = np.std(normal_data)
        self.q1 = np.percentile(normal_data, 25)
        self.q3 = np.percentile(normal_data, 75)
        self.iqr = self.q3 - self.q1
        
        print(f"   Mean: {self.mean:.2f}, Std: {self.std:.2f}")
        print(f"   Q1: {self.q1:.2f}, Q3: {self.q3:.2f}, IQR: {self.iqr:.2f}")
        print(f"   Z-Threshold: {self.z_threshold}, IQR-Multiplier: {self.iqr_multiplier}")
        
        self.is_fitted = True
        print(f"✅ {self.name} Training abgeschlossen!")
        return self
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model muss erst trainiert werden!")
        
        # Z-Score
        z_scores = np.abs((X - self.mean) / self.std) # type: ignore
        
        # IQR-basierte Outlier-Erkennung
        lower_bound = self.q1 - self.iqr_multiplier * self.iqr # type: ignore
        upper_bound = self.q3 + self.iqr_multiplier * self.iqr # type: ignore
        iqr_outliers = (X < lower_bound) | (X > upper_bound)
        
        # Kombiniere beide Scores (normalisiert)
        scores = z_scores / self.z_threshold + iqr_outliers.astype(float)
        
        return scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model muss erst trainiert werden!")
            
        # Anomalie wenn Z-Score > threshold ODER IQR-Outlier
        z_anomalies = np.abs((X - self.mean) / self.std) > self.z_threshold # type: ignore
        iqr_anomalies = (X < (self.q1 - self.iqr_multiplier * self.iqr)) | (X > (self.q3 + self.iqr_multiplier * self.iqr)) #type: ignore
        
        return (z_anomalies | iqr_anomalies).astype(int)
    
    def save(self, filepath: str):
        """Speichert Statistical Model"""
        if not self.is_fitted:
            raise ValueError("Model muss erst trainiert werden!")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'mean': self.mean,
            'std': self.std,
            'q1': self.q1,
            'q3': self.q3,
            'iqr': self.iqr,
            'z_threshold': self.z_threshold,
            'iqr_multiplier': self.iqr_multiplier,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"💾 {self.name} gespeichert: {filepath}")
    
    def load(self, filepath: str):
        """Lädt Statistical Model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model nicht gefunden: {filepath}")
        
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.mean = save_dict['mean']
        self.std = save_dict['std']
        self.q1 = save_dict['q1']
        self.q3 = save_dict['q3']
        self.iqr = save_dict['iqr']
        self.z_threshold = save_dict['z_threshold']
        self.iqr_multiplier = save_dict['iqr_multiplier']
        self.is_fitted = save_dict['is_fitted']
        
        print(f"📂 {self.name} geladen: {filepath}")

class EnsembleAnomalyDetector:
    """
    Ensemble-System für Anomalie-Erkennung mit vollständiger Model Persistence
    """
    
    def __init__(self):
        self.detectors = {}
        self.weights = {}
        self.is_fitted = False
        self.ensemble_threshold = 0.5
        self.ensemble_model_path = "models/ensemble_metadata.pkl"
        
    def add_detector(self, detector: BaseDetector, weight: float = 1.0):
        """Fügt einen Detector zum Ensemble hinzu"""
        self.detectors[detector.name] = detector
        self.weights[detector.name] = weight
        print(f"➕ Detector hinzugefügt: {detector.name} (Gewicht: {weight})")
    
    def save_ensemble_metadata(self):
        """Speichert Ensemble-Konfiguration"""
        os.makedirs('models', exist_ok=True)
        
        metadata = {
            'weights': self.weights,
            'ensemble_threshold': self.ensemble_threshold,
            'detector_names': list(self.detectors.keys()),
            'is_fitted': self.is_fitted
        }
        
        with open(self.ensemble_model_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"💾 Ensemble-Metadata gespeichert: {self.ensemble_model_path}")
    
    def load_ensemble_metadata(self):
        """Lädt Ensemble-Konfiguration"""
        if not os.path.exists(self.ensemble_model_path):
            return False
        
        try:
            with open(self.ensemble_model_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.weights = metadata['weights']
            self.ensemble_threshold = metadata['ensemble_threshold']
            self.is_fitted = metadata['is_fitted']
            
            print(f"📂 Ensemble-Metadata geladen: {self.ensemble_model_path}")
            return True
        except Exception as e:
            print(f"⚠️ Fehler beim Laden der Ensemble-Metadata: {e}")
            return False
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
            force_retrain: bool = False) -> 'EnsembleAnomalyDetector':
        """
        Trainiert alle Detektoren im Ensemble mit vollständiger Persistierung
        """
        print(f"🚀 Trainiere Ensemble mit {len(self.detectors)} Detektoren...")
        
        # Lade Ensemble-Konfiguration falls vorhanden
        if not force_retrain:
            self.load_ensemble_metadata()
        
        trained_count = 0
        loaded_count = 0
        
        for name, detector in self.detectors.items():
            # Bestimme Model-Pfad
            model_filename = f"ensemble_{name.lower().replace(' ', '_')}_detector"
            
            # Verschiedene Dateierweiterungen je nach Detector-Typ
            if isinstance(detector, LSTMDetector):
                model_path = f"models/{model_filename}.pth"  # PyTorch
            else:
                model_path = f"models/{model_filename}.pkl"  # Pickle
            
            # Versuche Model zu laden
            if not force_retrain and os.path.exists(model_path):
                try:
                    detector.load(model_path)
                    loaded_count += 1
                    continue
                except Exception as e:
                    print(f"⚠️ Laden fehlgeschlagen für {name}: {e}")
                    print(f"🔄 Starte Neutraining für {name}...")
            
            # Trainiere Detector
            try:
                detector.fit(X, y)
                trained_count += 1
                
                # Speichere trainiertes Model
                detector.save(model_path)
                
            except Exception as e:
                print(f"❌ Training fehlgeschlagen für {name}: {e}")
                # Entferne fehlerhaften Detector
                del self.detectors[name]
                del self.weights[name]
        
        # Speichere Ensemble-Konfiguration
        self.is_fitted = True
        self.save_ensemble_metadata()
        
        print(f"✅ Ensemble Training abgeschlossen!")
        print(f"   Detektoren geladen: {loaded_count}")
        print(f"   Detektoren trainiert: {trained_count}")
        print(f"   Aktive Detektoren: {len(self.detectors)}")
        
        return self
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Vorhersagen aller Detektoren + Ensemble-Entscheidung
        """
        if not self.is_fitted:
            raise ValueError("Ensemble muss erst trainiert werden!")
        
        predictions = {}
        scores = {}
        
        # Einzelne Detector-Vorhersagen
        for name, detector in self.detectors.items():
            try:
                predictions[name] = detector.predict(X)
                scores[name] = detector.decision_function(X)
            except Exception as e:
                print(f"⚠️ Vorhersage fehlgeschlagen für {name}: {e}")
                # Fallback: Alles als normal klassifizieren
                predictions[name] = np.zeros(len(X), dtype=int)
                scores[name] = np.zeros(len(X))
        
        # Ensemble-Entscheidung (Weighted Voting)
        ensemble_score = np.zeros(len(X))
        total_weight = sum(self.weights.values())
        
        if total_weight > 0:
            for name, weight in self.weights.items():
                if name in scores:
                    # Normalisiere Scores auf [0, 1]
                    normalized_score = self._normalize_scores(scores[name])
                    ensemble_score += (weight / total_weight) * normalized_score
        
        # Finale Ensemble-Vorhersage
        predictions['ensemble'] = (ensemble_score > self.ensemble_threshold).astype(int)
        predictions['ensemble_score'] = ensemble_score
        
        return predictions
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalisiert Scores auf [0, 1] Bereich"""
        if len(scores) == 0:
            return scores
            
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.zeros_like(scores)
        
        return (scores - min_score) / (max_score - min_score)
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict:
        """
        Evaluiert alle Detektoren auf Test-Daten
        """
        predictions = self.predict(X)
        
        results = {}
        
        # Evaluiere jeden Detector
        for name in self.detectors.keys():
            if name in predictions:
                y_pred = predictions[name]
                
                # Metriken berechnen
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                results[name] = {
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'recall': recall_score(y_true, y_pred, zero_division=0),
                    'f1': f1_score(y_true, y_pred, zero_division=0),
                    'accuracy': np.mean(y_true == y_pred)
                }
        
        # Ensemble-Ergebnisse
        if 'ensemble' in predictions:
            y_pred_ensemble = predictions['ensemble']
            results['ensemble'] = {
                'precision': precision_score(y_true, y_pred_ensemble, zero_division=0), # type: ignore
                'recall': recall_score(y_true, y_pred_ensemble, zero_division=0), # type: ignore
                'f1': f1_score(y_true, y_pred_ensemble, zero_division=0), # type: ignore
                'accuracy': np.mean(y_true == y_pred_ensemble)
            }
        
        return results
    
    def plot_comparison(self, X: np.ndarray, y_true: np.ndarray, 
                       sample_range: Tuple[int, int] = (0, 200)):
        """
        Visualisiert Vorhersagen aller Detektoren
        """
        start, end = sample_range
        X_sample = X[start:end]
        y_sample = y_true[start:end]
        
        predictions = self.predict(X_sample)
        
        fig, axes = plt.subplots(len(self.detectors) + 1, 1, 
                                figsize=(15, 3 * (len(self.detectors) + 1)))
        
        if len(self.detectors) <= 1:
            axes = [axes] if len(self.detectors) == 1 else []
        
        # Plot für jeden Detector
        for i, (name, detector) in enumerate(self.detectors.items()):
            if i < len(axes) - 1:  # Lasse Platz für Ensemble-Plot
                ax = axes[i]
                
                # Zeitreihe
                ax.plot(X_sample, 'b-', alpha=0.7, label='Temperature', linewidth=1)
                
                # Echte Anomalien
                anomaly_indices = np.where(y_sample == 1)[0]
                if len(anomaly_indices) > 0:
                    ax.scatter(anomaly_indices, X_sample[anomaly_indices], 
                              color='red', s=50, label='True Anomalies', zorder=5)
                
                # Detector-Vorhersagen
                if name in predictions:
                    predicted_indices = np.where(predictions[name] == 1)[0]
                    if len(predicted_indices) > 0:
                        ax.scatter(predicted_indices, X_sample[predicted_indices], 
                                  color='orange', s=30, marker='^', 
                                  label=f'{name} Predictions', zorder=4)
                
                ax.set_title(f'{name} Detector')
                ax.set_ylabel('Temperature (°C)')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Ensemble Plot
        if len(axes) > 0:
            ax = axes[-1]
            ax.plot(X_sample, 'b-', alpha=0.7, label='Temperature', linewidth=1)
            
            # Echte Anomalien
            anomaly_indices = np.where(y_sample == 1)[0]
            if len(anomaly_indices) > 0:
                ax.scatter(anomaly_indices, X_sample[anomaly_indices], 
                          color='red', s=50, label='True Anomalies', zorder=5)
            
            # Ensemble-Vorhersagen
            if 'ensemble' in predictions:
                ensemble_indices = np.where(predictions['ensemble'] == 1)[0]
                if len(ensemble_indices) > 0:
                    ax.scatter(ensemble_indices, X_sample[ensemble_indices], 
                              color='purple', s=40, marker='*', 
                              label='Ensemble Predictions', zorder=4)
            
            ax.set_title('Ensemble Detector')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Temperature (°C)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_model_info(self) -> Dict:
        """Gibt Informationen über geladene/trainierte Models zurück"""
        info = {
            'ensemble_fitted': self.is_fitted,
            'total_detectors': len(self.detectors),
            'detector_status': {},
            'weights': self.weights.copy(),
            'ensemble_threshold': self.ensemble_threshold
        }
        
        for name, detector in self.detectors.items():
            info['detector_status'][name] = {
                'fitted': detector.is_fitted,
                'type': type(detector).__name__
            }
        
        return info

def detailed_anomaly_analysis(ensemble, synthetic_df, sample_range: Tuple[int, int] = None): # type: ignore
    """
    Detaillierte Analyse: Erkannte vs. Eingebaute Anomalien in synthetischen Daten
    """
    print("🔍 === DETAILLIERTE ANOMALIE-ANALYSE ===")
    
    # Prüfe ob Labels vorhanden sind
    if 'is_anomaly' not in synthetic_df.columns:
        print("❌ Keine 'is_anomaly' Spalte in synthetischen Daten gefunden!")
        return None
    
    # Sample-Bereich definieren
    if sample_range is None:
        sample_range = (0, min(1000, len(synthetic_df)))
    
    start, end = sample_range
    df_sample = synthetic_df.iloc[start:end].copy().reset_index(drop=True)
    
    X_synthetic = df_sample['avg'].values
    y_true_synthetic = df_sample['is_anomaly'].astype(int).values
    
    print(f"Analysiere Bereich: {start}-{end} ({len(df_sample)} Datenpunkte)")
    print(f"Eingebaute Anomalien: {y_true_synthetic.sum()} ({y_true_synthetic.mean():.1%})")
    
    # Ensemble-Vorhersagen
    predictions = ensemble.predict(X_synthetic)
    y_pred_ensemble = predictions['ensemble']
    
    print(f"Erkannte Anomalien: {y_pred_ensemble.sum()} ({y_pred_ensemble.mean():.1%})")
    
    # Detaillierte Metriken
    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_synthetic, y_pred_ensemble)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"True Negatives (korrekt normal):  {tn}")
    print(f"False Positives (fälschlich anomal): {fp}")
    print(f"False Negatives (übersehene Anomalien): {fn}")
    print(f"True Positives (korrekt erkannte Anomalien): {tp}")
    
    # Detaillierte Metriken
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_synthetic, y_pred_ensemble, average='binary', zero_division=0)
    
    print(f"\n📈 DETAILLIERTE METRIKEN:")
    print(f"Precision (von erkannten, wie viele waren echt): {precision:.3f}")
    print(f"Recall (von echten, wie viele erkannt): {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    # Spezifische Anomalie-Analyse
    true_anomaly_indices = np.where(y_true_synthetic == 1)[0]
    detected_anomaly_indices = np.where(y_pred_ensemble == 1)[0]
    
    print(f"\n🎯 ANOMALIE-MATCHING:")
    print(f"Eingebaute Anomalie-Zeitpunkte: {true_anomaly_indices.tolist()}")
    print(f"Erkannte Anomalie-Zeitpunkte: {detected_anomaly_indices.tolist()}")
    
    # Overlap-Analyse
    correct_detections = np.intersect1d(true_anomaly_indices, detected_anomaly_indices)
    missed_anomalies = np.setdiff1d(true_anomaly_indices, detected_anomaly_indices)
    false_alarms = np.setdiff1d(detected_anomaly_indices, true_anomaly_indices)
    
    print(f"\n✅ Korrekt erkannte Anomalien: {len(correct_detections)} von {len(true_anomaly_indices)}")
    print(f"   Zeitpunkte: {correct_detections.tolist()}")
    
    print(f"\n❌ Übersehene Anomalien: {len(missed_anomalies)}")
    if len(missed_anomalies) > 0:
        print(f"   Zeitpunkte: {missed_anomalies.tolist()}")
        print(f"   Temperaturen: {X_synthetic[missed_anomalies]}")
        
        # Analysiere warum übersehen
        if 'anomaly_type' in df_sample.columns:
            missed_types = df_sample.iloc[missed_anomalies]['anomaly_type'].value_counts()
            print(f"   Übersehene Anomalie-Typen: {dict(missed_types)}")
    
    print(f"\n⚠️ False Alarms: {len(false_alarms)}")
    if len(false_alarms) > 0:
        print(f"   Zeitpunkte: {false_alarms.tolist()}")
        print(f"   Temperaturen: {X_synthetic[false_alarms]}")
    
    # Einzelne Detector-Analyse
    print(f"\n🔍 DETECTOR-SPEZIFISCHE ANALYSE:")
    for detector_name in ['Isolation_Forest', 'Statistical']:  # LSTM entfernt
        if detector_name in predictions:
            detector_pred = predictions[detector_name]
            detector_recall = recall_score(y_true_synthetic, detector_pred, zero_division=0)
            detector_precision = precision_score(y_true_synthetic, detector_pred, zero_division=0)
            
            print(f"{detector_name}:")
            print(f"  Precision: {detector_precision:.3f}, Recall: {detector_recall:.3f}")
            print(f"  Erkannte: {detector_pred.sum()}, Davon korrekt: {np.sum((detector_pred == 1) & (y_true_synthetic == 1))}")
    
    # Anomalie-Typ Analyse (falls verfügbar)
    if 'anomaly_type' in df_sample.columns:
        print(f"\n🏷️ ANOMALIE-TYP ANALYSE:")
        anomaly_types = df_sample[df_sample['is_anomaly'] == 1]['anomaly_type'].value_counts()
        
        for anomaly_type, count in anomaly_types.items():
            type_mask = (df_sample['anomaly_type'] == anomaly_type) & (df_sample['is_anomaly'] == 1)
            type_indices = np.where(type_mask)[0]
            detected_of_type = np.sum(y_pred_ensemble[type_indices])
            
            print(f"{anomaly_type}: {detected_of_type}/{count} erkannt ({detected_of_type/count:.1%})")
    
    return {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'correct_detections': correct_detections,
        'missed_anomalies': missed_anomalies,
        'false_alarms': false_alarms
    }

def plot_anomaly_detection_comparison(ensemble, synthetic_df, sample_range: Tuple[int, int] = (0, 200)):
    """
    Visualisiert Eingebaute vs. Erkannte Anomalien
    """
    start, end = sample_range
    df_sample = synthetic_df.iloc[start:end].copy().reset_index(drop=True)
    
    X_sample = df_sample['avg'].values
    y_true = df_sample['is_anomaly'].astype(int).values if 'is_anomaly' in df_sample.columns else None
    
    if y_true is None:
        print("❌ Keine Labels für Visualisierung verfügbar!")
        return
    
    # Ensemble-Vorhersagen
    predictions = ensemble.predict(X_sample)
    y_pred = predictions['ensemble']
    
    # Erstelle Plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Zeitreihe mit Anomalien
    ax1 = axes[0]
    
    # Temperatur-Zeitreihe
    ax1.plot(X_sample, 'b-', alpha=0.7, label='Temperature', linewidth=1)
    
    # Eingebaute Anomalien (rot)
    true_anomaly_indices = np.where(y_true == 1)[0]
    if len(true_anomaly_indices) > 0:
        ax1.scatter(true_anomaly_indices, X_sample[true_anomaly_indices], 
                   color='red', s=80, label='Eingebaute Anomalien', 
                   marker='o', zorder=5, alpha=0.8)
    
    # Erkannte Anomalien (orange)
    detected_indices = np.where(y_pred == 1)[0]
    if len(detected_indices) > 0:
        ax1.scatter(detected_indices, X_sample[detected_indices], 
                   color='orange', s=60, label='Erkannte Anomalien', 
                   marker='^', zorder=4, alpha=0.8)
    
    # Korrekt erkannte (grün)
    correct_indices = np.intersect1d(true_anomaly_indices, detected_indices)
    if len(correct_indices) > 0:
        ax1.scatter(correct_indices, X_sample[correct_indices], 
                   color='green', s=100, label='Korrekt erkannt', 
                   marker='*', zorder=6, alpha=1.0)
    
    ax1.set_title('Anomalie-Detection Vergleich: Eingebaut vs. Erkannt')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Detection-Status über Zeit
    ax2 = axes[1]
    
    # Status-Encoding
    status = np.zeros(len(X_sample))  # 0 = Normal
    status[true_anomaly_indices] = 1  # 1 = Eingebaute Anomalie
    status[detected_indices] = 2  # 2 = Erkannte Anomalie (überschreibt)
    status[correct_indices] = 3  # 3 = Korrekt erkannt (überschreibt)
    
    # False Positives
    false_positives = np.setdiff1d(detected_indices, true_anomaly_indices)
    status[false_positives] = 4  # 4 = False Positive
    
    # Color-coding
    colors = ['lightblue', 'red', 'orange', 'green', 'purple']
    labels = ['Normal', 'Eingebaut (übersehen)', 'Erkannt (falsch)', 'Korrekt erkannt', 'False Positive']
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        mask = (status == i)
        if np.any(mask):
            ax2.scatter(np.where(mask)[0], status[mask], 
                       color=color, label=label, s=30, alpha=0.7)
    
    ax2.set_title('Detection-Status über Zeit')
    ax2.set_xlabel('Zeit (Stunden)')
    ax2.set_ylabel('Status')
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(['Normal', 'Übersehen', 'Falsch erkannt', 'Korrekt', 'False Alarm'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistiken ausgeben
    total_injected = len(true_anomaly_indices)
    total_detected = len(detected_indices)
    correct_detections = len(correct_indices)
    false_positives_count = len(false_positives)
    false_negatives_count = total_injected - correct_detections
    
    print(f"\n📊 VISUALISIERUNGS-STATISTIKEN:")
    print(f"Eingebaute Anomalien: {total_injected}")
    print(f"Erkannte Anomalien: {total_detected}")
    print(f"Korrekt erkannt: {correct_detections}/{total_injected} ({correct_detections/total_injected:.1%} Recall)")
    print(f"False Positives: {false_positives_count}")
    print(f"False Negatives: {false_negatives_count}")

def prefect_anomaly_validation(ensemble, synthetic_df, validation_threshold: float = 0.7):
    """
    Validierung für Prefect Pipeline: Prüft ob Anomalie-Detection funktioniert
    
    Args:
        ensemble: Trainiertes Ensemble
        synthetic_df: Synthetische Daten mit Labels
        validation_threshold: Minimum Recall für Success
    
    Returns:
        bool: True wenn Validation erfolgreich
    """
    print("🚀 === PREFECT ANOMALIE-VALIDIERUNG ===")
    
    # Quick validation auf ersten 500 Punkten
    sample_size = min(500, len(synthetic_df))
    df_sample = synthetic_df.head(sample_size)
    
    if 'is_anomaly' not in df_sample.columns:
        print("❌ Keine Labels für Validierung!")
        return False
    
    X = df_sample['avg'].values
    y_true = df_sample['is_anomaly'].astype(int).values
    
    # Vorhersagen
    predictions = ensemble.predict(X)
    y_pred = predictions['ensemble']
    
    # Metriken
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    
    injected_count = y_true.sum()
    detected_count = y_pred.sum()
    correct_count = np.sum((y_true == 1) & (y_pred == 1))
    
    print(f"Injizierte Anomalien: {injected_count}")
    print(f"Erkannte Anomalien: {detected_count}")
    print(f"Korrekt erkannt: {correct_count}")
    print(f"Recall: {recall:.3f}")
    print(f"Precision: {precision:.3f}")
    
    # Validierung
    validation_passed = recall >= validation_threshold
    
    if validation_passed:
        print(f"✅ VALIDATION ERFOLGREICH! Recall {recall:.3f} >= {validation_threshold}")
    else:
        print(f"❌ VALIDATION FEHLGESCHLAGEN! Recall {recall:.3f} < {validation_threshold}")
        
        # Debug-Info bei Failure
        if injected_count > 0:
            missed_indices = np.where((y_true == 1) & (y_pred == 0))[0]
            print(f"🔍 Übersehene Anomalien an Positionen: {missed_indices.tolist()}")
            print(f"🔍 Temperaturen der übersehenen: {X[missed_indices]}")
    
    return validation_passed

def main_phase1(retrain: bool = False, detailed_analysis: bool = True):
    """
    Hauptfunktion: Lädt Daten, trainiert Ensemble, führt detaillierte Analyse durch
    """
    print("🚀 === ENSEMBLE ANOMALY DETECTOR - PHASE 1 ===")
    
    # Lade Trainingsdaten
    try:
        df = pd.read_csv('data/labeled_training_data_cleaned.csv')
        print(f"✅ Trainingsdaten geladen: {len(df)} Datenpunkte")
    except FileNotFoundError:
        print("❌ Trainingsdaten nicht gefunden!")
        return None
    
    # Prepare data
    X = df['avg'].values
    y = df['anomaly_label'].values if 'anomaly_label' in df.columns else None
    
    # Train/Test split für Evaluation
    split_idx = int(0.8 * len(X))
    X_train = X[:split_idx]
    y_train = y[:split_idx] if y is not None else None
    
    # Initialize und trainiere Ensemble (ohne LSTM - schlechte Performance)
    ensemble = EnsembleAnomalyDetector()
    # ensemble.add_detector(LSTMDetector(sequence_length=24, hidden_dim=32), weight=1.5)  # Entfernt: hoher Recall aber sehr niedrige Precision
    ensemble.add_detector(IsolationForestDetector(contamination=0.1), weight=1.0)
    ensemble.add_detector(StatisticalDetector(z_threshold=2.5), weight=1.0)
    
    ensemble.fit(X_train, y_train, force_retrain=retrain) # type: ignore
    
    # Lade synthetische Daten
    try:
        df_synthetic = pd.read_csv('data/synthetic/arimax_synthetic_heating_data.csv')
        print("✅ Synthetische Daten geladen")
        
        if detailed_analysis:
            print("\n" + "="*50)
            
            # Detaillierte Anomalie-Analyse
            analysis_results = detailed_anomaly_analysis(ensemble, df_synthetic, sample_range=(0, 500))
            
            # Visualisierung
            plot_anomaly_detection_comparison(ensemble, df_synthetic, sample_range=(0, 200))
            
            # Prefect-Validierung
            validation_passed = prefect_anomaly_validation(ensemble, df_synthetic, validation_threshold=0.7)
            
            return ensemble, analysis_results, validation_passed
        
    except FileNotFoundError:
        print("⚠️ Keine synthetischen Daten gefunden. Führe zuerst arimax_generator.py aus.")
        return ensemble, None, False
    
    return ensemble

if __name__ == "__main__":
    # Ausführung mit detaillierter Analyse
    result = main_phase1(retrain=True, detailed_analysis=True)
    
    if isinstance(result, tuple):
        ensemble, analysis_results, validation_passed = result
        
        print(f"\n✅ Ensemble Anomaly Detector abgeschlossen!")
        
        if analysis_results:
            print(f"🎯 Anomalie-Detection Qualität:")
            print(f"   Recall: {analysis_results['recall']:.3f}")
            print(f"   Precision: {analysis_results['precision']:.3f}")
            print(f"   F1-Score: {analysis_results['f1']:.3f}")
        
        if validation_passed:
            print("✅ Prefect-Validierung: ERFOLGREICH")
        else:
            print("❌ Prefect-Validierung: FEHLGESCHLAGEN")