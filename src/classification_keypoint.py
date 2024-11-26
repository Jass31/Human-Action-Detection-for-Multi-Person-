import numpy as np
import torch
import torch.nn as nn

class AngleFeatureExtractor:
    def __init__(self):
        self.angle_pairs = [
            # Define keypoint pairs for angle calculation
            ((5, 7), (7, 9)),   # Right arm
            ((6, 8), (8, 10)),  # Left arm
            ((11, 13), (13, 15)), # Right leg
            ((12, 14), (14, 16)), # Left leg
            ((5, 6), (11, 12)),   # Shoulders to hips
            ((5, 11), (6, 12)),   # Right to left side
            ((7, 5), (5, 6)),     # Upper body right
            ((8, 6), (6, 5))      # Upper body left
        ]

    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points with safety checks."""
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # Check for zero vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-10 or v2_norm < 1e-10:
            return 0.0
        
        # Normalize vectors
        v1_normalized = v1 / v1_norm
        v2_normalized = v2 / v2_norm
        
        # Calculate dot product and clip to prevent numerical errors
        dot_product = np.clip(np.dot(v1_normalized, v2_normalized), -1.0, 1.0)
        
        angle = np.arccos(dot_product)
        return np.degrees(angle)

    def calculate_angles(self, keypoints):
        """Calculate all angles from keypoints with error handling."""
        # Reshape keypoints into (x,y) coordinates
        points = keypoints.reshape(-1, 2)
        angles = []
        
        for (p1_idx, p2_idx), (p3_idx, p4_idx) in self.angle_pairs:
            try:
                # Get points for first vector
                p1 = points[p1_idx]
                p2 = points[p2_idx]
                # Get points for second vector
                p3 = points[p3_idx]
                p4 = points[p4_idx]
                
                # Calculate angle between vectors
                angle1 = self.calculate_angle(p1, p2, p4)
                angles.append(angle1)
            except (IndexError, ValueError) as e:
                print(f"Error calculating angle: {e}")
                angles.append(0.0)
        
        return np.array(angles)

class AngleLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, lstm_dropout=0.3, fc_dropout=0.5):
        super(AngleLSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def preprocess_data(train_data, test_data, sequence_length=10):
    """Preprocess the data with proper sequence handling."""
    angle_extractor = AngleFeatureExtractor()
    
    def extract_features(data):
        keypoints = data.iloc[:, 2:].values
        angles_list = []
        for keypoint_row in keypoints:
            angles = angle_extractor.calculate_angles(keypoint_row)
            angles_list.append(angles)
        return np.array(angles_list)
    
    X_train = extract_features(train_data)
    X_test = extract_features(test_data)
    y_train = train_data['label'].values
    y_test = test_data['label'].values
    
    # Reshape data into sequences
    def create_sequences(X, seq_length):
        sequences = []
        num_sequences = len(X) - seq_length + 1
        for i in range(num_sequences):
            sequence = X[i:i + seq_length]
            sequences.append(sequence)
        return np.array(sequences)
    
    X_train_seq = create_sequences(X_train, sequence_length)
    X_test_seq = create_sequences(X_test, sequence_length)
    y_train_seq = y_train[sequence_length-1:]
    y_test_seq = y_test[sequence_length-1:]
    
    return X_train_seq, X_test_seq, y_train_seq, y_test_seq
