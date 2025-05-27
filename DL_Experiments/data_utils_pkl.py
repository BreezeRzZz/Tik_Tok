import pickle
import numpy as np
from sklearn.model_selection import train_test_split

def load_pkl_data(mon_pkl_path, length=5000, typ=0, test_size=0.2, val_size=0.1):
    """
    Load data from pkl files for closed-world experiments
    
    Args:
        mon_pkl_path: Path to ts_mon.pkl file
        length: Sequence length (default 5000)
        typ: Data representation type (0=direction, 1=directional_timing, 2=timing)
        test_size: Test set ratio
        val_size: Validation set ratio
    
    Returns:
        X_train, y_train, X_valid, y_valid, X_test, y_test
    """
    
    # Load monitored data
    with open(mon_pkl_path, 'rb') as f:
        mon_data = pickle.load(f)
    
    X, y = [], []
    
    for class_id, traces in mon_data.items():
        for trace in traces:
            # Process each trace
            sequence = process_trace(trace, typ=typ, length=length)
            if sequence is not None:
                X.append(sequence)
                y.append(class_id)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
    )
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def process_trace(trace, typ=0, length=5000):
    """
    Process a single trace based on representation type
    
    Args:
        trace: List of timestamp*direction values
        typ: 0=direction, 1=directional_timing, 2=timing
        length: Target sequence length
    
    Returns:
        Processed sequence as numpy array
    """
    
    timestamps = []
    directions = []
    
    for value in trace:
        if value == 0:
            # First packet, treat as outgoing
            timestamps.append(0.0)
            directions.append(1)
        else:
            timestamp = abs(value)
            direction = 1 if value > 0 else -1
            timestamps.append(timestamp)
            directions.append(direction)
    
    # Choose representation
    if typ == 0:  # Direction only
        sequence = directions
    elif typ == 1:  # Directional timing (timestamp * direction)
        sequence = [timestamps[i] * directions[i] for i in range(len(timestamps))]
    elif typ == 2:  # Timing only
        sequence = timestamps
    else:
        raise ValueError("Invalid typ value. Use 0, 1, or 2.")
    
    # Pad or truncate to fixed length
    sequence = np.array(sequence)
    if len(sequence) < length:
        sequence = np.hstack((sequence, np.zeros(length - len(sequence))))
    else:
        sequence = sequence[:length]
    
    # Reshape for CNN input
    sequence = sequence.reshape((length, 1))
    
    return sequence