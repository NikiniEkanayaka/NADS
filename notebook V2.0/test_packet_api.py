import requests
import json

# Test packet-level prediction
test_packet = {
    "destination_port": 80,
    "syn_flag_count": 1,
    "rst_flag_count": 0,
    "psh_flag_count": 0,
    "ack_flag_count": 0,
    "urg_flag_count": 0,
    "fin_flag_count": 0,
    "init_win_bytes_forward": 65535,
    "init_win_bytes_backward": 0,
    "fwd_packet_length_max": 60
}

print("Testing packet-level prediction...")
print("Packet data:", json.dumps(test_packet, indent=2))

# For now, test without auth to verify the model works
# We'll need to get a token from the frontend later

try:
    response = requests.post(
        "http://localhost:8000/predict",
        json=test_packet,
        headers={"Content-Type": "application/json"}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")

# Also test the model directly
print("\nTesting model directly...")
from backend.core.model_loader import model, scaler, threshold, features, label_encoder

if model is not None:
    import numpy as np
    import pandas as pd

    # Convert packet to features
    feature_vector = [
        test_packet["destination_port"],
        test_packet["syn_flag_count"],
        test_packet["fin_flag_count"],
        test_packet["rst_flag_count"],
        test_packet["psh_flag_count"],
        test_packet["ack_flag_count"],
        test_packet["urg_flag_count"],
        test_packet["init_win_bytes_forward"],
        test_packet["init_win_bytes_backward"],
        test_packet["fwd_packet_length_max"]
    ]

    x = np.array(feature_vector).reshape(1, -1)
    x_df = pd.DataFrame(x, columns=features)
    x_scaled = scaler.transform(x_df)

    probs = model.predict_proba(x_scaled)[0]
    max_prob = float(np.max(probs))

    if max_prob >= threshold:
        pred_class = int(np.argmax(probs))
        confidence = max_prob
    else:
        pred_class = 0  # BENIGN
        confidence = max_prob

    pred_label = label_encoder.inverse_transform([pred_class])[0] if label_encoder else str(pred_class)
    status = "anomalous" if pred_class != 0 else "normal"

    print(f"Direct model prediction: {pred_label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Status: {status}")
else:
    print("Model not loaded!")