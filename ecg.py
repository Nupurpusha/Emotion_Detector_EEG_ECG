import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_ecg(file_path, n_components=None):
    df = pd.read_csv(file_path).dropna()
    features = df.iloc[:, :-1]
    labels = df.iloc[:, -1]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    if n_components:
        max_components = min(int(n_components), features_scaled.shape[1], features_scaled.shape[0])
        if max_components > 0:
            pca = PCA(n_components=max_components, random_state=42)
            features_scaled = pca.fit_transform(features_scaled)

    return features_scaled, labels

