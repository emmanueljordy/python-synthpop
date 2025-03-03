# privacy_metrics.py

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

class DisclosureProtection:
    """
    A class to compute the disclosure protection metric for synthetic data.

    The metric is defined as 1 minus the proportion of synthetic records that are too similar
    (i.e. within a risk threshold) to a record in the real dataset.

    Parameters
    ----------
    real_data : pd.DataFrame
        A DataFrame containing the real data. The data should be numeric or preprocessed.
    synthetic_data : pd.DataFrame
        A DataFrame containing the synthetic data (with the same columns as real_data).
    threshold : float, optional
        A distance threshold under which a synthetic record is considered a potential disclosure risk.
        If not provided, it is computed as the 10th percentile of the nearest-neighbor distances among real records.
    """
    
    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, threshold: float = None):
        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()
        self.threshold = threshold
        self._compute_threshold()

    def _compute_threshold(self):
        """
        Compute the threshold if not provided. Uses the 10th percentile of the nearest-neighbor
        distances among real records (excluding self-distance).
        """
        if self.threshold is None:
            # Fit a nearest neighbor model on the real data.
            # n_neighbors=2 because the closest neighbor of a record is itself.
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(self.real_data)
            distances, _ = nn.kneighbors(self.real_data)
            # distances[:, 1] are the distances to the closest distinct record.
            self.threshold = np.percentile(distances[:, 1], 10)
    
    def score(self) -> float:
        """
        Compute the disclosure protection score.
        
        For each synthetic record, compute its distance to the nearest real record.
        The risk rate is the proportion of synthetic records with distance below the threshold.
        The disclosure protection score is 1 - risk_rate (higher is better).

        Returns
        -------
        float
            Disclosure protection score between 0 and 1.
        """
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(self.real_data)
        distances, _ = nn.kneighbors(self.synthetic_data)
        distances = distances.flatten()
        risk_count = np.sum(distances < self.threshold)
        risk_rate = risk_count / len(distances)
        return 1 - risk_rate

    def report(self) -> dict:
        """
        Generate a detailed report of the disclosure protection metric.

        Returns
        -------
        dict
            A dictionary containing the threshold, risk rate, and the final disclosure protection score.
        """
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(self.real_data)
        distances, _ = nn.kneighbors(self.synthetic_data)
        distances = distances.flatten()
        risk_count = np.sum(distances < self.threshold)
        risk_rate = risk_count / len(distances)
        score = 1 - risk_rate
        return {
            "threshold": self.threshold,
            "risk_rate": risk_rate,
            "disclosure_protection_score": score
        }
