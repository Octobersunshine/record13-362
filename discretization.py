import pandas as pd
import numpy as np
from typing import List, Union, Optional, Tuple
from sklearn.cluster import KMeans


class DataDiscretizer:
    def __init__(self):
        self.bins_ = {}
        self.labels_ = {}
        self.methods_ = {}
        self.right_ = {}
        self.include_lowest_ = {}

    def fit(
        self,
        data: pd.DataFrame,
        columns: List[str],
        method: str = 'equal_width',
        n_bins: int = 3,
        custom_bins: Optional[dict] = None,
        labels: Optional[dict] = None,
        include_lowest: bool = True,
        right: bool = True
    ) -> 'DataDiscretizer':
        for col in columns:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            
            col_labels = labels.get(col) if labels else None
            col_custom_bins = custom_bins.get(col) if custom_bins else None
            
            if method == 'custom' and col_custom_bins is None:
                raise ValueError(f"Custom bins must be provided for column '{col}' when method='custom'")
            
            if method == 'equal_width':
                bins = self._equal_width_bins(data[col], n_bins)
            elif method == 'equal_freq':
                bins = self._equal_freq_bins(data[col], n_bins)
            elif method == 'kmeans':
                bins = self._kmeans_bins(data[col], n_bins)
            elif method == 'custom':
                bins = col_custom_bins
            else:
                raise ValueError(f"Unknown method: {method}. Use 'equal_width', 'equal_freq', 'kmeans', or 'custom'")
            
            if col_labels is None:
                col_labels = self._generate_default_labels(len(bins) - 1)
            
            if len(col_labels) != len(bins) - 1:
                raise ValueError(
                    f"Number of labels ({len(col_labels)}) must be equal to number of bins minus one ({len(bins) - 1})"
                )
            
            self.bins_[col] = bins
            self.labels_[col] = col_labels
            self.methods_[col] = method
            self.right_[col] = right
            self.include_lowest_[col] = include_lowest
        
        return self

    def transform(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        inplace: bool = False,
        suffix: str = '_discretized',
        handle_unknown: str = 'clip'
    ) -> pd.DataFrame:
        if not self.bins_:
            raise ValueError("Discretizer has not been fitted. Call fit() first.")

        if handle_unknown not in ['clip', 'error', 'nan']:
            raise ValueError("handle_unknown must be 'clip', 'error', or 'nan'")

        if columns is None:
            columns = list(self.bins_.keys())

        if not inplace:
            data = data.copy()

        for col in columns:
            if col not in self.bins_:
                raise ValueError(f"Column '{col}' was not fitted. Call fit() with this column first.")

            bins = self.bins_[col].copy()
            labels = self.labels_[col]
            right = self.right_[col]
            include_lowest = self.include_lowest_[col]
            col_data = data[col]

            if not right and include_lowest:
                bins[-1] = np.nextafter(bins[-1], bins[-1] + 1)

            if handle_unknown == 'clip':
                col_data = col_data.clip(self.bins_[col][0], self.bins_[col][-1])
            elif handle_unknown == 'error':
                lo = self.bins_[col][0]
                hi = self.bins_[col][-1]
                if right:
                    out_of_range = (col_data < lo) | (col_data > hi)
                else:
                    out_of_range = (col_data < lo) | (col_data > hi)
                if out_of_range.any():
                    raise ValueError(
                        f"Column '{col}' contains values outside fitted bin range "
                        f"[{lo}, {hi}]: {col_data[out_of_range].tolist()}"
                    )

            data[f"{col}{suffix}"] = pd.cut(
                col_data,
                bins=bins,
                labels=labels,
                include_lowest=include_lowest,
                right=right
            )

        return data

    def fit_transform(
        self,
        data: pd.DataFrame,
        columns: List[str],
        method: str = 'equal_width',
        n_bins: int = 3,
        custom_bins: Optional[dict] = None,
        labels: Optional[dict] = None,
        include_lowest: bool = True,
        right: bool = True,
        inplace: bool = False,
        suffix: str = '_discretized',
        handle_unknown: str = 'clip'
    ) -> pd.DataFrame:
        self.fit(data, columns, method, n_bins, custom_bins, labels, include_lowest, right)
        return self.transform(data, columns, inplace, suffix, handle_unknown)

    def get_bin_info(self, column: Optional[str] = None) -> Union[dict, pd.DataFrame]:
        if column is not None:
            if column not in self.bins_:
                raise ValueError(f"Column '{column}' was not fitted.")
            return self._get_column_bin_info(column)
        
        info = {}
        for col in self.bins_:
            info[col] = self._get_column_bin_info(col)
        return info

    def _get_column_bin_info(self, column: str) -> pd.DataFrame:
        bins = self.bins_[column]
        labels = self.labels_[column]
        method = self.methods_[column]
        right = self.right_[column]
        include_lowest = self.include_lowest_[column]

        intervals = []
        for i in range(len(bins) - 1):
            left = bins[i]
            right_val = bins[i + 1]
            if right:
                left_bracket = '[' if (i == 0 and include_lowest) else '('
                right_bracket = ']'
            else:
                left_bracket = '['
                right_bracket = ']' if (i == len(bins) - 2 and include_lowest) else ')'
            intervals.append(f"{left_bracket}{left}, {right_val}{right_bracket}")

        return pd.DataFrame({
            'label': labels,
            'interval': intervals,
            'bin_start': bins[:-1],
            'bin_end': bins[1:],
            'method': method,
            'right': right,
            'include_lowest': include_lowest
        })

    @staticmethod
    def _equal_width_bins(series: pd.Series, n_bins: int) -> np.ndarray:
        min_val = series.min()
        max_val = series.max()
        return np.linspace(min_val, max_val, n_bins + 1)

    @staticmethod
    def _equal_freq_bins(series: pd.Series, n_bins: int) -> np.ndarray:
        return series.quantile(np.linspace(0, 1, n_bins + 1)).values

    @staticmethod
    def _kmeans_bins(series: pd.Series, n_bins: int) -> np.ndarray:
        values = series.values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
        kmeans.fit(values)
        
        centers = np.sort(kmeans.cluster_centers_.flatten())
        edges = np.concatenate([
            [series.min()],
            (centers[:-1] + centers[1:]) / 2,
            [series.max()]
        ])
        return edges

    @staticmethod
    def _generate_default_labels(n: int) -> List[str]:
        if n <= 3:
            return ['低', '中', '高'][:n] if n <= 3 else ['低', '中低', '中', '中高', '高'][:n]
        elif n <= 5:
            return ['很低', '低', '中', '高', '很高'][:n]
        else:
            return [f'等级{i+1}' for i in range(n)]
