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
        self.interpolation_ = {}
        self.n_bins_ = {}

    def fit(
        self,
        data: pd.DataFrame,
        columns: List[str],
        method: str = 'equal_width',
        n_bins: int = 3,
        custom_bins: Optional[dict] = None,
        labels: Optional[dict] = None,
        include_lowest: bool = True,
        right: bool = True,
        interpolation: str = 'linear'
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
                bins = self._equal_freq_bins(data[col], n_bins, interpolation)
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
            self.interpolation_[col] = interpolation
            self.n_bins_[col] = n_bins
        
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
        interpolation: str = 'linear',
        inplace: bool = False,
        suffix: str = '_discretized',
        handle_unknown: str = 'clip'
    ) -> pd.DataFrame:
        self.fit(
            data, columns, method, n_bins, custom_bins, labels,
            include_lowest, right, interpolation
        )
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

    def get_bin_counts(
        self,
        data: pd.DataFrame,
        column: Optional[str] = None,
        suffix: str = '_discretized',
        normalize: bool = False
    ) -> Union[pd.Series, dict]:
        if not self.bins_:
            raise ValueError("Discretizer has not been fitted. Call fit() first.")

        if column is not None:
            if column not in self.bins_:
                raise ValueError(f"Column '{column}' was not fitted.")
            col_name = f"{column}{suffix}"
            if col_name not in data.columns:
                transformed = self.transform(data[[column]], columns=[column], suffix=suffix)
                series = transformed[col_name]
            else:
                series = data[col_name]
            counts = series.value_counts().sort_index()
            if normalize:
                counts = counts / counts.sum()
            return counts

        result = {}
        for col in self.bins_:
            col_name = f"{col}{suffix}"
            if col_name not in data.columns:
                transformed = self.transform(data[[col]], columns=[col], suffix=suffix)
                series = transformed[col_name]
            else:
                series = data[col_name]
            counts = series.value_counts().sort_index()
            if normalize:
                counts = counts / counts.sum()
            result[col] = counts
        return result

    def describe_equal_freq(
        self,
        data: pd.DataFrame,
        column: str
    ) -> pd.DataFrame:
        if column not in self.bins_:
            raise ValueError(f"Column '{column}' was not fitted.")
        if self.methods_[column] != 'equal_freq':
            raise ValueError(
                f"Column '{column}' uses method '{self.methods_[column]}', "
                f"this method is only valid for 'equal_freq'."
            )

        bin_info = self._get_column_bin_info(column)
        counts = self.get_bin_counts(data, column)
        proportions = self.get_bin_counts(data, column, normalize=True)

        bin_info['count'] = counts.values
        bin_info['proportion'] = (proportions.values * 100).round(2).astype(str) + '%'

        n_total = len(data)
        n_bins = self.n_bins_[column]
        expected = n_total / n_bins
        bin_info['expected_count'] = round(expected, 2)
        bin_info['diff'] = bin_info['count'] - expected
        bin_info['diff_pct'] = ((bin_info['diff'] / expected) * 100).round(2).astype(str) + '%'

        return bin_info

    def _get_column_bin_info(self, column: str) -> pd.DataFrame:
        bins = self.bins_[column]
        labels = self.labels_[column]
        method = self.methods_[column]
        right = self.right_[column]
        include_lowest = self.include_lowest_[column]
        interpolation = self.interpolation_.get(column, 'linear')
        n_bins = self.n_bins_.get(column, len(bins) - 1)

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
            'n_bins': n_bins,
            'right': right,
            'include_lowest': include_lowest,
            'interpolation': interpolation
        })

    @staticmethod
    def _equal_width_bins(series: pd.Series, n_bins: int) -> np.ndarray:
        min_val = series.min()
        max_val = series.max()
        return np.linspace(min_val, max_val, n_bins + 1)

    @staticmethod
    def _equal_freq_bins(
        series: pd.Series,
        n_bins: int,
        interpolation: str = 'linear'
    ) -> np.ndarray:
        valid_methods = ['linear', 'lower', 'higher', 'midpoint', 'nearest']
        if interpolation not in valid_methods:
            raise ValueError(
                f"Invalid interpolation '{interpolation}'. Must be one of {valid_methods}"
            )

        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = series.quantile(quantiles, interpolation=interpolation).values

        unique_bins, counts = np.unique(bins, return_counts=True)
        if len(unique_bins) < len(bins):
            sorted_unique = np.sort(series.dropna().unique())
            if len(sorted_unique) <= n_bins:
                bins = np.concatenate([[sorted_unique[0]], sorted_unique, [sorted_unique[-1]]])
                bins = np.unique(bins)
                if len(bins) > n_bins + 1:
                    step = (len(bins) - 1) / n_bins
                    idx = [int(round(i * step)) for i in range(n_bins + 1)]
                    bins = bins[idx]
            else:
                bins = unique_bins
                if len(bins) > n_bins + 1:
                    idx = np.linspace(0, len(bins) - 1, n_bins + 1).astype(int)
                    bins = bins[idx]
                elif len(bins) < 2:
                    min_val = float(series.min())
                    max_val = float(series.max())
                    if min_val == max_val:
                        bins = np.array([min_val - 1e-9, min_val + 1e-9])
                    else:
                        bins = np.linspace(min_val, max_val, n_bins + 1)
                else:
                    min_val = float(series.min())
                    max_val = float(series.max())
                    if bins[0] > min_val:
                        bins = np.concatenate([[min_val], bins])
                    if bins[-1] < max_val:
                        bins = np.concatenate([bins, [max_val]])
                    bins = np.unique(bins)
                    if len(bins) < n_bins + 1:
                        extra = n_bins + 1 - len(bins)
                        for i in range(extra):
                            insert_pos = len(bins) - 1 - i
                            if insert_pos > 0:
                                mid = (bins[insert_pos - 1] + bins[insert_pos]) / 2
                                bins = np.insert(bins, insert_pos, mid)

        if bins[0] > series.min():
            bins[0] = series.min()
        if bins[-1] < series.max():
            bins[-1] = series.max()

        bins = np.unique(bins)
        if len(bins) < 2:
            min_val = float(series.min())
            max_val = float(series.max())
            if min_val == max_val:
                bins = np.array([min_val - 1e-9, min_val + 1e-9])
            else:
                bins = np.linspace(min_val, max_val, n_bins + 1)

        return bins

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
