from .tensor import Tensor
import csv
import math


class Dataset:

    def __init__(
            self,
            path: str,
            target_column: str = None,
            delimiter: str = ",",
            ) -> None:

        self.data_path = path

        raw_data: list[dict[str, str]] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            self.fieldnames = list(reader.fieldnames) if reader.fieldnames else []
            for row in reader:
                raw_data.append(row)

        if not raw_data:
            raise ValueError(f"No data rows found in {path!r}")

        # Separate feature columns from target column
        feature_names = [c for c in self.fieldnames if c != target_column]
        target_name = target_column

        # Detect column types
        self.column_types: dict[str, str] = {}
        for col in feature_names:
            self.column_types[col] = self._detect_type(raw_data, col)

        if target_name:
            self.column_types[target_name] = self._detect_type(raw_data, target_name)

        # Build encoding maps for categorical columns
        self.encodings: dict[str, dict[str, int]] = {}
        for col in self.fieldnames:
            if self.column_types.get(col) == "categorical":
                unique = sorted({row[col] for row in raw_data if row[col]})
                self.encodings[col] = {v: i for i, v in enumerate(unique)}

        # Store normalization parameters
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}

        # Determine which columns are one-hot expanded from categoricals
        self.processed_feature_names: list[str] = []
        for col in feature_names:
            if self.column_types.get(col) == "categorical":
                for cat in sorted(self.encodings[col].keys()):
                    self.processed_feature_names.append(f"{col}_{cat}")
            else:
                self.processed_feature_names.append(col)

        self.target_name = target_name
        self.n_samples = len(raw_data)

        # Build numeric feature matrix (rows x cols) before normalization
        n_cols = len(self.processed_feature_names)
        X_rows: list[list[float]] = []
        y_rows: list[list[float]] = []

        for row in raw_data:
            x_row: list[float] = []
            for col in feature_names:
                val = row.get(col, "").strip()
                if self.column_types.get(col) == "categorical":
                    one_hot = [0.0] * len(self.encodings[col])
                    if val in self.encodings[col]:
                        one_hot[self.encodings[col][val]] = 1.0
                    x_row.extend(one_hot)
                else:
                    try:
                        x_row.append(float(val) if val else 0.0)
                    except ValueError:
                        x_row.append(0.0)

            if target_name:
                raw_y = row.get(target_name, "").strip()
                try:
                    y_val = float(raw_y) if raw_y else 0.0
                except ValueError:
                    y_val = 0.0
                y_rows.append([y_val])

            X_rows.append(x_row)

        # Compute normalization params for numeric columns
        numeric_indices: list[int] = []
        idx = 0
        for col in feature_names:
            if self.column_types.get(col) == "categorical":
                idx += len(self.encodings[col])
            else:
                numeric_indices.append(idx)
                idx += 1

        for i in numeric_indices:
            col_vals = [row[i] for row in X_rows]
            mean = sum(col_vals) / len(col_vals)
            variance = sum((v - mean) ** 2 for v in col_vals) / len(col_vals)
            std = math.sqrt(variance) if variance > 1e-12 else 1.0
            self.means[self.processed_feature_names[i]] = mean
            self.stds[self.processed_feature_names[i]] = std

        # Normalize numeric columns
        for i in numeric_indices:
            mean = self.means[self.processed_feature_names[i]]
            std = self.stds[self.processed_feature_names[i]]
            for row in X_rows:
                row[i] = (row[i] - mean) / std

        # Normalize target if numeric
        self.y_mean = 0.0
        self.y_std = 1.0
        if target_name and self.column_types.get(target_name) == "numeric":
            y_vals = [row[0] for row in y_rows]
            self.y_mean = sum(y_vals) / len(y_vals)
            y_variance = sum((v - self.y_mean) ** 2 for v in y_vals) / len(y_vals)
            self.y_std = math.sqrt(y_variance) if y_variance > 1e-12 else 1.0
            for row in y_rows:
                row[0] = (row[0] - self.y_mean) / self.y_std

        # Transpose to (n_features, n_samples) — each column is a sample
        X_T = [[X_rows[r][c] for r in range(len(X_rows))] for c in range(len(X_rows[0]))]
        self.X = Tensor(X_T)

        y_T = [[y_rows[r][c] for r in range(len(y_rows))] for c in range(len(y_rows[0]))] if target_name else None
        self.y = Tensor(y_T) if target_name else None

    @staticmethod
    def _detect_type(rows: list[dict[str, str]], col: str) -> str:
        for row in rows[:100]:
            val = row.get(col, "").strip()
            if val:
                try:
                    float(val)
                except ValueError:
                    return "categorical"
        return "numeric"

    def unnormalize_y(self, y: Tensor) -> Tensor:
        if self.y_std == 1.0 and self.y_mean == 0.0:
            return y
        flat = [v * self.y_std + self.y_mean for v in y.flat]
        return Tensor(flat)

    def __repr__(self) -> str:
        return (f'Dataset(data_path="{self.data_path}", '
                f'samples={self.n_samples}, '
                f'features={len(self.processed_feature_names)}, '
                f'target={self.target_name})')
