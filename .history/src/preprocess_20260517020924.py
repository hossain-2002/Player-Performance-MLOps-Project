"""Preprocessing utilities for player-performance-mlops.

This module provides functions to preprocess the `new-players-data-full.csv`
dataset and save processed training/testing arrays and fitted artifacts.
"""

from typing import Tuple
import os
import re
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def parse_money(val: object) -> float:
    """Parse a money string like "€100M", "€500K", "€1.2B" into float euros.

    Args:
        val: value to parse (str, int, float or NaN)

    Returns:
        float: numeric value in euros, or np.nan if unparsable.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        try:
            return float(val)
        except Exception:
            return np.nan

    s = str(val).strip()
    # remove currency symbols and spaces
    s = s.replace('€', '').replace('$', '').replace(',', '').strip()
    # regex to capture number and optional suffix
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([KMBkmb]?)$", s)
    if not m:
        try:
            return float(s)
        except Exception:
            return np.nan
    num, suf = m.groups()
    num = float(num)
    suf = suf.upper()
    if suf == 'K':
        return num * 1e3
    if suf == 'M':
        return num * 1e6
    if suf == 'B':
        return num * 1e9
    return num


def prepare_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder, StandardScaler]:
    """Preprocess the input DataFrame and return processed arrays and fitted encoders/scaler.

    Steps performed:
    - Drop specified unused columns
    - Separate outfield players (non-GK) and drop goalkeeper columns for outfield
    - Parse `value` and `wage` into numeric euros
    - Encode `best_position` and `preferred_foot` with LabelEncoder
    - Fill numeric nulls with median

    Args:
        df: raw DataFrame loaded from CSV

    Returns:
        Tuple of (processed_df, le_position, le_foot, scaler) where processed_df includes
        features and target column `overall_rating`.
    """
    df = df.copy()

    # 1) Drop columns if they exist
    drop_cols = [
        'player_slug', 'version', 'full_name', 'description', 'image', 'dob', 'play_styles'
    ]
    existing_drops = [c for c in drop_cols if c in df.columns]
    if existing_drops:
        df.drop(columns=existing_drops, inplace=True)

    # 2) Separate outfield players and goalkeepers
    if 'best_position' in df.columns:
        goalkeeper_mask = df['best_position'].astype(str).str.upper() == 'GK'
    else:
        goalkeeper_mask = pd.Series([False] * len(df), index=df.index)

    outfield = df[~goalkeeper_mask].copy()

    # 3) For outfield players, drop all gk_ prefixed columns
    gk_cols = [c for c in outfield.columns if c.startswith('gk_')]
    if gk_cols:
        outfield.drop(columns=gk_cols, inplace=True)

    # 4) Parse `value` column
    if 'value' in outfield.columns:
        outfield['value_eur'] = outfield['value'].apply(parse_money)

    # 5) Parse `wage` column
    if 'wage' in outfield.columns:
        outfield['wage_eur'] = outfield['wage'].apply(parse_money)

    # 6) Encode `best_position` and `preferred_foot`
    le_position = LabelEncoder()
    le_foot = LabelEncoder()

    if 'best_position' in outfield.columns:
        outfield['best_position'] = outfield['best_position'].fillna('missing').astype(str)
        outfield['best_position_enc'] = le_position.fit_transform(outfield['best_position'])
    else:
        outfield['best_position_enc'] = 0

    if 'preferred_foot' in outfield.columns:
        outfield['preferred_foot'] = outfield['preferred_foot'].fillna('missing').astype(str)
        outfield['preferred_foot_enc'] = le_foot.fit_transform(outfield['preferred_foot'])
    else:
        outfield['preferred_foot_enc'] = 0

    # 7) Fill remaining numeric nulls with column median
    numeric_cols = outfield.select_dtypes(include=[np.number]).columns.tolist()
    # Keep overall_rating as target; still fill its nulls if present
    for col in numeric_cols:
        median = outfield[col].median()
        outfield[col].fillna(median, inplace=True)

    return outfield, le_position, le_foot


def split_scale_save(df: pd.DataFrame,
                     le_position: LabelEncoder,
                     le_foot: LabelEncoder,
                     output_data_dir: str = None,
                     models_dir: str = None) -> None:
    """Split features/target, scale features, and save arrays and artifacts.

    Args:
        df: preprocessed DataFrame (outfield players) containing `overall_rating`.
        le_position: fitted LabelEncoder for positions.
        le_foot: fitted LabelEncoder for preferred_foot.
        output_data_dir: directory to save numpy arrays (defaults to `data/`)
        models_dir: directory to save pickled artifacts (defaults to `models/`)
    """
    if output_data_dir is None:
        output_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    if 'overall_rating' not in df.columns:
        raise ValueError('`overall_rating` target column not found in dataframe')

    X = df.drop(columns=['overall_rating'])
    y = df['overall_rating'].values

    # Remove original object columns that are not numeric (keep encoded columns)
    X = X.select_dtypes(include=[np.number]).copy()

    # 9) Apply StandardScaler to X
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42
    )

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 10) Save processed arrays as numpy files
    np.save(os.path.join(output_data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_data_dir, 'y_test.npy'), y_test)

    # 11) Save scaler and label encoders as pickle files
    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(models_dir, 'le_best_position.pkl'), 'wb') as f:
        pickle.dump(le_position, f)
    with open(os.path.join(models_dir, 'le_preferred_foot.pkl'), 'wb') as f:
        pickle.dump(le_foot, f)

    # 13) Print shapes
    print('X_train shape:', X_train.shape)
    print('X_test shape :', X_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape :', y_test.shape)


def main(csv_path: str = None) -> None:
    """Main entrypoint for preprocessing.

    Args:
        csv_path: optional path to the CSV file. If None, uses the project's data folder.
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'new-players-data-full.csv')

    df = pd.read_csv(csv_path)
    processed_df, le_pos, le_foot = prepare_dataframe(df)
    split_scale_save(processed_df, le_pos, le_foot,
                     output_data_dir=os.path.join(os.path.dirname(__file__), '..', 'data'),
                     models_dir=os.path.join(os.path.dirname(__file__), '..', 'models'))


if __name__ == '__main__':
    main()
