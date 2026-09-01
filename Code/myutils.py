import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.preprocessing import PowerTransformer

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error




def skew_transform(df, max_iterations=5):
    """
    Detects and reduces skewness in numeric features.

    For each numeric feature:
    - Skips constant columns
    - Skips binary columns
    - Skips ID columns
    - If skewness is already between -0.5 and 0.5,
      no transformation is applied
    - Otherwise repeatedly tries:
        1. Square Root
        2. Square
        3. Log1p
        4. Yeo-Johnson
        5. Box-Cox

    At each iteration, the transformation producing the
    smallest absolute skewness is selected.

    The loop stops when:
    - skewness is between -0.5 and 0.5
    - no transformation improves skewness
    - max_iterations is reached

    Returns
    -------
    transformed_df : DataFrame
        Copy of the original dataframe with transformed
        numeric features.

    skew_table : DataFrame
        Summary containing:
        - Feature
        - Original Skewness
        - Transformations
        - Final Skewness
        - Iterations
        - Status
    """

    # ---------------------------------------
    # Copy original dataframe
    # ---------------------------------------
    transformed_df = df.copy()

    results = []

    # Select numeric columns
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    # ---------------------------------------
    # Loop through numeric columns
    # ---------------------------------------
    for column in numeric_columns:

        # Remove missing values temporarily
        original = (
            df[column]
            .dropna()
            .astype(float)
        )

        unique_count = original.nunique()

        # -----------------------------------
        # Skip empty / constant columns
        # -----------------------------------
        if unique_count <= 1:
            continue

        # -----------------------------------
        # Skip binary columns
        # -----------------------------------
        if unique_count == 2:
            continue

        # -----------------------------------
        # Skip ID columns
        # -----------------------------------
        column_lower = column.lower()

        if (
            column_lower == "id"
            or column_lower.endswith("_id")
            or column_lower.startswith("id_")
        ):
            continue

        # -----------------------------------
        # Original skewness
        # -----------------------------------
        original_skew = original.skew()

        # Start with original values
        current = original.copy()

        transformations_used = []

        iteration = 0

        # ===================================
        # Repeated transformation loop
        # ===================================
        while (
            abs(current.skew()) > 0.5
            and iteration < max_iterations
        ):

            current_skew = current.skew()

            candidates = {}

            # ===================================
            # 1. Square Root
            # ===================================
            if current.min() >= 0:

                sqrt_values = np.sqrt(current)

                sqrt_values = pd.Series(
                    sqrt_values,
                    index=current.index
                )

                candidates["Square Root"] = {
                    "values": sqrt_values,
                    "skewness": sqrt_values.skew()
                }

            # ===================================
            # 2. Square
            # Mainly useful for left skew
            # ===================================
            if (
                current_skew < 0
                and current.min() >= 0
            ):

                square_values = np.square(current)

                square_values = pd.Series(
                    square_values,
                    index=current.index
                )

                candidates["Square"] = {
                    "values": square_values,
                    "skewness": square_values.skew()
                }

            # ===================================
            # 3. Log1p
            # ===================================
            if current.min() >= 0:

                log_values = np.log1p(current)

                log_values = pd.Series(
                    log_values,
                    index=current.index
                )

                candidates["Log1p"] = {
                    "values": log_values,
                    "skewness": log_values.skew()
                }

            # ===================================
            # 4. Yeo-Johnson
            # Works with negative / zero values
            # ===================================
            try:

                pt_yeo = PowerTransformer(
                    method="yeo-johnson",
                    standardize=False
                )

                yeo_values = pt_yeo.fit_transform(
                    current
                    .to_numpy()
                    .reshape(-1, 1)
                ).flatten()

                yeo_values = pd.Series(
                    yeo_values,
                    index=current.index
                )

                candidates["Yeo-Johnson"] = {
                    "values": yeo_values,
                    "skewness": yeo_values.skew()
                }

            except Exception:
                pass

            # ===================================
            # 5. Box-Cox
            # Requires strictly positive values
            # ===================================
            try:

                # Shift values if they contain
                # zero or negative numbers
                if current.min() <= 0:

                    shift = (
                        abs(current.min()) + 1
                    )

                    boxcox_input = (
                        current + shift
                    )

                else:

                    boxcox_input = (
                        current.copy()
                    )

                # Box-Cox requires variation
                if boxcox_input.nunique() > 1:

                    pt_boxcox = PowerTransformer(
                        method="box-cox",
                        standardize=False
                    )

                    boxcox_values = (
                        pt_boxcox
                        .fit_transform(
                            boxcox_input
                            .to_numpy()
                            .reshape(-1, 1)
                        )
                        .flatten()
                    )

                    boxcox_values = pd.Series(
                        boxcox_values,
                        index=current.index
                    )

                    candidates["Box-Cox"] = {
                        "values": boxcox_values,
                        "skewness":
                            boxcox_values.skew()
                    }

            except Exception:
                pass

            # ===================================
            # Check if transformations exist
            # ===================================
            if len(candidates) == 0:
                break

            # ===================================
            # Remove invalid candidates
            # ===================================
            candidates = {
                name: result
                for name, result
                in candidates.items()

                if np.isfinite(
                    result["skewness"]
                )
            }

            if len(candidates) == 0:
                break

            # ===================================
            # Find transformation closest to 0
            # ===================================
            best_name = min(
                candidates,
                key=lambda name: abs(
                    candidates[name][
                        "skewness"
                    ]
                )
            )

            best_values = (
                candidates[
                    best_name
                ]["values"]
            )

            best_skew = (
                candidates[
                    best_name
                ]["skewness"]
            )

            # ===================================
            # Stop if no improvement
            # ===================================
            if abs(best_skew) >= abs(
                current_skew
            ):
                break

            # ===================================
            # Apply best transformation
            # ===================================
            current = best_values.astype(
                float
            )

            transformations_used.append(
                best_name
            )

            iteration += 1

        # =======================================
        # Convert original dataframe column
        # to float before inserting decimals
        # =======================================
        transformed_df[column] = (
            transformed_df[column]
            .astype(float)
        )

        # =======================================
        # Store final transformed values
        # =======================================
        transformed_df.loc[
            current.index,
            column
        ] = current.astype(float)

        # =======================================
        # Calculate final skewness
        # =======================================
        final_skew = current.skew()

        # =======================================
        # Determine status
        # =======================================
        if abs(final_skew) <= 0.5:
            status = "Acceptable"
        else:
            status = "Still skewed"

        # Transformation description
        if transformations_used:

            transformation_text = (
                " -> ".join(
                    transformations_used
                )
            )

        else:

            transformation_text = "None"

        # =======================================
        # Save results
        # =======================================
        results.append({

            "Feature":
                column,

            "Original Skewness":
                round(
                    original_skew,
                    3
                ),

            "Transformations":
                transformation_text,

            "Final Skewness":
                round(
                    final_skew,
                    3
                ),

            "Iterations":
                iteration,

            "Status":
                status
        })

    # ===========================================
    # Create summary table
    # ===========================================
    skew_table = pd.DataFrame(
        results
    )

    return transformed_df, skew_table


def plot_transformations(df, skew_table):

    """
    Applies the best transformation selected by skew_calc()
    and plots before vs after.

    Returns a transformed copy of the dataframe.
    """

    # Keep original dataframe
    transformed_df = df.copy()

    for _, row in skew_table.iterrows():

        feature = row["Feature"]

        transformation = row[
            "Recommended Transformation"
        ]

        valid_mask = df[feature].notna()

        original = (
            df.loc[valid_mask, feature]
            .astype(float)
        )

        # ===============================
        # Apply transformation
        # ===============================

        if transformation == "None":

            transformed = original.copy()

        elif transformation == "Square Root":

            transformed = np.sqrt(original)

        elif transformation == "Square":

            transformed = np.square(original)

        elif transformation == "Log1p":

            transformed = np.log1p(original)

        elif transformation == "Yeo-Johnson":

            pt = PowerTransformer(
                method="yeo-johnson",
                standardize=False
            )

            values = pt.fit_transform(
                original.to_numpy().reshape(-1, 1)
            ).flatten()

            transformed = pd.Series(
                values,
                index=original.index
            )

        else:

            transformed = original.copy()

        # ==================================
        # Save transformed feature
        # ==================================

        new_column = f"{feature}_transform"

        transformed_df.loc[
            valid_mask,
            new_column
        ] = transformed

        # ==================================
        # Calculate skewness
        # ==================================

        before_skew = original.skew()
        after_skew = transformed.skew()

        # ==================================
        # Plot
        # ==================================

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 4)
        )

        sns.histplot(
            original,
            kde=True,
            ax=axes[0]
        )

        axes[0].set_title(
            f"{feature} — Before\n"
            f"Skewness = {before_skew:.3f}"
        )

        sns.histplot(
            transformed,
            kde=True,
            ax=axes[1]
        )

        axes[1].set_title(
            f"{feature} — {transformation}\n"
            f"Skewness = {after_skew:.3f}"
        )

        plt.tight_layout()
        plt.show()

    return transformed_df


    
def apply_transformations(df, skew_table):
    """
    Applies the transformation chain already recorded in `skew_table`
    (the "Transformations" column produced by skew_transform, e.g.
    "Box-Cox -> Yeo-Johnson") to the matching columns in `df`.
 
    Use this to replay a previously-found transformation chain on new
    data (e.g. a test set, or a fresh batch) instead of re-running the
    full skewness search on it.
 
    Parameters
    ----------
    df : DataFrame
        Data to transform.
    skew_table : DataFrame
        Output table from skew_transform, containing "Feature" and
        "Transformations" columns.
 
    Returns
    -------
    transformed_df : DataFrame
        Copy of df with each column's recorded transformation chain
        applied. Columns with "Transformations" == "None", or not
        present in df, are left unchanged.
    """
 
    transformed_df = df.copy()
 
    # ---------------------------------------
    # Single-step transformation functions,
    # matching the ones used in skew_transform
    # ---------------------------------------
    def _square_root(s):
        return pd.Series(np.sqrt(s.to_numpy()), index=s.index)
 
    def _square(s):
        return pd.Series(np.square(s.to_numpy()), index=s.index)
 
    def _log1p(s):
        return pd.Series(np.log1p(s.to_numpy()), index=s.index)
 
    def _yeo_johnson(s):
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        values = pt.fit_transform(
            s.to_numpy().reshape(-1, 1)
        ).flatten()
        return pd.Series(values, index=s.index)
 
    def _box_cox(s):
        # Box-Cox requires strictly positive input;
        # shift if needed, same as in skew_transform
        if s.min() <= 0:
            shifted = s + (abs(s.min()) + 1)
        else:
            shifted = s.copy()
 
        pt = PowerTransformer(method="box-cox", standardize=False)
        values = pt.fit_transform(
            shifted.to_numpy().reshape(-1, 1)
        ).flatten()
        return pd.Series(values, index=s.index)
 
    step_functions = {
        "Square Root": _square_root,
        "Square": _square,
        "Log1p": _log1p,
        "Yeo-Johnson": _yeo_johnson,
        "Box-Cox": _box_cox,
    }
 
    # ---------------------------------------
    # Replay each feature's recorded chain
    # ---------------------------------------
    for _, row in skew_table.iterrows():
 
        column = row["Feature"]
        chain_str = row["Transformations"]
 
        if column not in transformed_df.columns:
            continue
 
        if pd.isna(chain_str) or chain_str == "None":
            continue
 
        current = (
            transformed_df[column]
            .dropna()
            .astype(float)
        )
 
        for step_name in chain_str.split(" -> "):
 
            step_name = step_name.strip()
            step_func = step_functions.get(step_name)
 
            if step_func is None:
                continue
 
            try:
                current = step_func(current)
            except Exception:
                # if a step can't be applied to this data
                # (e.g. new negative values), stop the chain here
                break
 
        transformed_df[column] = (
            transformed_df[column]
            .astype(float)
        )
 
        transformed_df.loc[
            current.index,
            column
        ] = current.astype(float)
 
    return transformed_df

    
def evaluate_model(model, X_train, X_test, y_train_log, y_test_log, model_name="Model"):
    """
    Predicts and calculates R2 and RMSE for both train and test sets.
    Then prints the Train R2 and RMSE along with the Test R2 and RMSE
    """
    print(f"--- {model_name} Performance ---")
    
    # Generate predictions for training and testing data
    y_train_pred_log = model.predict(X_train)
    y_test_pred_log = model.predict(X_test)

    # 2. Convert predictions back to the original rent scale
    y_train_pred = np.expm1(y_train_pred_log)
    y_test_pred = np.expm1(y_test_pred_log)

    # 3. Convert actual values back to the original rent scale
    y_train = np.expm1(y_train_log)
    y_test = np.expm1(y_test_log)
    # Calculate R² for training and testing data
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Calculate RMSE for training and testing data
    train_rmse = np.sqrt(
        mean_squared_error(y_train, y_train_pred)
    )

    test_rmse = np.sqrt(
        mean_squared_error(y_test, y_test_pred)
    )

    # Calculate MAE for training and testing data
    train_rmse = np.sqrt(
        mean_squared_error(y_train, y_train_pred)
    )

    test_rmse = np.sqrt(
        mean_squared_error(y_test, y_test_pred)
    )

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(f"Train MAE: {train_mae}")  # Output: 0.5
    print(f"Test MAE: {test_mae}")  # Output: 0.5
    # Print results
    print(f"Train R²:   {train_r2:.4f}")
    print(f"Test R²:    {test_r2 :.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE:  {test_rmse:.4f}")    