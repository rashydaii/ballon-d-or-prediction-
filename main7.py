import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 1. SETTINGS
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)

# Change these paths if your files are elsewhere
HISTORICAL_FILE = r"C:\Users\Acer\Downloads\archive (3)\ballondor_performance_vs_pr.csv"
DATA_2026_FILE = r"C:\Users\Acer\Downloads\2026_ballon_dor_dataset.csv"

RESULTS_FOLDER = "results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

print("\n" + "=" * 75)
print("       BALLON D'OR 2026 WINNER PREDICTION")
print("=" * 75)


# ============================================================
# 2. LOAD HISTORICAL DATA
# ============================================================

print("\nLoading historical dataset...")

try:
    historical_df = pd.read_csv(HISTORICAL_FILE)
except FileNotFoundError:
    print(f"\nERROR: Could not find:\n{HISTORICAL_FILE}")
    print("Check the file path.")
    raise SystemExit

print("Historical dataset loaded.")
print("Shape:", historical_df.shape)
print("Columns:", historical_df.columns.tolist())


if "Rank" not in historical_df.columns:
    print("\nERROR: Historical dataset must contain 'Rank'.")
    raise SystemExit

print("\nLoading 2026 dataset...")

try:
    df2026 = pd.read_csv(DATA_2026_FILE)
except FileNotFoundError:
    print(f"ERROR: Could not find {DATA_2026_FILE}")
    raise SystemExit

print("\n2026 dataset loaded.")
print("Shape:", df2026.shape)

print("\n2026 COLUMNS:")
print(df2026.columns.tolist())
# ============================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n" + "=" * 75)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 75)

print("\nDataset Shape:")
print(historical_df.shape)

print("\nColumn Data Types:")
print(historical_df.dtypes)

print("\nMissing Values:")
print(historical_df.isnull().sum())

print("\nDescriptive Statistics:")
print(historical_df.describe(include="all").transpose())

# Save EDA summary for the report
eda_summary = pd.DataFrame({
    "Column": historical_df.columns,
    "Data_Type": historical_df.dtypes.astype(str).values,
    "Missing_Values": historical_df.isnull().sum().values,
    "Unique_Values": historical_df.nunique().values
})

eda_summary.to_csv(
    os.path.join(RESULTS_FOLDER, "eda_summary.csv"),
    index=False
)

# Winner distribution
plt.figure(figsize=(6, 5))
historical_df["Rank"].eq(1).astype(int).value_counts().sort_index().plot(
    kind="bar"
)
plt.title("Ballon d'Or Winner Class Distribution")
plt.xlabel("Winner (0 = No, 1 = Yes)")
plt.ylabel("Number of Records")
plt.xticks([0, 1], ["Not Winner", "Winner"], rotation=0)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_FOLDER, "target_distribution.png"),
    dpi=300
)
plt.show()

# Numerical correlation heatmap
numeric_eda = historical_df.select_dtypes(include=np.number)

if numeric_eda.shape[1] >= 2:
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        numeric_eda.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )
    plt.title("Numerical Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_FOLDER, "correlation_heatmap.png"),
        dpi=300
    )
    plt.show()

print("\nEDA completed. EDA files and graphs were saved in the results folder.")

# ============================================================
# 3. CREATE TARGET
# ============================================================

# Rank 1 = Ballon d'Or winner
historical_df["Winner"] = (
    historical_df["Rank"] == 1
).astype(int)

print("\nTarget distribution:")
print(historical_df["Winner"].value_counts())


# ============================================================
# 4. PREPARE FEATURES
# ============================================================

# Do not use Rank or Points because they leak the final result.
# Do not use Player because it is an identifier.

remove_columns = [
    "Winner",
    "Rank",
    "Points",
    "Player",
    "Name"
]

feature_columns = [
    c for c in historical_df.columns
    if c not in remove_columns
]

X = historical_df[feature_columns].copy()
y = historical_df["Winner"].copy()

print("\nTraining features:")
print(feature_columns)


# ============================================================
# 5. ENCODE CATEGORICAL FEATURES
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    dummy_na=True
)

X = X.apply(pd.to_numeric, errors="coerce")


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

imputer = SimpleImputer(strategy="median")

X_imputed = imputer.fit_transform(X)

X = pd.DataFrame(
    X_imputed,
    columns=X.columns,
    index=X.index
)

training_feature_names = X.columns.tolist()

print("\nNumber of processed features:", len(training_feature_names))


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 8. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 9. CLASS WEIGHTS
# ============================================================

classes = np.unique(y_train)

class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(
    zip(classes, class_weight_values)
)

print("\nClass weights:")
print(class_weights)


# ============================================================
# 10. MLP MODEL
# ============================================================

mlp_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation="relu"),
    Dropout(0.30),
    Dense(32, activation="relu"),
    Dropout(0.20),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

mlp_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

mlp_model.summary()

early_stopping_mlp = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

print("\nTraining MLP...")

mlp_history = mlp_model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.20,
    epochs=100,
    batch_size=8,
    class_weight=class_weights,
    callbacks=[early_stopping_mlp],
    verbose=1
)

mlp_probability = mlp_model.predict(
    X_test_scaled,
    verbose=0
).ravel()

mlp_prediction = (
    mlp_probability >= 0.50
).astype(int)

print("\n" + "=" * 75)
print("MODEL 1: MLP NEURAL NETWORK")
print("=" * 75)


# ============================================================
# 11. DNN MODEL
# ===========================================================
dnn_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(128, activation="relu"),
    Dropout(0.30),
    Dense(64, activation="relu"),
    Dropout(0.30),
    Dense(32, activation="relu"),
    Dropout(0.20),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

dnn_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

dnn_model.summary()

early_stopping_dnn = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

print("\nTraining DNN...")

dnn_history = dnn_model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.20,
    epochs=100,
    batch_size=8,
    class_weight=class_weights,
    callbacks=[early_stopping_dnn],
    verbose=1
)

dnn_probability = dnn_model.predict(
    X_test_scaled,
    verbose=0
).ravel()

dnn_prediction = (
    dnn_probability >= 0.50
).astype(int)

print("\n" + "=" * 75)
print("MODEL 2: DEEP NEURAL NETWORK")
print("=" * 75)


# ============================================================
# 12. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, y_true, y_pred, probabilities):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    try:
        auc = roc_auc_score(
            y_true, probabilities
        )
    except ValueError:
        auc = 0.0

    print("\n" + "-" * 75)
    print(model_name)
    print("-" * 75)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Not Winner", "Winner"],
            zero_division=0
        )
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc
    }


mlp_results = evaluate_model(
    "MLP Neural Network",
    y_test,
    mlp_prediction,
    mlp_probability
)

dnn_results = evaluate_model(
    "Deep Neural Network",
    y_test,
    dnn_prediction,
    dnn_probability
)


# ============================================================
# 13. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame([
    mlp_results,
    dnn_results
])

print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)
print(comparison.to_string(index=False))

comparison.to_csv(
    os.path.join(
        RESULTS_FOLDER,
        "model_comparison.csv"
    ),
    index=False
)


# ============================================================
# 14. ROC CURVE
# ============================================================

fpr_mlp, tpr_mlp, _ = roc_curve(
    y_test, mlp_probability
)

fpr_dnn, tpr_dnn, _ = roc_curve(
    y_test, dnn_probability
)

auc_mlp = roc_auc_score(
    y_test, mlp_probability
)

auc_dnn = roc_auc_score(
    y_test, dnn_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr_mlp,
    tpr_mlp,
    label=f"MLP (AUC = {auc_mlp:.3f})"
)

plt.plot(
    fpr_dnn,
    tpr_dnn,
    label=f"DNN (AUC = {auc_dnn:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    "--",
    label="Random Guess"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Ballon d'Or Classification")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "roc_curve.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    title,
    filename
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\n" + title)
    print(cm)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    plt.xticks(
        [0, 1],
        ["Not Winner", "Winner"]
    )

    plt.yticks(
        [0, 1],
        ["Not Winner", "Winner"]
    )

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_FOLDER,
            filename
        ),
        dpi=300
    )

    plt.show()


save_confusion_matrix(
    y_test,
    mlp_prediction,
    "MLP Confusion Matrix",
    "mlp_confusion_matrix.png"
)

save_confusion_matrix(
    y_test,
    dnn_prediction,
    "DNN Confusion Matrix",
    "dnn_confusion_matrix.png"
)


# ============================================================
# 16. SELECT BEST MODEL
# ============================================================

# F1 is used because Winner is a minority class.
if dnn_results["F1"] > mlp_results["F1"]:
    best_model = dnn_model
    best_model_name = "Deep Neural Network"
else:
    best_model = mlp_model
    best_model_name = "MLP Neural Network"

print("\n" + "=" * 75)
print("BEST MODEL")
print("=" * 75)
print("Selected:", best_model_name)


# ============================================================
# 17. LOAD 2026 DATASET
# ============================================================

print("\n" + "=" * 75)
print("LOADING 2026 PLAYER DATA")
print("=" * 75)

try:
    df2026 = pd.read_csv(DATA_2026_FILE)
except FileNotFoundError:
    print(f"\nERROR: Could not find:")
    print(DATA_2026_FILE)
    raise SystemExit

print("\n2026 dataset shape:", df2026.shape)
print("\n2026 columns:")
print(df2026.columns.tolist())


# ============================================================
# 18. GET PLAYER NAMES
# ============================================================

player_names = df2026["Player"].astype(str)


# ============================================================
# 19. CREATE 2026 PERFORMANCE SCORE
# ============================================================
#
# This score uses the actual performance variables in the
# 2026 dataset.
#
# It is NOT an official Ballon d'Or voting score.
# It is a feature-based score for ranking candidates.
#
# ============================================================

score = (
    df2026["League_Rating"] * 0.15
    + df2026["League_Goals"] * 1.00
    + df2026["League_Assists"] * 0.80
    + df2026["UCL_Rating"] * 0.20
    + df2026["UCL_Goals"] * 1.50
    + df2026["UCL_Assists"] * 1.20
    + df2026["WC_Rating"] * 0.20
    + df2026["WC_Goals"] * 1.50
    + df2026["WC_Assists"] * 1.20
    + df2026["International_Goals"] * 0.80
    + df2026["International_Assists"] * 0.60
    + df2026["Trophies"] * 5.00
    + df2026["MOTM"] * 0.50
    + df2026["Google_Trend"] * 0.05
    + df2026["Wiki_Page_Views"] * 0.05
)


# ============================================================
# 20. NORMALIZE PERFORMANCE SCORE
# ============================================================

score_min = score.min()
score_max = score.max()

if score_max > score_min:

    performance_score = (
        (score - score_min)
        /
        (score_max - score_min)
    ) * 100

else:

    performance_score = pd.Series(
        50.0,
        index=df2026.index
    )


# ============================================================
# 21. PREPARE 2026 DATA FOR THE NEURAL NETWORK
# ============================================================

remove_2026_columns = [
    "Player",
    "Name",
    "Rank",
    "Points",
    "Winner",
    "Top3"
]

X2026 = df2026.drop(
    columns=[
        c for c in remove_2026_columns
        if c in df2026.columns
    ],
    errors="ignore"
)


# Encode categorical columns
categorical_2026 = X2026.select_dtypes(
    include=["object", "category"]
).columns.tolist()

X2026 = pd.get_dummies(
    X2026,
    columns=categorical_2026,
    dummy_na=True
)

X2026 = X2026.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# 22. MAP 2026 FEATURES TO HISTORICAL FEATURES
# ============================================================
#
# Your historical dataset has different column names.
# Therefore, manually create the same feature structure.
#
# ============================================================

X2026_mapped = pd.DataFrame(
    0.0,
    index=df2026.index,
    columns=training_feature_names
)


def put_feature(
    historical_name,
    values
):

    if historical_name in X2026_mapped.columns:

        X2026_mapped[
            historical_name
        ] = values


# Basic features

put_feature(
    "League Rating",
    df2026["League_Rating"]
)

put_feature(
    "UCL Rating",
    df2026["UCL_Rating"]
)

put_feature(
    "WC Rating",
    df2026["WC_Rating"]
)


# Other competition rating
if "Other Rating" in X2026_mapped.columns:

    X2026_mapped["Other Rating"] = (
        df2026["International_Goals"]
        * 0.5
        +
        df2026["International_Assists"]
        * 0.3
    )


# Google Trends

put_feature(
    "Time-Weighted Google Trend Score",
    df2026["Google_Trend"]
)


# Wikipedia

put_feature(
    "Wiki Page Views",
    df2026["Wiki_Page_Views"]
)


# ============================================================
# 23. CONVERT PERFORMANCE INTO HISTORICAL BINARY FEATURES
# ============================================================

# League

if "League Top Scorer" in X2026_mapped.columns:

    X2026_mapped[
        "League Top Scorer"
    ] = (
        df2026["League_Goals"] >=
        df2026["League_Goals"].quantile(0.75)
    ).astype(float)


if "League Top Assist Provider" in X2026_mapped.columns:

    X2026_mapped[
        "League Top Assist Provider"
    ] = (
        df2026["League_Assists"] >=
        df2026["League_Assists"].quantile(0.75)
    ).astype(float)


# Champions League

if "UCL Top Scorer" in X2026_mapped.columns:

    X2026_mapped[
        "UCL Top Scorer"
    ] = (
        df2026["UCL_Goals"] >=
        df2026["UCL_Goals"].quantile(0.75)
    ).astype(float)


if "UCL Top Assist Provider" in X2026_mapped.columns:

    X2026_mapped[
        "UCL Top Assist Provider"
    ] = (
        df2026["UCL_Assists"] >=
        df2026["UCL_Assists"].quantile(0.75)
    ).astype(float)


# World Cup

if "WC Top Scorer" in X2026_mapped.columns:

    X2026_mapped[
        "WC Top Scorer"
    ] = (
        df2026["WC_Goals"] >=
        df2026["WC_Goals"].quantile(0.75)
    ).astype(float)


if "WC Top Assist Provider" in X2026_mapped.columns:

    X2026_mapped[
        "WC Top Assist Provider"
    ] = (
        df2026["WC_Assists"] >=
        df2026["WC_Assists"].quantile(0.75)
    ).astype(float)


# ============================================================
# 24. TROPHIES
# ============================================================

if "Continental Cup" in X2026_mapped.columns:

    X2026_mapped[
        "Continental Cup"
    ] = (
        df2026["Trophies"] > 0
    ).astype(float)


# ============================================================
# 25. HANDLE MISSING VALUES
# ============================================================

X2026_mapped = X2026_mapped.replace(
    [np.inf, -np.inf],
    np.nan
)

X2026_mapped = X2026_mapped.fillna(0)


# ============================================================
# 26. SCALE 2026 FEATURES
# ============================================================

X2026_scaled = scaler.transform(
    X2026_mapped
)


# ============================================================
# 27. GET NEURAL NETWORK SCORES
# ============================================================

mlp_2026_output = (
    mlp_model.predict(
        X2026_scaled,
        verbose=0
    ).ravel()
)

dnn_2026_output = (
    dnn_model.predict(
        X2026_scaled,
        verbose=0
    ).ravel()
)


# ============================================================
# 28. NORMALIZE EACH MODEL OUTPUT
# ============================================================

def normalize_scores(values):

    minimum = values.min()
    maximum = values.max()

    if maximum > minimum:

        return (
            (values - minimum)
            /
            (maximum - minimum)
        ) * 100

    else:

        return np.full(
            len(values),
            50.0
        )


mlp_score = normalize_scores(
    mlp_2026_output
)

dnn_score = normalize_scores(
    dnn_2026_output
)


# ============================================================
# 29. COMBINE MODEL + PERFORMANCE
# ============================================================
#
# 25% MLP
# 25% DNN
# 50% actual performance score
#
# This produces a meaningful ranking even if the neural
# networks output nearly identical probabilities.
#
# ============================================================

final_score = (

    mlp_score * 0.25

    +

    dnn_score * 0.25

    +

    performance_score.values * 0.50

)


# ============================================================
# 30. CREATE RESULTS TABLE
# ============================================================

prediction_results = pd.DataFrame({

    "Player": player_names,

    "MLP_Score": mlp_score,

    "DNN_Score": dnn_score,

    "Performance_Score":
        performance_score.values,

    "Final_Prediction_Score":
        final_score

})


# ============================================================
# 31. SORT
# ============================================================

prediction_results = (
    prediction_results
    .sort_values(
        by="Final_Prediction_Score",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# 32. ADD PREDICTED RANK
# ============================================================

prediction_results.insert(
    0,
    "Predicted_Rank",
    range(
        1,
        len(prediction_results) + 1
    )
)


prediction_results[
    "MLP_Score"
] = prediction_results[
    "MLP_Score"
].round(2)


prediction_results[
    "DNN_Score"
] = prediction_results[
    "DNN_Score"
].round(2)


prediction_results[
    "Performance_Score"
] = prediction_results[
    "Performance_Score"
].round(2)


prediction_results[
    "Final_Prediction_Score"
] = prediction_results[
    "Final_Prediction_Score"
].round(2)


# ============================================================
# 33. PRINT TOP 10
# ============================================================

print("\n")
print("=" * 85)
print("2026 BALLON D'OR PREDICTION RANKING")
print("=" * 85)

print(
    prediction_results[
        [
            "Predicted_Rank",
            "Player",
            "MLP_Score",
            "DNN_Score",
            "Performance_Score",
            "Final_Prediction_Score"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# 34. PREDICTED WINNER
# ============================================================

winner = prediction_results.iloc[0]

print("\n")
print("=" * 85)
print(" PREDICTED 2026 BALLON D'OR WINNER")
print("=" * 85)

print(
    f"\nPlayer: {winner['Player']}"
)

print(
    f"Final Prediction Score: "
    f"{winner['Final_Prediction_Score']:.2f}/100"
)

print(
    f"MLP Score: "
    f"{winner['MLP_Score']:.2f}"
)

print(
    f"DNN Score: "
    f"{winner['DNN_Score']:.2f}"
)

print(
    f"Performance Score: "
    f"{winner['Performance_Score']:.2f}"
)


# ============================================================
# 35. SAVE RESULTS
# ============================================================

prediction_results.to_csv(
    os.path.join(
        RESULTS_FOLDER,
        "2026_ballon_dor_predictions.csv"
    ),
    index=False
)


prediction_results.head(10).to_csv(
    os.path.join(
        RESULTS_FOLDER,
        "2026_top10_predictions.csv"
    ),
    index=False
)


# ============================================================
# 35A. RESULTS AND ANALYSIS SUMMARY
# ============================================================

best_by_f1 = comparison.loc[
    comparison["F1"].idxmax(),
    "Model"
]

best_by_auc = comparison.loc[
    comparison["AUC"].idxmax(),
    "Model"
]

analysis_text = (
    "Assignment 7 Results and Analysis\\n"
    "=================================\\n\\n"
    f"Best model by F1-score: {best_by_f1}\\n"
    f"Best model by ROC-AUC: {best_by_auc}\\n\\n"
    "The models were evaluated using accuracy, precision, recall, "
    "F1-score and ROC-AUC. F1-score is particularly useful here "
    "because Ballon d'Or winners form a minority class. ROC-AUC "
    "measures the models' ability to distinguish winners from "
    "non-winners across classification thresholds. The training "
    "and validation graphs can be used to discuss learning "
    "behaviour and possible overfitting. The confusion matrices "
    "show the numbers of true positives, true negatives, false "
    "positives and false negatives.\\n"
)

print("\n" + analysis_text)

with open(
    os.path.join(RESULTS_FOLDER, "results_analysis.txt"),
    "w",
    encoding="utf-8"
) as f:
    f.write(analysis_text)

# ============================================================
# 36. TOP 10 GRAPH
# ============================================================

top10 = prediction_results.head(10)

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    top10["Player"][::-1],
    top10["Final_Prediction_Score"][::-1]
)

plt.xlabel(
    "Final Prediction Score"
)

plt.ylabel(
    "Player"
)

plt.title(
    "Top 10 Predicted 2026 Ballon d'Or Candidates"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "2026_top10_prediction.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 37. SAVE WINNER
# ============================================================

with open(
    os.path.join(
        RESULTS_FOLDER,
        "2026_predicted_winner.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "2026 BALLON D'OR MODEL PREDICTION\n"
    )

    f.write(
        "=================================\n\n"
    )

    f.write(
        f"Predicted Winner: {winner['Player']}\n"
    )

    f.write(
        f"Final Prediction Score: "
        f"{winner['Final_Prediction_Score']:.2f}/100\n"
    )

    f.write(
        f"MLP Score: {winner['MLP_Score']:.2f}\n"
    )

    f.write(
        f"DNN Score: {winner['DNN_Score']:.2f}\n"
    )

    f.write(
        f"Performance Score: "
        f"{winner['Performance_Score']:.2f}\n"
    )


print("\n")
print("=" * 85)
print("PROJECT COMPLETED")
print("=" * 85)

print(
    f"\n Predicted Winner: "
    f"{winner['Player']}"
)

print(
    f"Final Score: "
    f"{winner['Final_Prediction_Score']:.2f}/100"
)

print(
    "\nResults saved in the 'results' folder."
)

print("=" * 85)