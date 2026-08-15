
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error

from data_processing import prepare_data_from_excels
from model import build_cnn_lstm_lnn_attention


RESULT_DIR = "E:/jiro/result"
FEATURE_COLS = ["phase", "v", "f"]
TARGET_COLS = ["MT", "WT", "EI", "CS", "Force"]

COLUMN_MAP = {
    "Time (s)": "time",
    "V(V)": "v",
    "LOAD (N)": "f",
    "AMT(°C)": "MT",
    "HWT(°C)": "WT",
    "AMD (%)": "EI",
    "AMS（MPa）": "CS",
    "AMF(N)": "Force",
}


EXCEL_PATHS = (
    [f"E:/jiro/{i}.xlsx" for i in range(1, 16)] +
    [f"E:/jiro/juxinbo/{i}.xlsx" for i in range(1, 4)]
)

VAL_GROUP_IDS = [10, 11, 12, 13, 14, 17]

LOOKBACK = 64
EPOCHS = 500
BATCH_SIZE = 32


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    ensure_dir(RESULT_DIR)

    print("=" * 50)
    print("Start training the model")
    print("=" * 50)


    print("\n data loading...")
    X_train_s, Y_train_s, X_val_s, Y_val_s, Y_train_raw, Y_val_raw, x_scaler, y_scaler = prepare_data_from_excels(
        excel_paths=EXCEL_PATHS,
        feature_cols=FEATURE_COLS,
        target_cols=TARGET_COLS,
        lookback=LOOKBACK,
        horizon=0,
        val_group_ids=VAL_GROUP_IDS,
        sheet_name=0,
        column_map=COLUMN_MAP,
        random_split=True,
        val_ratio=0.2,
        random_state=42,
    )

    print(f"traing set: {X_train_s.shape[0]} sample")
    print(f"val set: {X_val_s.shape[0]} sample")


    dataset_info = pd.DataFrame({
        "split": ["train", "validation"],
        "n_samples": [X_train_s.shape[0], X_val_s.shape[0]],
    })
    dataset_info.to_csv(os.path.join(RESULT_DIR, "dataset_info.csv"), index=False)


    np.save(os.path.join(RESULT_DIR, "X_train_s.npy"), X_train_s)
    np.save(os.path.join(RESULT_DIR, "Y_train_s.npy"), Y_train_s)
    np.save(os.path.join(RESULT_DIR, "Y_train_raw.npy"), Y_train_raw)
    np.save(os.path.join(RESULT_DIR, "X_val_s.npy"), X_val_s)
    np.save(os.path.join(RESULT_DIR, "Y_val_s.npy"), Y_val_s)
    np.save(os.path.join(RESULT_DIR, "Y_val_raw.npy"), Y_val_raw)


    use_pooling = LOOKBACK >= 8
    model = build_cnn_lstm_lnn_attention(
        lookback=LOOKBACK,
        in_dim=len(FEATURE_COLS),
        out_dim=len(TARGET_COLS),
        use_pooling=use_pooling
    )
    model.summary()


    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(RESULT_DIR, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True
        ),
    ]


    print("\n begin training...")
    history = model.fit(
        X_train_s, Y_train_s,
        validation_data=(X_val_s, Y_val_s),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )


    model.save(os.path.join(RESULT_DIR, "final_model.keras"))
    joblib.dump(x_scaler, os.path.join(RESULT_DIR, "x_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(RESULT_DIR, "y_scaler.pkl"))


    hist_df = pd.DataFrame({
        "epoch": range(1, len(history.history["loss"]) + 1),
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "train_mae": history.history["mae"],
        "val_mae": history.history["val_mae"],
    })
    hist_df.to_csv(os.path.join(RESULT_DIR, "training_history.csv"), index=False)


    print("\n index...")


    Y_train_pred_s = model.predict(X_train_s, verbose=0)
    Y_val_pred_s = model.predict(X_val_s, verbose=0)


    Y_train_pred = y_scaler.inverse_transform(Y_train_pred_s)
    Y_val_pred = y_scaler.inverse_transform(Y_val_pred_s)


    metrics_list = []
    for split_name, y_true, y_pred in [
        ("train", Y_train_raw, Y_train_pred),
        ("val", Y_val_raw, Y_val_pred)
    ]:
        for i, name in enumerate(TARGET_COLS):
            yt = y_true[:, i]
            yp = y_pred[:, i]
            metrics_list.append({
                "split": split_name,
                "target": name,
                "R2": r2_score(yt, yp),
                "RMSE": np.sqrt(mean_squared_error(yt, yp)),
                "MAE": np.mean(np.abs(yt - yp)),
            })


    for split_name, y_true, y_pred in [
        ("train", Y_train_raw, Y_train_pred),
        ("val", Y_val_raw, Y_val_pred)
    ]:
        metrics_list.append({
            "split": split_name,
            "target": "OVERALL",
            "R2": r2_score(y_true.flatten(), y_pred.flatten()),
            "RMSE": np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten())),
            "MAE": np.mean(np.abs(y_true.flatten() - y_pred.flatten())),
        })

    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(os.path.join(RESULT_DIR, "final_metrics.csv"), index=False)


    print("\n" + "=" * 60)
    print("train index:")
    print("-" * 60)
    train_metrics = df_metrics[df_metrics["split"] == "train"]
    for _, row in train_metrics.iterrows():
        print(f"{row['target']:>8s} | R²={row['R2']:.4f}  RMSE={row['RMSE']:.4f}  MAE={row['MAE']:.4f}")

    print("\n val index:")
    print("-" * 60)
    val_metrics = df_metrics[df_metrics["split"] == "val"]
    for _, row in val_metrics.iterrows():
        print(f"{row['target']:>8s} | R²={row['R2']:.4f}  RMSE={row['RMSE']:.4f}  MAE={row['MAE']:.4f}")

    print("\n" + "=" * 50)
    print("ok")
    print(f"save: {RESULT_DIR}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
