from tensorflow.keras.models import load_model

model = load_model("ocr_model_kaggle.keras")
model.save("ocr_model_legacy.h5", save_format="h5")


