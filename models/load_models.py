from ultralytics import ASSETS, SAM, FastSAM

models_path = {"fastSAM": "FastSAM-s.pt",
               "SAM2-s": "sam2_s.pt"}

def install_models()->None:
    pass

def get_model():
    fastSAM_model = FastSAM(models_path["fastSAM"])
    print(f"FastSAM model info: \n {fastSAM_model.info()}")
    sam2_model = SAM(models_path["SAM2-s"])
    print(f"SAM2 model info: \n {sam2_model.info()}")


if __name__ == "__main__":
    print("Models loaded successfully.")
    get_model()