import os
import joblib

class ModelLoader:
	_instance = None

	@classmethod
	def get_model(cls, model_path: str = "models/lgbm_demand_model.pkl"):
		if cls._instance is None:
			if not os.path.exists(model_path):
				raise FileNotFoundError(f"{model_path} does not exist")
			artifact = joblib.load(model_path)
			cls._instance = artifact
		return cls._instance

