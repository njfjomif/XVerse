import os
import subprocess
import tempfile
from pathlib import Path
from typing import List
from cog import BasePredictor, Input, Path as CogPath

class Predictor(BasePredictor):
    def setup(self):
        """
        Sets up the environment by exporting model paths.
        This runs once when the model container boots up.
        """
        print("Setting up model environment variables...")
        os.environ["FLORENCE2_MODEL_PATH"] = "./checkpoints/Florence-2-large"
        os.environ["SAM2_MODEL_PATH"] = "./checkpoints/sam2.1_hiera_large.pt"
        os.environ["FACE_ID_MODEL_PATH"] = "./checkpoints/model_ir_se50.pth"
        os.environ["CLIP_MODEL_PATH"] = "./checkpoints/clip-vit-large-patch14"
        os.environ["FLUX_MODEL_PATH"] = "./checkpoints/FLUX.1-dev"
        os.environ["DPG_VQA_MODEL_PATH"] = "./checkpoints/mplug_visual-question-answering_coco_large_en"
        os.environ["DINO_MODEL_PATH"] = "./checkpoints/dino-vits16"
        print("Model environment setup complete.")

    def predict(
        self,
        images: List[CogPath] = Input(description="A list of input images for subject reference."),
        captions: str = Input(description="Comma-separated list of captions for each image (e.g., 'a woman,a girl'). The model will use these as placeholders ENT1, ENT2, etc."),
        idips: str = Input(description="Comma-separated list of booleans (true/false) to indicate if ID weights should be used for each image (e.g., 'true,true')."),
        prompt: str = Input(description="The main text prompt. Use ENT1, ENT2, etc., as placeholders for your subjects."),
        seed: int = Input(description="Random seed for reproducibility.", default=42),
        target_height: int = Input(description="Height of the generated image.", default=768),
        target_width: int = Input(description="Width of the generated image.", default=768),
        weight_id: float = Input(description="Weight for subject identity consistency.", default=2.0),
        weight_ip: float = Input(description="Weight for image prompt consistency.", default=5.0),
        latent_lora_scale: float = Input(description="Scale for the latent LoRA.", default=0.85),
        vae_lora_scale: float = Input(description="Scale for the VAE LoRA.", default=1.3),
        num_images: int = Input(description="Number of images to generate.", default=1, ge=1, le=4)
    ) -> List[CogPath]:
        """Runs a single prediction on the model."""

        # Convert comma-separated string inputs into lists for the command line
        caption_list = [c.strip() for c in captions.split(',')]
        idips_list = [i.strip() for i in idips.split(',')]
        image_path_list = [str(p) for p in images]

        if not (len(image_path_list) == len(caption_list) == len(idips_list)):
            raise ValueError("The number of images, captions, and idips must be the same.")

        output_dir = Path(tempfile.mkdtemp())
        output_paths = []

        for i in range(num_images):
            current_seed = seed + i
            output_filename = f"output_{i}.png"
            output_path = output_dir / output_filename
            
            command = [
                "python", "inference_single_sample.py",
                "--prompt", prompt,
                "--seed", str(current_seed),
                "--target_height", str(target_height),
                "--target_width", str(target_width),
                "--weight_id", str(weight_id),
                "--weight_ip", str(weight_ip),
                "--latent_lora_scale", str(latent_lora_scale),
                "--vae_lora_scale", str(vae_lora_scale),
                "--images", *image_path_list,
                "--captions", *caption_list,
                "--idips", *idips_list,
                "--save_path", str(output_path),
                "--num_images", "1"
            ]

            print(f"Running command: {' '.join(command)}")
            try:
                # Using subprocess to call the original inference script
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                print(result.stdout)
                output_paths.append(output_path)
            except subprocess.CalledProcessError as e:
                print("--- Inference Script STDOUT ---")
                print(e.stdout)
                print("--- Inference Script STDERR ---")
                print(e.stderr)
                raise RuntimeError(f"The inference script failed: {e.stderr}") from e

        return [CogPath(p) for p in output_paths]