import torch
import torch.nn as nn
import pytorch_lightning as pl
from diffusers import DDPMScheduler
from utils.tensor_to_amino_acids import *
from data.acid_propertis import Features
import time
from data.Diffusion_dataset import *
import pandas as pd
from tqdm import tqdm
import argparse


def generate_sequences(model, latent_vectors, condition):
    model.eval()

    # Sample latent vectors from a standard normal distribution
    with torch.no_grad():
        condition = condition.to(latent_vectors.device)
        generated_sequences = model.decode(latent_vectors, condition)

    # Convert the generated tensor to protein sequences
    generated_sequences_list = [tensor_to_amino_acids(seq) for seq in generated_sequences]

    return generated_sequences_list


device = torch.device("cpu")

# Load pre-trained VAE model and UNet model
cvae_model = torch.load("pretrained/cvae.pth")
cvae_model.eval()

bioalinger_model = torch.load("pretrained/bioalinger.pth")
bioalinger_model.eval()

unet_model = torch.load("pretrained/diffusion.pth")
unet_model.eval()

# Define DDPM Scheduler (num_train_timesteps is now dynamic)
def get_noise_scheduler(num_train_timesteps):
    return DDPMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)

def inference(text_input, vae_model, unet_model, bioalinger_model, noise_scheduler, device, num, feature):
    print("Inferencing....")
    start_time = time.time()

    # Encode the text input
    text_condition = bioalinger_model.text_encoder.encode(text_input, convert_to_tensor=True).to(device)

    # Generate a random initial latent vector
    latent_dim = 512 # for the current pre-trained model
    with torch.no_grad():
        z = torch.randn((num, latent_dim)).to(device)
        z = z.unsqueeze(1)  # Adjust dimensions to match the UNet input

        # Gradually denoise
        for t in tqdm(reversed(range(noise_scheduler.config.num_train_timesteps)), desc="Denoising steps", unit="step"):
            t_tensor = torch.full((num,), t, device=device, dtype=torch.long).to(device)
            with torch.no_grad():
                predicted_noise = unet_model(z, t_tensor, text_condition).to(device)
                z = noise_scheduler.step(predicted_noise, t, z)['prev_sample'].to(device)

        # Final inference result
        output_sequence = vae_model.decode(z.squeeze(1), feature)

    end_time = time.time()
    print(f"Time taken for inference: {end_time - start_time:.2f} seconds")
    return output_sequence


def generate_features_from_dataset(dataset, num_samples):
    batch = []
    test_dataloader = dataset.get_test_dataloader(batch_size=num_samples)

    for _, features, _, _ in test_dataloader:
        batch.append(features)
        break  # Load only one batch

    # Stack all sequences into one batch tensor
    feature = torch.cat(batch, dim=0)
    return feature


def generate_random_features(num_samples, seq_length):
    amino_acid_features = Features()
    batch = []
    for _ in range(num_samples):
        sequence_tensor = amino_acid_features.generate_random_sequence_tensor(min_length=10, seq_length=seq_length)
        batch.append(sequence_tensor)
    return torch.stack(batch)


def generate_text_from_dataset(dataset, num_samples):
    test_dataloader = dataset.get_test_dataloader(batch_size=num_samples)
    for _, _, _, text in test_dataloader:
        return [text] * num_samples


def generate_text_from_user_input(num_samples):
    # Provide an interactive input method for the user
    print("Please enter the text description for each sample (press Enter to continue):")
    user_text = []
    for i in range(num_samples):
        print(f"Enter the text description for sample {i+1}:")
        text = input("Text: ")
        user_text.append(text)
    return user_text


def run_inference(use_dataset_for_features, use_dataset_for_text, num_samples, seq_length, custom_text=None, num_timesteps=1000):
    print(f"use_dataset_for_text: {use_dataset_for_text}")  # Print for debugging

    # Choose whether to use dataset or generate randomly based on input
    if use_dataset_for_features == "True":
        pt_file_path = f"data/swiss_onehot_and_feature_{seq_length}.pt"
        csv_file_path = f"data/swiss_text_{seq_length}.csv"
        
        dataset = Diffusion(pt_file_path, csv_file_path)
        feature = generate_features_from_dataset(dataset, num_samples)
    else:
        feature = generate_random_features(num_samples, seq_length)

    if use_dataset_for_text == "True":
        pt_file_path = f"data/swiss_onehot_and_feature_{seq_length}.pt"
        csv_file_path = f"data/swiss_text_{seq_length}.csv"
        dataset = Diffusion(pt_file_path, csv_file_path)
        text_input = generate_text_from_dataset(dataset, num_samples)
    else:
        if custom_text is None:
            print("Entering interactive mode")
            # Let the user provide custom text via interactive input
            text_input = generate_text_from_user_input(num_samples)
        else:
            print("Using custom text")
            # If custom text is provided, use it as input
            text_input = [custom_text] * num_samples

    # Create DDPM noise scheduler
    noise_scheduler = get_noise_scheduler(num_timesteps)

    # Run inference
    output_tensor = inference(text_input, cvae_model, unet_model, bioalinger_model, noise_scheduler, device, num_samples, feature)

    generated_sequences = [tensor_to_amino_acids(sequence) for sequence in output_tensor]

    # Create a dictionary to save the data
    data_to_save = {
        "Sequence_ID": [f"Seq_{i+1}" for i in range(len(generated_sequences))],
        "Protein_Sequence": generated_sequences
    }

    # Create DataFrame
    df = pd.DataFrame(data_to_save)

    # Define the path to save the CSV file
    output_csv_path = "result.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"Generated sequences have been saved to {output_csv_path}")

    # Define the path to save the FASTA file
    output_fasta_path = "result.fasta"
    with open(output_fasta_path, "w") as fasta_file:
        for i, sequence in enumerate(generated_sequences):
            fasta_file.write(f">Seq_{i+1}\n")
            fasta_file.write(f"{sequence}\n")
    print(f"Generated sequences have been saved to {output_fasta_path}")


if __name__ == "__main__":
    # Create command line argument parser
    parser = argparse.ArgumentParser(description="Protein sequence generation via diffusion model")
    parser.add_argument("--use_dataset_for_features", type=str, default="True", help="Whether to use dataset for features")
    parser.add_argument("--use_dataset_for_text", type=str, default="True", help="Whether to use dataset for text input")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length for protein sequences")
    parser.add_argument("--custom_text", type=str, default=None, help="Custom text input for generation")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of  timesteps for the inference")

    # Parse command line arguments
    args = parser.parse_args()

    # Run inference
    run_inference(args.use_dataset_for_features, args.use_dataset_for_text, args.num_samples, args.seq_length, args.custom_text, args.num_timesteps)
