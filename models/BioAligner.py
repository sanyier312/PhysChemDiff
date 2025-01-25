import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

class CLIPModel(nn.Module):
    def __init__(self, text_encoder, sequence_encoder):
        super(CLIPModel, self).__init__()
        self.text_encoder = text_encoder  # Pre-trained text encoding model
        self.sequence_encoder = sequence_encoder  # Sequence encoding model

    def forward(self, text_input, sequence_input, padding_mask):
        # Encode text input
        text_embeddings = self.text_encoder.encode(text_input, convert_to_tensor=True)
        # Encode sequence input
        sequence_embeddings = self.sequence_encoder(sequence_input, padding_mask)

        # Average pooling over sequence embeddings
        sequence_embeddings = sequence_embeddings.mean(dim=1)  # Average over the sequence dimension

        return text_embeddings, sequence_embeddings

class TransformerSequenceEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, seq_length, num_heads, num_layers, output_dim):
        super(TransformerSequenceEncoder, self).__init__()

        # Linear layer to project the input to the embedding dimension
        self.input_projection = nn.Linear(input_dim, embedding_dim)

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, embedding_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final Linear layer to project to the output dimension
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x, padding_mask):
        x = x.to(torch.float32)  # Ensure input data is float

        # Apply the input projection
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Apply the Transformer encoder with padding mask
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Apply the final linear layer
        return self.fc(x)

def test_model():
    # Assume some parameters
    input_dim = 18       # Input feature dimension
    embedding_dim = 384  # Embedding dimension
    seq_length = 128     # Sequence length
    num_heads = 2        # Number of attention heads
    num_layers = 3       # Number of Transformer layers
    batch_size = 64      # Batch size

    # Initialize text encoder and sequence encoder
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained text encoder
    sequence_encoder = TransformerSequenceEncoder(input_dim, embedding_dim, seq_length, num_heads, num_layers, embedding_dim)

    # Initialize CLIP model
    model = CLIPModel(text_encoder, sequence_encoder)

    # Generate random input
    text_input = ["This is a test sentence."] * batch_size  # Example text input
    sequence_input = torch.rand(batch_size, seq_length, input_dim)  # Randomly generated sequence input

    # Create a random padding mask
    # Let's assume the first half of the sequence is valid and the rest is padding
    padding_mask = torch.zeros(batch_size, seq_length).bool()  # Initialize padding mask
    padding_mask[:, 64:] = 1  # Example: set the second half of each sequence to padding

    # Run the model
    with torch.no_grad():
        text_embeddings, sequence_embeddings = model(text_input, sequence_input, padding_mask)

    # Output the shapes of embeddings
    print(f'Text embeddings shape: {text_embeddings.shape}')
    print(f'Sequence embeddings shape: {sequence_embeddings.shape}')


if __name__ == "__main__":
    # Call the test function
    test_model()
