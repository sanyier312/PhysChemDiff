import torch
import torch.nn.functional as F
from data.acid_propertis import Features
import math


# Direct physicochemical properties
def tensor_to_amino_acids_by_feature(input_tensor, end_token='*', zero_threshold=1e-5):
    # Create an instance of the Features class
    amino_acid_features = Features()

    # Get normalized physicochemical values for amino acids
    normalized_features = amino_acid_features.normalize_features()

    # Convert the physicochemical values of amino acids into tensors
    amino_acid_vectors = []
    amino_acids = list(normalized_features.keys())
    for aa in amino_acids:
        # Directly get the feature values from normalized_features[aa]
        vector = [normalized_features[aa][prop] for prop in normalized_features[aa]]
        amino_acid_vectors.append(vector)
    amino_acid_tensor = torch.tensor(amino_acid_vectors)

    # Calculate the similarity between each input vector and the amino acid vectors, 
    # and select the most similar amino acid
    sequence = []
    for input_vector in input_tensor:
        # Check if the vector is a zero vector using L2 norm (Euclidean distance)
        if torch.norm(input_vector) < zero_threshold:
            sequence.append(end_token)
            break  # Stop decoding when the end token is encountered
        similarities = F.cosine_similarity(input_vector.unsqueeze(0), amino_acid_tensor)
        most_similar_idx = similarities.argmax().item()
        sequence.append(amino_acid_features.short[amino_acids[most_similar_idx]])
    return ''.join(sequence)


def tensor_to_amino_acids(tensor):
    features = Features()
    mapping = {index: v for index, v in enumerate(features.short.values())}
    sequences = []
    for residue in tensor:
        if torch.all(residue == 0):
            amino_acids = '-'  # This represents a padding or missing value
        else:
            amino_acids = mapping[torch.argmax(residue).item()]
        if amino_acids == '*':  # End token
            break
        sequences.append(amino_acids)

    return ''.join(sequences)


if __name__ == "__main__":
    # Load the .pt file
    file_path = ''

    # Load the .pt file
    data = torch.load(file_path)
    onehot = data['onehot_tensors']
    input_tensor = onehot[0]

    # Output the generated amino acid sequence using the end token '*'
    amino_acid_sequence = tensor_to_amino_acids(input_tensor)

    # Print the generated amino acid sequence
    print("Generated Amino Acid Sequence:", amino_acid_sequence)
