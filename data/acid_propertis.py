import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch
# =============================================================================
class Features():
    def __init__(self):
        # ==================================================================== #
        #                               Amino acid parameters                  #
        # ==================================================================== #
        # Steric parameter = stc |
        # Helix prob = alpha | Sheet prob = beta |
        # Hydrophobicity = H_1 | Hydrophilicity = H_2 |
        # Polarity = P_1 | Polarizability = P_2 | isoelectric_PH = P_i |
        # side chain net charge number = NCN | solvent accessible surface area = SASA |
        # A1 = Accessibility | A2 = Antegenic | T = Turns | E = Exposed | F = Flexibility |
        self.Features = {
        'ALA':{'stc':1.28,'P_1':8.1, 'P_2':0.046,'vol':1.00,'H_1':0.62, 'H_2':-0.5,'P_i': 6.11,'alpha':0.42,'beta':0.23,'NCN':0.007187,'SASA':1.181, 'A1': 0.49,'A2': 1.064,'T':-0.8,  'E': 15,'F': -1.27,'P12':-0.8,  'H12':2.1},
        'CYS':{'stc':1.77,'P_1':5.50,'P_2':0.128,'vol':2.43,'H_1':0.29, 'H_2':-1.0,'P_i': 6.35,'alpha':0.17,'beta':0.41,'NCN':-0.03661,'SASA':1.461, 'A1': 0.26,'A2': 1.412,'T':0.83,  'E': 5, 'F': -1.09,'P12':0.83,  'H12':1.4},
        'ASP':{'stc':1.60,'P_1':13.0,'P_2':0.105,'vol':2.78,'H_1':-0.9, 'H_2':3.0, 'P_i': 2.95,'alpha':0.25,'beta':0.20,'NCN':-0.02382,'SASA':1.587, 'A1': 0.78,'A2': 0.866,'T':1.65,  'E': 50,'F': 1.42, 'P12':1.65,   'H12':10},
        'GLU':{'stc':1.56,'P_1':12.3,'P_2':0.151,'vol':3.78,'H_1':-0.74,'H_2':3.0, 'P_i': 3.09,'alpha':0.42,'beta':0.21,'NCN':0.006802,'SASA':1.862, 'A1': 0.84,'A2': 0.851,'T':-0.92, 'E': 55,'F': 1.6,  'P12':-0.92, 'H12':7.8},
        'PHE':{'stc':2.94,'P_1':5.20,'P_2':0.29, 'vol':5.89,'H_1':1.19, 'H_2':-2.5,'P_i': 5.67,'alpha':0.30,'beta':0.38,'NCN':0.037552,'SASA':2.228, 'A1': 0.42,'A2': 1.091,'T':0.18,  'E': 10,'F': -2.14,'P12':0.18, 'H12':-9.2},
        'GLY':{'stc':0.00,'P_1':9.0, 'P_2':0.00, 'vol':0.00,'H_1':0.48, 'H_2':0.0, 'P_i': 6.07,'alpha':0.13,'beta':0.15,'NCN':0.179052,'SASA':0.881, 'A1': 0.48,'A2': 0.874,'T':-0.55, 'E': 10,'F': 1.86, 'P12':-0.55, 'H12':5.7},
        'HIS':{'stc':2.99,'P_1':20.4,'P_2':0.23, 'vol':4.66,'H_1':-0.4, 'H_2':-0.5,'P_i': 7.69,'alpha':0.27,'beta':0.30,'NCN':-0.01069,'SASA':2.025, 'A1': 0.84,'A2': 1.105,'T':0.11,  'E': 56,'F': -0.82,'P12':0.11,  'H12':2.1},
        'ILE':{'stc':4.19,'P_1':5.20,'P_2':0.186,'vol':4.00,'H_1':1.38, 'H_2':-1.8,'P_i': 6.04,'alpha':0.30,'beta':0.45,'NCN':0.021631,'SASA':1.810, 'A1': 0.34,'A2': 1.152,'T':-1.53, 'E': 13,'F': -2.89,'P12':-1.53,  'H12':-8},
        'LYS':{'stc':1.89,'P_1':11.3,'P_2':0.219,'vol':4.77,'H_1':-1.5, 'H_2':3.0, 'P_i': 9.99,'alpha':0.32,'beta':0.27,'NCN':0.017708,'SASA':2.258, 'A1': 0.97,'A2': 0.930,'T':-1.06, 'E': 85,'F': 2.88, 'P12':-1.06, 'H12':5.7},
        'LEU':{'stc':2.59,'P_1':4.90,'P_2':0.186,'vol':4.00,'H_1':1.06, 'H_2':-1.8,'P_i': 6.04,'alpha':0.39,'beta':0.31,'NCN':0.051672,'SASA':1.931, 'A1': 0.40,'A2': 1.250,'T':-1.01, 'E': 16,'F': -2.29,'P12':-1.01,'H12':-9.2},
        'MET':{'stc':2.35,'P_1':5.70,'P_2':0.221,'vol':4.43,'H_1': 0.64,'H_2':-1.3,'P_i': 5.71,'alpha':0.38,'beta':0.32,'NCN':0.002683,'SASA':2.034, 'A1': 0.48,'A2': 0.826,'T':-1.48, 'E': 20,'F': -1.84,'P12':-1.48,'H12':-4.2},
        'ASN':{'stc':1.60,'P_1':11.6,'P_2':0.134,'vol':2.95,'H_1':-0.78,'H_2':2.0, 'P_i': 6.52,'alpha':0.21,'beta':0.22,'NCN':0.005392,'SASA':1.655, 'A1': 0.81,'A2': 0.776,'T':3.0,   'E': 49,'F': 1.77, 'P12':3.0,   'H12':7.0},
        'PRO':{'stc':2.67,'P_1':8.0, 'P_2':0.131,'vol':2.72,'H_1': 0.12,'H_2':0.0, 'P_i': 6.80,'alpha':0.13,'beta':0.34,'NCN':0.239530,'SASA':1.468, 'A1': 0.49,'A2': 1.064,'T':-0.8,  'E': 15,'F': 0.52, 'P12':-0.8,  'H12':2.1},
        'GLN':{'stc':1.56,'P_1':10.5,'P_2':0.180,'vol':3.95,'H_1':-0.85,'H_2':0.2, 'P_i': 5.65,'alpha':0.36,'beta':0.25,'NCN':0.049211,'SASA':1.932, 'A1': 0.84,'A2': 1.015,'T':0.11,  'E': 56,'F': 1.18, 'P12':0.11,  'H12':6.0},
        'ARG':{'stc':2.34,'P_1':10.5,'P_2':0.291,'vol':6.13,'H_1':-2.53,'H_2':3.0, 'P_i':10.74,'alpha':0.36,'beta':0.25,'NCN':0.043587,'SASA':2.560, 'A1': 0.95,'A2': 0.873,'T':-1.15, 'E': 67,'F': 2.79, 'P12':-1.15, 'H12':4.2},
        'SER':{'stc':1.31,'P_1':9.20,'P_2':0.062,'vol':1.60,'H_1':-0.18,'H_2':0.3, 'P_i': 5.70,'alpha':0.20,'beta':0.28,'NCN':0.004627,'SASA':1.298, 'A1': 0.65,'A2': 1.012,'T':1.34,  'E': 32,'F': 3.0,  'P12':1.34,  'H12':6.5},
        'THR':{'stc':3.03,'P_1':8.60,'P_2':0.108,'vol':2.60,'H_1':-0.05,'H_2':-0.4,'P_i': 5.60,'alpha':0.21,'beta':0.36,'NCN':0.003352,'SASA':1.525, 'A1': 0.70,'A2': 0.909,'T':0.27,  'E': 32,'F': 1.18, 'P12':0.27,  'H12':5.2},
        'VAL':{'stc':3.67,'P_1':5.90,'P_2':0.140,'vol':3.00,'H_1':1.08, 'H_2':-1.5,'P_i': 6.02,'alpha':0.27,'beta':0.49,'NCN':0.057004,'SASA':1.645, 'A1': 0.36,'A2': 1.383,'T':-0.83, 'E': 14,'F': -1.75,'P12':-0.83,'H12':-3.7},
        'TRP':{'stc':3.21,'P_1':5.40,'P_2':0.409,'vol':8.08,'H_1':0.81, 'H_2':-3.4,'P_i': 5.94,'alpha':0.32,'beta':0.42,'NCN':0.037977,'SASA':2.663, 'A1': 0.51,'A2': 0.893,'T':-0.97, 'E': 17,'F': -3.78,'P12':-0.97, 'H12':-10},
        'TYR':{'stc':2.94,'P_1':6.20,'P_2':0.298,'vol':6.47,'H_1':0.26, 'H_2':-2.3,'P_i': 5.66,'alpha':0.25,'beta':0.41,'NCN':0.023599,'SASA':2.368, 'A1': 0.76,'A2': 1.161,'T':-0.29, 'E': 41,'F': -3.3, 'P12':-0.29,'H12':-1.9},
        }
        # ------------------------------------- #
        self.AAs = list(self.Features.keys())
        self.AA_prop_keys = [list(Values.keys()) for Keys,Values in self.Features.items()][0]
        # ------------------------------------- #
        self.Props = {}
        self.X_Props = {}
        self.Max_Props = {}
        self.Min_Props = {}
        for prop in self.AA_prop_keys:
            props = []
            for AA in self.AAs:
                props.append(self.Features[AA][prop])
            self.X_Props[prop] = np.median(props)
            self.Min_Props[prop] = min(props)
            self.Max_Props[prop] = max(props)
            props.append(np.median(props))
            self.Props[prop] = props
        # ------------------------------------- #
        self.AA_dict = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,
                        'K':8,'L':9,'M':10, 'N':11,'P':12,'Q':13,'R':14,
                        'S':15,'T':16,'V':17,'W':18,'Y':19,'X':20,'Z':21,'B':22,'J':23,'U':24,'O':25}
        # ------------------------------------- #
        self.Amino_acids = {
                    'A':self.Features['ALA'],'C':self.Features['CYS'],
                    'D':self.Features['ASP'],'E':self.Features['GLU'],
                    'F':self.Features['PHE'],'G':self.Features['GLY'],
                    'H':self.Features['HIS'],'I':self.Features['ILE'],
                    'K':self.Features['LYS'],'L':self.Features['LEU'],
                    'M':self.Features['MET'],'N':self.Features['ASN'],
                    'P':self.Features['PRO'],'Q':self.Features['GLN'],
                    'R':self.Features['ARG'],'S':self.Features['SER'],
                    'T':self.Features['THR'],'V':self.Features['VAL'],
                    'W':self.Features['TRP'],'Y':self.Features['TYR'],
                    'X': self.X_Props,
                    'Z': random.choice([self.Features['GLU'], self.Features['GLN']]),
                    'B': random.choice([self.Features['ASP'], self.Features['ASN']]),
                    'J': random.choice([self.Features['LEU'], self.Features['ILE']]),
                    'U':self.X_Props,
                    'O':self.X_Props,
                    }
        # ------------------------------------- #
        self.short = {
        'ALA': 'A', 'CYS': 'C',
        'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G',
        'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V',
        'TRP': 'W', 'TYR': 'Y',
        'END': '*',
        
    }

    def normalize_features(self, scale_factor=1):
        normalized_features = {}
        for aa in self.AAs:
            normalized_features[aa] = {}
            for prop in self.AA_prop_keys:
                normalized_value = scale_factor * (self.Features[aa][prop] - self.Min_Props[prop]) / (
                        self.Max_Props[prop] - self.Min_Props[prop])
                normalized_features[aa][prop] = normalized_value
        return normalized_features

    def generate_random_sequence_tensor(self, min_length, max_length):
        
        normalized_features = self.normalize_features()


        sequence_length = random.randint(min_length, max_length)

        
        sequence = []

        for _ in range(sequence_length):
            random_aa = random.choice(list(normalized_features.keys())) 
            
            value_vector = [normalized_features[random_aa][prop] for prop in self.AA_prop_keys]
            sequence.append(value_vector)

        
        padding_length = max_length - sequence_length
        if padding_length > 0:
            padding_vector = [0.0] * len(self.AA_prop_keys)
            sequence.extend([padding_vector] * padding_length)

       
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)

        return sequence_tensor

    def sample_sequence_tensor(self, min_length, max_length):
        """
        Generate a random sequence tensor based on empirical amino acid distribution.

        Args:
            min_length (int): Minimum length of the sequence.
            max_length (int): Maximum length of the sequence.

        Returns:
            torch.Tensor: A tensor representation of the sequence.
        """
        
        normalized_features = self.normalize_features()

        # 使用默认的氨基酸分布
        aa_distribution = {
            'ALA': 0.0809, 'CYS': 0.0202, 'ASP': 0.0464, 'GLU': 0.0653, 'PHE': 0.0362,
            'GLY': 0.0669, 'HIS': 0.0205, 'ILE': 0.0612, 'LYS': 0.0806, 'LEU': 0.0891,
            'MET': 0.0292, 'ASN': 0.0388, 'PRO': 0.0381, 'GLN': 0.0367, 'ARG': 0.0686,
            'SER': 0.0596, 'THR': 0.0519, 'VAL': 0.0739, 'TRP': 0.0087, 'TYR': 0.0270
        }

        
        aa_keys = list(aa_distribution.keys())
        probabilities = np.array([aa_distribution[aa] for aa in aa_keys])
        probabilities /= probabilities.sum()  # 归一化

        
        sequence_length = random.randint(min_length, max_length)

       
        sequence = []

        for _ in range(sequence_length):
           
            random_aa = np.random.choice(aa_keys, p=probabilities)
            
            value_vector = [normalized_features[random_aa][prop] for prop in self.AA_prop_keys]
            sequence.append(value_vector)

       
        padding_length = max_length - sequence_length
        if padding_length > 0:
            padding_vector = [0.0] * len(self.AA_prop_keys)
            sequence.extend([padding_vector] * padding_length)

        
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)

        return sequence_tensor

