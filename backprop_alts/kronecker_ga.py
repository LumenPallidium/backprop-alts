import torch
import numpy as np
from copy import deepcopy

class KroneckerGATorch(torch.nn.Module):
    def __init__(self,
                 evaluation_function,
                 minimize = True,
                 out_size = 8,
                 pop_size = 100):
        super().__init__()
        self.out_size = out_size
        self.minimize = minimize
        self.pop_size = pop_size
        self.codon_size = int(np.log(self.out_size) / np.log(2))

        self.fitness = evaluation_function

        building_blocks = self.generate_building_block_tensor()
        self.register_buffer("building_blocks", building_blocks)
        self.n_blocks = building_blocks.shape[0]

    def generate_building_block_tensor():
        """
        This function generates a 36x2x2 tensor of building blocks. Each 2x2 
        matrix satisfies the following conditions:

        1. The allowed element values are 0, 1, -1
        2. Diagonal elements can take any of the allowed values
        3. Upper-triangular elements can only be 0 or 1
        4. Lower-triangular elements can only be 0 or -1

        In total there are 3x3x2x2 = 36 possible building blocks.
        """
        building_blocks = []
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    for l in range(2):
                        block = torch.zeros(2, 2)
                        block[0, 0] = i - 1
                        block[1, 1] = j - 1
                        block[0, 1] = k
                        block[1, 0] = l - 1
                        building_blocks.append(block)
        return torch.stack(building_blocks)

    def geno2pheno(self, weights, indices):
        matrix = torch.zeros(self.out_size, self.out_size)
        for weight, index in zip(weights, indices):
            blocks = self.building_blocks.index_select(0, index)
            kron_prod = torch.ones(1, 1)
            for block in blocks:
                kron_prod = torch.kron(kron_prod, block)
            matrix += weight * kron_prod
        return matrix
    
    def generate_individual(self):
        weights = torch.randn(self.pop_size)
        indices = torch.randint(self.n_blocks,
                                size = (self.pop_size, self.codon_size))
        return weights, indices
        


BASES = {
    "1" : torch.tensor([[1., 0.], [0., 1.]]),
    "2" : torch.tensor([[1., 1.], [1., 1.]]),
    "3" : torch.tensor([[0., -1.], [-1., 0.]]),
    "4" : torch.tensor([[-1., 0.], [0., -1.]]),
    "5" : torch.tensor([[0., 1.], [1., 0.]]),
    "6" : torch.tensor([[1., 0.], [1., 0.]]),
}
# generate a matrix of transition probs based on distance between codons
def generate_transition_matrix(bases = BASES):
    n_bases = len(bases)
    transition_matrix = torch.zeros((n_bases, n_bases))
    for i, base1 in enumerate(bases.values()):
        for j, base2 in enumerate(bases.values()):
            if i == j:
                transition_matrix[i, j] = 0
            else:
                transition_matrix[i, j] = 1 / torch.sum(torch.abs(base1 - base2))
    transition_matrix = transition_matrix / torch.sum(transition_matrix, dim = 1, keepdim = True)
    return transition_matrix

#TODO : add genome cleaning to remove redundant codons
class KroneckerGA:
    def __init__(self,
                 evaluation_function,
                 minimize = True,
                 max_codon_size = 20,
                 out_size = 8,
                 pop_size = 100,
                 n_elites = 10,
                 bases = BASES):
        
        if isinstance(out_size, int):
            out_size = (out_size, out_size)
        self.out_size = out_size
        self.minimize = minimize
        self.max_codon_size = max_codon_size

        self.bases = bases
        self.base_names = list(self.bases.keys())
        self.base_size = list(self.bases.values())[0].shape
        self.codon_size = self._init_codon_size()
        self.transition_matrix = generate_transition_matrix(bases = self.bases)

        self.pop_size = pop_size
        self.n_elites = n_elites
        self.population = [self.generate_individual() for _ in range(pop_size)]
        self.best = None

        self.fitness = evaluation_function

    def _init_codon_size(self):
        """
        Here, a codon means the length of the gene specifying the matrices
        comprosing the output matrix. So if the output is 8x8 and the bases
        are 2x2, then the codon size is log2(8) = 3.

        Currently only supports square matrices.
        """
        return int(np.log(self.out_size[0]) / np.log(self.base_size[0]))

    def generate_codons(self, gene, length_decay = 0.5):
        # continue adding codons until the length decay is reached
        roll = np.random.rand()
        while (roll > length_decay) & (len(gene) < self.max_codon_size):
            codon = np.random.choice(self.base_names, size = self.codon_size)
            gene.append([np.random.randn(), codon])
            roll = np.random.rand()
        return gene

    def generate_individual(self, length_decay = 0.5):
        first_codon = np.random.choice(self.base_names, size = self.codon_size)
        gene = [[np.random.randn(), first_codon]]

        gene = self.generate_codons(gene, length_decay = length_decay)

        return gene
    
    def geno2pheno(self, individual):
        """
        Converts the gene to the phenotype (the matrix).
        """
        matrix = torch.zeros(self.out_size)
        for weight, codon in individual:
            for i, base in enumerate(codon):
                if i == 0:
                    kron_prod = self.bases[base]
                else:
                    kron_prod = torch.kron(kron_prod, self.bases[base])
            matrix += weight * kron_prod
        return matrix

    def mutate(self,
               individual_in,
               weight_scale = 1,
               new_individual_rate = 0.2,
               point_mutation_rate = 0.5, 
               codon_mutation_rate = 0.1,
               transposition_rate = 0.05,
               addition_rate = 0.3,
               removal_rate = 0.2,
               crossover_rate = 0.1):
        """
        Applies mutations. Weight scale determines the scale of the gaussian
        noise added to the weights. Point mutation rate determines the rate
        of mutations of single symbols in the codons. Codon mutation rate
        determines the rate that codons are replace. Transposition rate
        determines how frequently weights are shifted to other codons.
        """
        if np.random.rand() < new_individual_rate:
            return self.generate_individual()
        else:
            individual = deepcopy(individual_in)
            for i, (weight, codon) in enumerate(individual):
                weight += weight_scale * np.random.randn()
                # mutate the weight
                roll = np.random.rand()
                if roll < point_mutation_rate:
                    old_codon = codon
                    j = np.random.randint(self.codon_size)
                    probs = self.transition_matrix[self.base_names.index(old_codon[j])]
                    new_base = np.random.choice(self.base_names, p = np.array(probs))
                    new_codon = old_codon.copy()
                    new_codon[j] = new_base
                    individual[i] = [weight, new_codon]

                # mutate the codon
                roll = np.random.rand()
                if roll < codon_mutation_rate:
                    codon = np.random.choice(self.base_names, size = self.codon_size)
                    individual[i] = [weight, codon]

            # transposition
            roll = np.random.rand()
            if roll < transposition_rate:
                i = np.random.randint(len(individual))
                j = np.random.randint(len(individual))
                # flip the weights
                individual[i][0] = individual[j][0]
                individual[j][0] = individual[i][0]

            # addition
            individual = self.generate_codons(individual,
                                              length_decay = addition_rate)

            # removal
            if len(individual) > 1:
                roll = np.random.rand()
                if roll < removal_rate:
                    i = np.random.randint(len(individual))
                    individual.pop(i)

            # crossover
            roll = np.random.rand()
            if roll < crossover_rate:
                other_individual = self.population[np.random.randint(self.pop_size)]
                individual = self.crossover(individual, other_individual)

            return individual

    #TODO : make this smarter - average shared codons, etc.
    def crossover(self, individual1, individual2):
        new_genome = []
        len1, len2 = len(individual1), len(individual2)
        shorter_genome = individual1 if len1 < len2 else individual2
        longer_genome = individual1 if len1 >= len2 else individual2
        min_len = min(len1, len2)

        for i, (weight, codon) in enumerate(longer_genome):
            if (i < min_len) & (np.random.rand() < 0.5):
                new_genome.append(shorter_genome[i])
            else:
                new_genome.append(longer_genome[i])
        return new_genome
    
    def select(self, population):
        fitnesses = [self.fitness(self.geno2pheno(individual)) for individual in population]
        fitnesses = np.array(fitnesses)

        if self.minimize:
            fit = np.min(fitnesses)
            selected_indices = np.argsort(fitnesses)[:self.n_elites]
        else:
            fit = np.max(fitnesses)
            selected_indices = np.argsort(fitnesses)[-self.n_elites:]
        self.best = population[selected_indices[0]]
        return [population[i] for i in selected_indices], fit

    def train_step(self):
        population, best_error = self.select(self.population)
        while len(population) < self.pop_size:
            individual = self.mutate(population[np.random.randint(self.n_elites)])
            population.append(individual)
        self.population = population
        return best_error

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    n_epochs = 10
    steps_per_epoch = 300

    x = torch.randn(100, 8)
    M = torch.randn(8, 8)
    GOAL = x @ M + torch.randn(100, 8) * 0.05

    def evaluation(individual, goal = GOAL):
        M = torch.Tensor(individual)
        y = x @ M
        error = torch.mean(torch.abs(y - goal))
        return error
    
    ga = KroneckerGA(evaluation_function = evaluation)

    errors = []
    for epoch in range(n_epochs):
        for step in tqdm(range(steps_per_epoch)):
            error = ga.train_step()
            errors.append(error)
        print(f"Epoch {epoch} : {error}")

    plt.plot(errors)