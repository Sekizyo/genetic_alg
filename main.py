import math
import numpy as np
from random import random, choice, randint, uniform
import tkinter as tk
from tkinter import ttk, Tk, Label, Entry

class Individual():
    def __init__(self, id: int) -> None:
        self.id = id               # Unique identifier for the individual
        self.x_real = 0       # The real value of x (within the range [a, b])
        self.x_int = 0
        self.fx = 0                # Evaluation value (calculated using F(x))
        self.gx = 0          # Fitness value
        self.p = 0
        self.q = 0
        self.is_parent = False
        self.x_bin = "" # Binary representation of the integer x
        self.child_bin = ""
        
        self.mutation_points = []
        self.bin_post_mutation = ""
        self.x_real_post_mutation = 0
        self.fx_post_mutation = 0
        
    def set_random_x(self, a: int, b: int, roundTo: float) -> None:
        self.x_real = round(uniform(a, b), roundTo) 
        
    def set_x_int(self, a: int, b: int, l: int) -> None:
        self.x_int = math.ceil((1/(b-a))*(self.x_real - a)*(2**l - 1))
    
    def set_evaluation(self, x_real) -> None:
        self.fx = (x_real % 1) * (math.cos(20 * math.pi * x_real) - math.sin(x_real))
        
    def set_post_mutation_evaluation(self, new_x_real: float) -> None:
        self.fx_post_mutation = (new_x_real % 1) * (math.cos(20 * math.pi * new_x_real) - math.sin(new_x_real))
        
    def set_fitness(self, shift_value: float) -> float:
        self.gx = self.fx + shift_value
            
    def set_x_bin(self, l: int) -> None:
        self.x_bin = format(self.x_int, f'0{l}b')
        
    def set_p(self, total_fitness: float, shift_value: float) -> None:
        self.p = (self.fx+shift_value)/total_fitness
        
    def set_q(self, q: float) -> None:
        self.q = q
        
    def set_is_parent(self, pk: float) -> None:
        if random() <= pk:
            self.is_parent = True
            
    def get_x_real(self, x: int, a: int, b: int, l: int, d: int) -> float:
        return round(((x*(b-a))/(2**l-1))+a, d)
    
    def get_bin_int(self, x: str) -> int:
        return int(x, 2)

    def mutate(self, pm: float, a: int, b: int, l: int, d: float) -> None:
        if self.child_bin:
            genes = list(self.child_bin)
        else:
            genes = list(self.x_bin)
        for i, gene in enumerate(genes):
            if pm >= random():
                self.mutation_points.append(i)
                if gene == "1":
                    genes[i] = "0"
                else:
                    genes[i] = "1"
                    
        self.bin_post_mutation = "".join(genes)
        new_x_int = self.get_bin_int(self.bin_post_mutation)
        self.x_real_post_mutation = self.get_x_real(new_x_int, a, b, l, d)
        self.set_post_mutation_evaluation(self.x_real_post_mutation)
        
    def print_values(self) -> list[int | float | str]:
        return [self.x_real, self.fx, self.gx, self.p, self.q, self.r]
    
class Symulation():
    def __init__(self, a: int, b: int, n: int, d: float, roundTo: int, pk: float, pm: float) -> None:
        self.a = a
        self.b = b
        self.d = d
        self.roundTo = roundTo
        self.population_size = n
        self.binSize = self.get_bin_size(self.a, self.b, d)
        self.pk = pk
        self.pm = pm
    
    def get_bin_size(self, a: int, b: int, d: int) -> int:
        return math.ceil(math.log(((b-a)/d)+1, 2))
        
    def create_population(self) -> list[Individual]:
        population = []
        for i in range(self.population_size):
            individual = Individual(i)
            individual.set_random_x(self.a, self.b, self.roundTo)
            individual.set_x_int(self.a, self.b, self.binSize)
            individual.set_evaluation(individual.x_real)
            individual.set_x_bin(self.binSize)
            population.append(individual)
            
        return population
    
    def selection(self, population: list[Individual]) -> list[Individual]:
        min_fitness = min(individual.fx for individual in population)
        if min_fitness < 0:
            shift_value = abs(min_fitness)
        else:
            shift_value = 0

        total_fitness = sum(individual.fx + shift_value for individual in population)
        
        probabilities = []
        for individual in population:
            individual.set_fitness(shift_value)
            individual.set_p(total_fitness, shift_value)
            probabilities.append(individual.p)
        
        selected_population = np.random.choice(population, size=len(population), p=probabilities, replace=True).tolist()
        return selected_population
    
    def cumulative_distribution(self, population: list[Individual], probabilities: list[float]) -> None:
        cumulative_sum = 0

        for i, q in enumerate(probabilities):
            cumulative_sum += q
            population[i].set_q(q)
     
    def set_parents(self, population: list[Individual]) -> list[Individual]:       
        for individual in population:
            individual.set_is_parent(self.pk)
        return population
            
    def pair_population(self, population: list[Individual]) -> list[Individual]:
        parents = []
        for individual in population:
            if individual.is_parent:
                parents.append(individual)
        
        pairs = [(parents[i], parents[i+1]) for i in range(0, len(parents) - 1, 2)]
        
        if len(parents) % 2 != 0:
            leftover = parents[-1]
            random_partner = choice(parents[:-1])
            pairs.append((leftover, random_partner))
            
        return population, pairs
    
    def mate(self, population: list[Individual], pairs: list[Individual]) -> list[Individual]:
        for parent1, parent2 in pairs:
            crossover_point = randint(1, len(parent1.x_bin)-1)
            parent1.child_bin = parent1.x_bin[:crossover_point] + parent2.x_bin[crossover_point:]
            parent2.child_bin = parent2.x_bin[:crossover_point] + parent1.x_bin[crossover_point:]
            
            # child1 = Individual(parent1.id)
            # child2 = Individual(parent2.id)
            
            # child1.x_bin, child2.x_bin = child1_bin, child2_bin
            
            # child1.x_int = child1.get_bin_int(child1.x_bin)
            # child1.x_real = child1.get_x_real(child1.x_int, self.a, self.b, self.binSize, self.roundTo)
            # child1.set_evaluation()
            
            # child2.x_int = child2.get_bin_int(child2.x_bin)
            # child2.x_real = child2.get_x_real(child2.x_int, self.a, self.b, self.binSize, self.roundTo)
            # child2.set_evaluation()
            
            # child1.set_fitness()
            # child2.set_fitness()
            
            # population[parent1.id] = child1
            # population[parent2.id] = child2
            
        return population
    
    def mutation(self, population: list[Individual]) -> list[Individual]:
        for individual in population:
            individual.mutate(self.pm, self.a, self.b, self.binSize, self.roundTo)
        return population
            
class Window():
    def __init__(self) -> None:
        self.root = Tk()
        self.root.geometry("1000x500")
        self.root.title("Algorytm genetyczny - Matas Pieczulis 21162")

    def get_data(self) -> list[int]:
        _map = {
            "0.1": 1,
            "0.01": 2,
            "0.001": 3,
            "0.0001": 4,
        }
        try:
            a = int(self.a_entry.get())
            b = int(self.b_entry.get())
            n = int(self.n_entry.get())
            d = self.d_box.get()
            pk = float(self.pk_entry.get())
            pm = float(self.pm_entry.get())
        except:
            return (-4, 12, 10, 0.01, 2, 0.001, 0.001)
        return (-4, 12, 10, 0.01, 2, 0.001, 0.001)
        return (a, b, n, d, _map[d], pk, pm)

    def plot_table(self, data: list[int]) -> None:
        columns = ["LP", "x_real", "f(x)", "g(x)", "p", "q", "r", "x_real", "x bin", "parents", "children", "population","mutation point", "mutation bin", "mutation real", "mutation f(x)"]
        
        tree = ttk.Treeview(self.root, columns=columns, show="headings", height=10)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=120)  # Set appropriate width
        
        for row in data:
            tree.insert("", tk.END, values=row)

        tree.grid(row=10, column=0, columnspan=6, padx=10, pady=10)

        vsb = ttk.Scrollbar(self.root, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(self.root, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.grid(row=10, column=6, sticky='ns')
        hsb.grid(row=11, column=0, columnspan=6, sticky='ew')

    def calc(self) -> None:
        a, b, n, d, roundTo, pk, pm = self.get_data()
        self.symulation = Symulation(a, b, n, d, roundTo, pk, pm)
        population = self.symulation.create_population()
        population = self.symulation.selection(population)
        population = self.symulation.mutation(population)
        # self.plot_table(processedData)
        return

    def draw(self) -> None:
        a_label = Label(self.root, text='a:')
        a_label.grid(column=0, row=0)
        self.a_entry = Entry(self.root)
        self.a_entry.grid(column=1, row=0)

        b_label = Label(self.root, text='b:')
        b_label.grid(column=3, row=0)
        self.b_entry = Entry(self.root)
        self.b_entry.grid(column=4, row=0)

        n_label = Label(self.root, text='N:')
        n_label.grid(column=0, row=1)
        self.n_entry = Entry(self.root)
        self.n_entry.grid(column=1, row=1)

        d_label = Label(self.root, text='d:')
        d_label.grid(column=3, row=1)
        self.d_box = ttk.Combobox(self.root, values=["0.1", "0.01", "0.001", "0.0001"])
        self.d_box.grid(column=4, row=1)
        
        pk_label = Label(self.root, text='pk:')
        pk_label.grid(column=0, row=2)
        self.pk_entry = Entry(self.root)
        self.pk_entry.grid(column=1, row=2)

        pm_label = Label(self.root, text='pm:')
        pm_label.grid(column=3, row=2)
        self.pm_entry = Entry(self.root)
        self.pm_entry.grid(column=4, row=2)

        calc_button = ttk.Button(self.root, text="Calculate", command=self.calc)
        calc_button.grid(column=2, row=8)
        self.root.mainloop()

if __name__ == "__main__":
    # win = Window()
    # win.draw()
    symulation = Symulation(-4, 12, 10, 0.01, 2, 0.5, 0.001)
    population = symulation.create_population()
    for inv in population:
        print(vars(inv))
    population = symulation.selection(population)
    population = symulation.set_parents(population)
    
    population, pairs = symulation.pair_population(population)
    population = symulation.mate(population, pairs)
    population = symulation.mutation(population)
    for inv in population:
        print(vars(inv))