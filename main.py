import math
import numpy as np
from random import random, uniform
import tkinter as tk
from tkinter import ttk, Tk, Label, Entry

class Individual():
    def __init__(self, id: int) -> None:
        self.id = id               # Unique identifier for the individual
        self.x_real = 0       # The real value of x (within the range [a, b])
        self.x_int = 0
        self.fx = 0                # Fitness value (calculated using F(x))
        self.x_bin = ""           # Binary representation of the integer x
        self.new_x = 0             # New value of x after selection, crossover, and mutation
        
    def set_random_x(self, a: int, b: int, roundTo: float) -> None:
        self.x_real = round(uniform(a, b), roundTo) 
        
    def set_x_int(self, a: int, b: int, l: int) -> int:
        self.x_int = math.ceil((1/(b-a))*(self.x_real - a)*(2**l - 1))
    
    def set_fitness(self) -> float:
        self.fx = (self.x_real % 1) * (math.cos(20 * math.pi * self.x_real) - math.sin(self.x_real))
        
    def set_x_bin(self, l: int):
        self.x_bin = format(self.x_int, f'0{l}b')
        
    def get_x_real(self, x: float, a: int, b: int, l: int, d: int) -> float:
        return round(((x*(b-a))/(2**l-1))+a, d)
    
    def get_bin_int(self, x: float) -> int:
        return int(x, 2)

    def mutate(self, p: float) -> int:
        mutated_count = 0
        genes = list(self.x_bin)
        for i, gene in enumerate(genes):
            if p >= random():
                mutated_count += 1
                if gene == "1":
                    genes[i] = "0"
                else:
                    genes[i] = "1"
                    
        self.x_bin = "".join(genes)
        
        return mutated_count
    
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
        self.p = pk
        self.q = 0
        self.pm = pm
    
    def get_bin_size(self, a: int, b: int, d: int) -> int:
        return math.ceil(math.log(((b-a)/d)+1, 2))
        
    def create_population(self) -> list[Individual]:
        population = []
        for i in range(self.population_size):
            individual = Individual(i)
            individual.set_random_x(self.a, self.b, self.roundTo)
            individual.set_x_int(self.a, self.b, self.binSize)
            individual.set_x_bin(self.binSize)
            individual.set_fitness()
            print(vars(individual))
            
            population.append(individual)
            
        return population
    
    def selection(self, population: list[Individual]) -> list[Individual]:
        min_fitness = min(individual.fx for individual in population)
        if min_fitness < 0:
            shift_value = abs(min_fitness)
        else:
            shift_value = 0

        total_fitness = sum(individual.fx + shift_value for individual in population)

        probabilities = [(individual.fx + shift_value) / total_fitness for individual in population]

        selected_population = np.random.choice(population, size=len(population), p=probabilities, replace=True)
        return selected_population
    
    def mutation(self, population: list[Individual]) -> list[Individual]:
        mutations = []
        for i, individual in enumerate(population):
            mutation = individual.mutate(self.p)
            if mutation > 0:
                mutations.append((i, mutation))
                
        return population, mutations
            
            
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
    symulation = Symulation(-4, 12, 10, 0.01, 2, 0.001, 0.5)
    population = symulation.create_population()
    population = symulation.selection(population)
    population, count = symulation.mutation(population)