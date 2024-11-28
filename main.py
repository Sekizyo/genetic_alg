#Matas Pieczulis ID05IO1 21162
import math
import numpy as np
import itertools
from random import random, choice, randint, uniform
import tkinter as tk
from tkinter import ttk, Tk, Label, Entry, messagebox, BooleanVar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 
from concurrent.futures import ThreadPoolExecutor, as_completed

class Individual():
    def __init__(self, id: int) -> None:
        self.id = id
        self.x_real = 0
        self.x_int = 0
        self.fx = 0 
        self.gx = 0
        self.parent_p = 0
        self.p = 0
        self.q = 0
        self.r = 0
        self.r2 = 0
        self.x_sel = 0
        self.x_bin = ""
        self.parent2 = "-"
        self.child_bin = "-"
        
        self.is_parent = "-"
        self.crossover_point = "-"
        
        self.mutation_points = []
        self.new_gen = ""
        self.bin_after_mutation = "-"
        self.x_real_after_mutation = "-"
        self.fx_after_mutation = "-"
        self.gx_after_mutation = ""
        
    def set_random_x(self, a: int, b: int, roundTo: float) -> None:
        self.x_real = round(uniform(a, b), roundTo) 
        
    def set_x_int(self, x_real: float,  a: int, b: int, l: int) -> None:
        self.x_int = math.ceil((1/(b-a))*(x_real - a)*(2**l - 1))
    
    def set_evaluation(self, x_real) -> None:
        self.fx = (x_real % 1) * (math.cos(20 * math.pi * x_real) - math.sin(x_real))
        
    def set_after_mutation_evaluation(self, new_x_real: float) -> None:
        self.fx_after_mutation = (new_x_real % 1) * (math.cos(20 * math.pi * new_x_real) - math.sin(new_x_real))
        
    def set_fitness(self, minF: float, d: float) -> float:
        self.gx = self.fx - minF + d 
        
    def set_fitness_after_mutation(self, minF: float, d: float) -> float:
        self.gx_after_mutation = self.fx_after_mutation - minF + d 
            
    def set_x_bin(self, l: int) -> None:
        self.x_bin = format(self.x_int, f'0{l}b')
        
    def set_p(self, total_fitness: float, shift_value: float) -> None:
        self.p = (self.fx+shift_value)/total_fitness
        
    def set_q(self, q: float) -> None:
        self.q = q
        
    def set_r(self, r: float) -> None:
        self.r = r
            
    def set_crossover_point(self, point: int) -> None:
        self.crossover_point = point
        
    def set_is_parent(self, pk: float) -> None:
        if self.r2 <= pk:
            self.is_parent = True
        else:
            self.is_parent = False
            
    def set_is_selected(self) -> None:
        self.is_selected = True
        
    def set_crossover_point(self, point: int) -> None:
        if self.is_parent:
            self.crossover_point = point
        else:
            self.crossover_point = "-"
            
    def set_parent2(self, parent2: str) -> None:
        self.parent2 = parent2
            
    def set_child_bin(self, bin: str) -> None:
        self.child_bin = bin
            
    def get_x_real(self, x: int, a: int, b: int, l: int, d: int) -> float:
        return round(((x*(b-a))/(2**l-1))+a, d)
    
    def get_bin_int(self, x: str) -> int:
        return int(x, 2)

    def mutate(self, pm: float, a: int, b: int, l: int, d: float) -> None:
        if self.is_parent:
            genes = list(self.child_bin)
            self.new_gen = self.child_bin
        else:
            genes = list(self.x_bin)
            self.new_gen = self.x_bin
        
        for i, gene in enumerate(genes):
            if pm >= random():
                self.mutation_points.append(i)
                if gene == "1":
                    genes[i] = "0"
                else:
                    genes[i] = "1"
                    
        genes = "".join(genes)
        if genes != "-":
            self.bin_after_mutation = genes
            new_x_int = self.get_bin_int(self.bin_after_mutation)
            self.x_real_after_mutation = self.get_x_real(new_x_int, a, b, l, d)
            self.set_after_mutation_evaluation(self.x_real_after_mutation)
        else:
            self.bin_after_mutation = self.x_bin
            new_x_int = self.get_bin_int(self.bin_after_mutation)
            self.x_real_after_mutation = self.get_x_real(new_x_int, a, b, l, d)
            self.set_after_mutation_evaluation(self.x_real_after_mutation)
            
    def print_values(self, d: int) -> list[str]:
        if self.mutation_points:
            points = self.mutation_points
        else:
            points = "-"
        return [self.id, self.x_real, self.fx, round(self.gx, d), round(self.p, d), round(self.q, d), self.r, self.x_sel, self.x_bin, self.r2, self.parent2, self.crossover_point, self.child_bin, self.new_gen, points, self.bin_after_mutation, self.x_real_after_mutation, self.fx_after_mutation]          
    
class Symulation():
    def __init__(self, a: int, b: int, n: int, d: float, roundTo: int, pk: float, pm: float, is_elite: bool) -> None:
        self.a = a
        self.b = b
        self.d = d
        self.roundTo = roundTo
        self.population_size = n
        self.binSize = self.get_bin_size()
        self.pk = pk
        self.pm = pm
        self.is_elite = is_elite
    
    def get_bin_size(self) -> int:
        return math.ceil(math.log(((self.b-self.a)/self.d)+1, 2))
        
    def set_elite(self, elite: Individual) -> Individual:
        elite.x_real_after_mutation = elite.x_real
        elite.fx_after_mutation = elite.fx
        elite.gx_after_mutation = elite.gx
        return elite
    
    def create_population(self) -> list[Individual]:
        population = []
        for i in range(self.population_size):
            individual = Individual(i)
            individual.set_random_x(self.a, self.b, self.roundTo)
            individual.set_x_int(individual.x_real, self.a, self.b, self.binSize)
            individual.set_evaluation(individual.x_real)
            individual.set_x_bin(self.binSize)
            population.append(individual)
            
        return population
    
    def new_population(self, population: list[Individual]) -> list[Individual]:
        if not population:
            population = self.create_population()
            population = self.selection(population)
            population, pairs = self.pair_population(population)
            population = self.mate(population, pairs)
            population = self.mutation(population)
            
        new_pop = []
        for individual in population:
            individual2 = Individual(individual.id)
            individual2.x_real = individual.x_real_after_mutation
            individual2.set_x_int(individual2.x_real, self.a, self.b, self.binSize)
            individual2.set_evaluation(individual2.x_real)
            individual2.set_x_bin(self.binSize)
            new_pop.append(individual2)
                
        return new_pop
    
    def evaluate(self, population: list[Individual]) -> list[Individual]:
        min_fitness = min(individual.fx_after_mutation for individual in population)
        for individual in population:
            individual.set_fitness_after_mutation(min_fitness, self.d)
            
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
            individual.set_fitness(min_fitness, self.d)
            individual.set_p(total_fitness, shift_value)
            probabilities.append(individual.p)
        
        self.cumulative_distribution(population, probabilities)
        x_values = []
        r_values = []
        q_values = []
        for individual in population:
            individual.r = round(uniform(0, 1), self.roundTo)
            individual.r2 = round(uniform(0, 1), self.roundTo)
            x_values.append(individual.x_real)
            r_values.append(individual.r)
            q_values.append(individual.q)
            
        for x, individual in enumerate(population):
            index = 0
            for i in range(1, len(q_values)):
                if q_values[i - 1] <= r_values[x] <= q_values[i]:
                    index = i
                    break
            individual.x_sel = x_values[index]
            individual.set_x_int(individual.x_sel, self.a, self.b, self.binSize)
            individual.set_evaluation(individual.x_sel)
            individual.set_x_bin(self.binSize)
            individual.set_is_parent(self.pk)
        
        return population
    
    def cumulative_distribution(self, population: list[Individual], probabilities: list[float]) -> None:
        cumulative_sum = 0

        for i, q in enumerate(probabilities):
            cumulative_sum += q
            population[i].set_q(cumulative_sum)
     
    def pair_population(self, population: list[Individual]) -> list[Individual]:
        parents = []
        for individual in population:
            if individual.is_parent:
                parents.append(individual)
        
        pairs = [(parents[i], parents[i+1]) for i in range(0, len(parents) - 1, 2)]
        if len(parents) % 2 != 0 and pairs:
            leftover = parents.pop(-1)
            random_partner = choice(pairs)
            pairs.append((leftover, random_partner[0]))
        return population, pairs
    
    def mate(self, population: list[Individual], pairs: list[Individual]) -> list[Individual]:
        for parent1, parent2 in pairs:
            crossover_point = randint(1, len(parent1.x_bin)-1)
            parent1.set_crossover_point(crossover_point)
            parent2.set_crossover_point(crossover_point)
            
            parent1.set_parent2(parent2.x_bin)
            parent2.set_parent2(parent1.x_bin)
            
            parent1.child_bin = parent1.x_bin[:crossover_point] + parent2.x_bin[crossover_point:]
            parent2.child_bin = parent2.x_bin[:crossover_point] + parent1.x_bin[crossover_point:]
            
        return population
    
    def mutation(self, population: list[Individual]) -> list[Individual]:
        for individual in population:
            individual.mutate(self.pm, self.a, self.b, self.binSize, self.roundTo)
        return population
    
    def run(self, t: int) -> list:
        output = []
        population = []
        elite = None
        for iter in range(t):
            population = self.new_population(population)
            if self.is_elite:
                population.sort(key=lambda individual: individual.fx, reverse=True)
                elite = population.pop(0)
                
            population = self.selection(population)
            population, pairs = self.pair_population(population)
            population = self.mate(population, pairs)
            population = self.mutation(population)
            
            if self.is_elite:
                elite = self.set_elite(elite)
                population.insert(int(uniform(0, len(population))), elite)
                
            population = self.evaluate(population)
                
            output.append(self.output(iter, population))
            
        return output
    
    def output(self, interation: int,  population: list[Individual]) -> list[Individual]:
        min_ = min(individual.fx_after_mutation for individual in population)
        avg_ = sum(individual.fx_after_mutation for individual in population)/len(population)
        max_ = max(individual.fx_after_mutation for individual in population)
        max_index = max(range(len(population)), key=lambda i: population[i].fx_after_mutation)
        bestX = population[max_index].x_real_after_mutation

        return (interation, min_, avg_, max_, bestX)
 
class Tests():
    def __init__(self, a: int, b: int, d: float, roundTo: int, is_elite) -> None:
        self.iterations = 10
        self.a = a
        self.b = b
        self.d = d
        self.roundTo = roundTo
        self.is_elite = is_elite
        self.n = [n for n in range(30, 80, 10)]
        self.pk = np.arange(0.5, 0.9, 0.1)
        self.pm = np.arange(0.001, 0.01, 0.001)
        self.t = [t for t in range(50, 150, 10)]
        self.best_x_real = 0 
        
    def run_test(self, nr: int,  n: int, pk: float, pm: float, t: int) -> list:
        avgMax = 0
        avgAvg = 0
        symulation = Symulation(self.a, self.b, n, self.d, self.roundTo, pk, pm, self.is_elite)
        for i in range(self.iterations):
            result = symulation.run(t)
            avgAvg += sum(r[2] for r in result)/len(result)
            avgMax += max(r[3] for r in result)
            self.best_x_real = max(r[3] for r in result)
            
        print(f"{(nr/1800)*100}%")
        return avgAvg/self.iterations, avgMax/self.iterations,
    
    def judge(self, combinations: list) -> list:
        bestAvg = 0
        bestMax = 0
        bestCombination = []
        for combination in combinations:
            comb, values = combination
            avg, max_ = values
            if avg >= bestAvg and max_ >= bestMax:
                bestAvg = avg
                bestMax = max_
                bestCombination = comb
        
        return bestCombination, bestAvg, bestMax
    
    def run_tests(self) -> None:
        results = []
        combinations = list(itertools.product(self.n, self.pk, self.pm, self.t))
        
        def test_runner(i, combination):
            n, pk, pm, t = combination
            return (combination, self.run_test(i, n, pk, pm, t))
        
        with ThreadPoolExecutor() as executor:
            future_to_combination = {
                executor.submit(test_runner, i, combination): combination
                for i, combination in enumerate(combinations)
            }
            
            for future in as_completed(future_to_combination):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"An error occurred: {e}")
        
        bestResult = self.judge(results)
        messagebox.showinfo('Info', f'Najlepsza znaleziona komfiguracja parametrów: N:{bestResult[0][0]}, pk: {bestResult[0][1]}, pm: {bestResult[0][2]}, T: {bestResult[0][3]}, Avg: {bestResult[1]}, Max: {bestResult[2]}, x_real: {self.best_x_real}')
            
class Window():
    def __init__(self) -> None:
        self.root = Tk()
        self.root.geometry("1400x800")
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
            d = float(self.d_box.get())
            roundTo = _map[self.d_box.get()]
            pk = float(self.pk_entry.get())
            pm = float(self.pm_entry.get())
            t = int(self.t_entry.get())
            is_elite = bool(self.elite_var.get())
        except:
            messagebox.showerror('Error', 'Enter valid values!')
            
        return [a, b, n, d, roundTo, pk, pm, t, is_elite]

    def plot_table(self, population: list[int]) -> None:
        columns = ["LP", "x_real", "f(x)", "g(x)", "p", "q", "r", "x sel", "x_bin", "r2", "parent", "crossover_point", "children", "new generation", "mutation points", "bin2", "x real2", "f(x)2"]
        
        tree = ttk.Treeview(self.root, columns=columns, show="headings", height=10)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=90)
        
        for individual in population:
            tree.insert("", tk.END, values=individual.print_values(self.symulation.roundTo))

        tree.grid(row=10, column=0, columnspan=6, padx=10, pady=10)

        vsb = ttk.Scrollbar(self.root, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(self.root, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.grid(row=10, column=6, sticky='ns')
        hsb.grid(row=11, column=0, columnspan=6, sticky='ew')
    
    def plot_summary(self, output: list) -> None:
        generations = [row[0] for row in output]
        min_values = [row[1] for row in output]
        avg_values = [row[2] for row in output]
        max_values = [row[3] for row in output]

        # Create a Figure
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        # Plot the data
        ax.plot(generations, max_values, label='Max f(x)', marker='o')
        ax.plot(generations, avg_values, label='Avg f(x)', marker='o')
        ax.plot(generations, min_values, label='Min f(x)', marker='o')

        # Customize the plot
        ax.set_xlabel('Pokolenie')
        ax.set_ylabel('f(x)')
        ax.set_title('Wartości f(x) dla pokoleń')
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=11, column=0, columnspan=5, sticky="nsew")

        canvas.draw()
        
    def calc(self) -> None:
        self.reset_output_labels()
        a, b, n, d, roundTo, pk, pm, t, is_elite = self.get_data()
        self.symulation = Symulation(a, b, n, d, roundTo, pk, pm, is_elite)
        output = self.symulation.run(t)
        self.plot_summary(output)
        # self.plot_table(population)
        self.set_output_labels(output)
        return
    
    def tests(self) -> None:
        messagebox.showinfo('Info', 'Test zajmuje około 10min!')
        a, b, n, d, roundTo, pk, pm, t, is_elite = self.get_data()
        tests = Tests(a, b, d, roundTo, is_elite)
        tests.run_tests()
        
    def reset_output_labels(self):
        self.iterationLabel.config(text = "Last iteration: ")
        self.minLabel.config(text = "Min: ")
        self.avgLabel.config(text = "Avg: ")
        self.maxLabel.config(text = "Max: ")
        self.bestLabel.config(text = "Best x real: ")

    def set_output_labels(self, output: list) -> None:
        iteration, min_, avg_, max_, bestInv = output[-1]
        self.iterationLabel.config(text = self.iterationLabel.cget("text")+str(iteration+1))
        self.minLabel.config(text = self.minLabel.cget("text")+str(round(min_, 4)))
        self.avgLabel.config(text = self.avgLabel.cget("text")+str(round(avg_, 4)))
        self.maxLabel.config(text = self.maxLabel.cget("text")+str(round(max_, 4)))
        self.bestLabel.config(text = self.bestLabel.cget("text")+f"{bestInv}")
        
    def draw(self) -> None:
        a_label = Label(self.root, text='a:')
        a_label.grid(column=0, row=0)
        self.a_entry = Entry(self.root)
        self.a_entry.grid(column=1, row=0)
        self.a_entry.insert(0, "-4")

        b_label = Label(self.root, text='b:')
        b_label.grid(column=3, row=0)
        self.b_entry = Entry(self.root)
        self.b_entry.grid(column=4, row=0)
        self.b_entry.insert(0, "12")

        n_label = Label(self.root, text='N:')
        n_label.grid(column=0, row=1)
        self.n_entry = Entry(self.root)
        self.n_entry.grid(column=1, row=1)
        self.n_entry.insert(0, "80")

        d_label = Label(self.root, text='d:')
        d_label.grid(column=3, row=1)
        self.d_box = ttk.Combobox(self.root, values=["0.1", "0.01", "0.001", "0.0001"])
        self.d_box.grid(column=4, row=1)
        self.d_box.insert(0, "0.01")
        
        pk_label = Label(self.root, text='pk:')
        pk_label.grid(column=0, row=2)
        self.pk_entry = Entry(self.root)
        self.pk_entry.grid(column=1, row=2)
        self.pk_entry.insert(0, "0.6")

        pm_label = Label(self.root, text='pm:')
        pm_label.grid(column=3, row=2)
        self.pm_entry = Entry(self.root)
        self.pm_entry.grid(column=4, row=2)
        self.pm_entry.insert(0, "0.001")
        
        t_label = Label(self.root, text='T:')
        t_label.grid(column=0, row=3)
        self.t_entry = Entry(self.root)
        self.t_entry.grid(column=1, row=3)
        self.t_entry.insert(0, "120")

        self.elite_var = BooleanVar()
        self.elite_var.set(True)
        elite_box = ttk.Checkbutton(self.root, text="Elite:", variable=self.elite_var)
        elite_box.grid(column=4, row=3)
        
        calc_button = ttk.Button(self.root, text="Calculate", command=self.calc)
        calc_button.grid(column=2, row=8)
        
        tests_button = ttk.Button(self.root, text="Test", command=self.tests)
        tests_button.grid(column=3, row=8)
        
        self.iterationLabel = Label(self.root, text="Last iteration: ")
        self.iterationLabel.grid(column=0, row=9)
        
        self.minLabel= Label(self.root, text="Min: ")
        self.minLabel.grid(column=1, row=9)
        
        self.avgLabel = Label(self.root, text="Avg: ")
        self.avgLabel.grid(column=2, row=9)
        
        self.maxLabel = Label(self.root, text="Max: ")
        self.maxLabel.grid(column=3, row=9)
        
        self.bestLabel = Label(self.root, text="Best individual: ")
        self.bestLabel.grid(column=4, row=9)
        
        self.root.mainloop()

if __name__ == "__main__":
    win = Window()
    win.draw()