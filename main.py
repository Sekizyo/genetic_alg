import math
from random import uniform
import tkinter as tk
from tkinter import ttk, Tk, Label, Entry

class Individual():
    def __init__(self, id: int, x_real: float) -> None:
        self.id = id
        self.x_real = x_real
        self.fx = 0
        self.gx = 0
        self.p = 0
        self.q = 0
        self.r = 0
        self.new_x = 0

class Window():
    def __init__(self) -> None:
        self.root = Tk()
        self.root.geometry("1000x500")
        self.root.title("Algorytm genetyczny - Matas Pieczulis 21162")

    def get_x(self, start: int, stop: int, step: float) -> float:
        return round(uniform(start, stop), step)

    def F(self, x: float, d: int):
        return (x % 1) * (math.cos(20 * math.pi * x) - math.sin(x))

    def x_real(self, x: float, a: int, b: int, l: int, d: int) -> float:
        return round(((x*(b-a))/(2**l-1))+a, d)
    
    def x_int(self, x: float, a: int, b: int, l: int) -> int:
        return math.ceil((1/(b-a))*(x - a)*(2**l - 1))
    
    def get_l(self, a: int, b: int, d: int) -> int:
        return math.ceil(math.log(((b-a)/d)+1, 2))
    
    def int_bin(self, x: float, l: int):
        return format(x, f'0{l}b')
    
    def bin_int(self, x: float) -> int:
        return int(x, 2)

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
            l = self.get_l(a, b, float(d))
        except:
            return (-4, 12, 10, 2, 13, 0, 0)
        return (a, b, n, _map[d], l, pk, pm)

    def process_data(self, data: list[int]) -> list[int]:
        a, b, n, d, l, pk, pm = data 
        output = []
        for i in range(1, n+1):
            x = self.get_x(a, b, d)
            x_int1 = self.x_int(x, a, b, l)
            x_bin = self.int_bin(x_int1, l)
            x_int2 = self.bin_int(x_bin)
            x_real2 = self.x_real(x_int2, a, b, l, d)

            func = self.F(x_real2, d)

            line = [i, x, x_int1, x_bin, x_int2, x_real2, func]
            output.append(line)

        return output

    def plot_table(self, data: list[int]) -> None:
        columns = ["LP", "x_real", "x_int", "x_bin", "x_int2", "x_real2", "f(x)"]
        
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
        data = self.get_data()
        processedData = self.process_data(data)
        self.plot_table(processedData)
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
    win = Window()
    win.draw()