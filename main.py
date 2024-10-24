import math
from random import uniform
from tkinter import *
from tkinter import ttk

class Window():
    def __init__(self) -> None:
        self.root = Tk()
        self.root.geometry("800x500")
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

        a = int(self.a_entry.get())
        b = int(self.b_entry.get())
        n = int(self.n_entry.get())
        d = self.d_box.get()
        l = self.get_l(a, b, float(d))
        return (a, b, n, _map[d], l)

    def process_data(self, data: list[int]) -> list[int]:
        a, b, n, d, l = data 
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
        lables = ["LP", "x_real", "x_int", "x_bin", "x_int", "x_real", "f(x)"]
        startRow = 10
        startCol = 10
        for column, label in enumerate(lables):
            b = Label(self.root, text=label)
            b.grid(row=startRow, column=startCol+column)
            for row, value in enumerate(data[column]):
                b = Label(self.root, text=value)
                b.grid(row=startCol+column+1, column=startRow+row)
            
    def calc(self) -> None:
        data = self.get_data()
        processedData = self.process_data(data)
        self.plot_table(processedData)
        return

    def draw(self) -> None:
        a_label = Label(self.root, text='a:')
        a_label.grid(column=0, row=0)

        b_label = Label(self.root, text='b:')
        b_label.grid(column=3, row=0)

        n_label = Label(self.root, text='N:')
        n_label.grid(column=0, row=1)

        d_label = Label(self.root, text='d:')
        d_label.grid(column=3, row=1)

        self.a_entry = Entry(self.root)
        self.a_entry.grid(column=1, row=0)

        self.b_entry = Entry(self.root)
        self.b_entry.grid(column=4, row=0)

        self.n_entry = Entry(self.root)
        self.n_entry.grid(column=1, row=1)

        self.d_box = ttk.Combobox(self.root, values=["0.1", "0.01", "0.001", "0.0001"])
        self.d_box.grid(column=4, row=1)

        calc_button = ttk.Button(self.root, text="Calculate", command=self.calc)
        calc_button.grid(column=2, row=8)
        self.root.mainloop()


if __name__ == "__main__":
    win = Window()
    win.draw()