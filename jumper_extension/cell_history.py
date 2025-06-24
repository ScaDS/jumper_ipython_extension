import time
import json

class CellHistory:
    def __init__(self):
        self.cells = []
        self.cell_counter = 0
        self.current_cell = None
        self.cell_timestamps = []
    
    def start_cell(self, raw_cell):
        self.current_cell = {
            'number': self.cell_counter,
            'raw_cell': raw_cell,
            'start_time': time.time(),
            'end_time': None,
        }
        self.cell_counter += 1
    
    def end_cell(self, result):
        if self.current_cell:
            self.current_cell['end_time'] = time.time()
            self.cells.append(self.current_cell)
            self.cell_timestamps.append((self.current_cell['start_time'], self.current_cell['end_time']))
            self.current_cell = None

    def print(self):
        for cell in self.cells:
            duration = cell['end_time'] - cell['start_time']
            print(f"Cell #{cell['number']} - Duration: {duration:.2f}s")
            print("-" * 40)
            print(cell['raw_cell'])
            print("=" * 40)

    def export(self, filename='cell_history.json'):
        with open(filename, 'w') as f:
            json.dump(self.cells, f, indent=2)        