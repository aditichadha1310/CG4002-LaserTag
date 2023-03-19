class StartOfMoveClass:

    def __init__(self):
        self.window = []
        WINDOW_SIZE = 12
        WINDOW_HALF = 6
        
    def input(self, data):
        self.window.append(data)
        if len(window) == WINDOW_SIZE:

