from statistics import mean

class StartOfMoveClass:
    WINDOW_SIZE = 12
    WINDOW_HALF = 6

    def __init__(self):
        self.window = []
        startOfMoveDetected = 0
        
    def input(self, data):
        self.window.append(data)
        if len(self.window) == self.WINDOW_SIZE:
            isPossibleMove = self.start_of_move_check(self.window)
            return isPossibleMove
        else:
            return 0

    def start_of_move_check(self,window):
        accX = []
        accY = []
        accZ = []
        for row in window:
            accX.append(float(row[0]))
            accY.append(float(row[1]))
            accZ.append(float(row[2]))

        diffX = mean(accX[self.WINDOW_HALF:]) - mean(accX[0:self.WINDOW_HALF])
        diffY = mean(accY[self.WINDOW_HALF:]) - mean(accY[0:self.WINDOW_HALF])
        diffZ = mean(accZ[self.WINDOW_HALF:]) - mean(accZ[0:self.WINDOW_HALF])

        mean_acc_change = diffX + diffY + diffZ

        if mean_acc_change > 10 or mean_acc_change < -10:
            self.startOfMoveDetected = 1
            return 0
        else:
            return 7