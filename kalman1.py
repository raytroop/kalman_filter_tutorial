# kalman1.py
# written by Greg Czerniak (email is greg {aT] czerniak [dOt} info )
#
# Implements a single-variable linear Kalman filter.
#
# Note: This code is part of a larger tutorial "Kalman Filters for Undergrads"
# located at http://greg.czerniak.info/node/5.
import random
import numpy as np
import matplotlib.pyplot as plt
from kl_model import KalmanFilterLinear

class Voltmeter:
    def __init__(self, _truevoltage, _noiselevel):
        self.truevoltage = _truevoltage
        self.noiselevel = _noiselevel

    def GetVoltage(self):
        return self.truevoltage

    def GetVoltageWithNoise(self):
        return random.gauss(self.GetVoltage(), self.noiselevel)


def main():
    numsteps = 60

    A = np.matrix([1])
    H = np.matrix([1])
    B = np.matrix([0])
    Q = np.matrix([0.00001])
    R = np.matrix([0.1])
    xhat = np.matrix([3])
    P = np.matrix([1])

    kl_model = KalmanFilterLinear(A, B, H, xhat, P, Q, R)
    voltmeter = Voltmeter(1.25, 0.25)

    measuredvoltage = []
    truevoltage = []
    kalman = []

    for _ in range(numsteps):
        measured = voltmeter.GetVoltageWithNoise()
        measuredvoltage.append(measured)
        truevoltage.append(voltmeter.GetVoltage())
        kalman.append(kl_model.GetCurrentState()[0, 0])
        kl_model.Step(np.matrix([0]), np.matrix([measured]))

    plt.plot(range(numsteps), measuredvoltage, 'b', range(
        numsteps), truevoltage, 'r', range(numsteps), kalman, 'g')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.title('Voltage Measurement with Kalman Filter')
    plt.legend(('measured', 'true voltage', 'kalman'))
    plt.show()


if __name__ == '__main__':
    main()
