from ai_interface_local import OL
import numpy as np

input = [[-6.67,1.77,1.27,-0.86,3.96,-0.1],
         [-7.15,1.43,0.35,-1.28,4.21,-0.2],
         [-10.16,-0.36,-0.21,-0.32,2.6,-0.65],
         [-11.17,1.03,-1.43,0.38,0.32,-0.58],
         [-9.41,1.12,-2.3,0.62,0.41,-0.39],
         [-11.2,1.62,-3.4,0.38,1.31,0.69],
         [-10.6,1.31,-2.94,0.07,0,0.1],
         [-9.66,0.85,-2.22,-0.12,-1.3,-0.05],
         [-8.74,0.15,-0.96,-0.11,-1.09,0.09],
         [-9.87,0.49,-0.05,-0.23,-1.03,-0.12],
         [-9.5,0.14,0.54,-0.05,-1.3,-0.25],
         [-9.49,0.26,1.2,0.15,-0.5,-0.19],
         [-10.8,0.66,1.11,0.21,-0.48,-0.23],
         [-9.7,0.88,1.25,0.27,-0.08,-0.13],
         [-10.87,0.52,1.04,0.29,0.13,-0.15],
         [-10.5,0.19,1.11,0.03,0.1,-0.09],
         [-10.03,-0.42,1.48,-0.06,0.01,-0.11],
         [-10.06,-0.82,1.42,0.03,0.11,-0.15],
         [-10.04,-0.59,1.18,0.05,0.1,-0.22],
         [-10.04,-0.35,0.72,0.08,0.22,-0.25]]

player1 = OL()
print(type(input))
nparr = np.array(input)
print(nparr.shape)
nparr = nparr.flatten()
print(nparr.shape)
print(player1.confirm_Action(nparr))