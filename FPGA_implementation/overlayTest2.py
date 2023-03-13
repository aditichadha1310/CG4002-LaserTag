# ONLY CAN RUN PYNQ LIB ON FPGA
import pynq
from pynq import Overlay
import numpy as np

class OL():
    # Load overlay
    def load_overlay():
        # Initialise overlay
        overlay = Overlay("amlpbd2.bit")
        # overlay.download()
        if (overlay.is_loaded()):
            print("Bitstream successfully loaded LESGO")
        dma = overlay.axi_dma_0
        return dma

    def confirm_action(input, xoutput, dma):
        # Insert start of move identification here
        # - Take in the first 5 datapoints
        # - If the first 5 shows a possible valid action, proceed to take in 50 datapoints to determine which of the 4 possible actions

        # Allocate input buffer of 6 floats
        in_buffer = pynq.allocate(shape=(6,), dtype=np.float32)

        # Allocate output buffer of 1 integer
        out_buffer = pynq.allocate(shape=(1,), dtype=np.float32)

        # 1st number of the input is the Player ID (i.e. 1 or 2)
        # 2nd to 7th numbers (total 6 numbers) are to be fed into the neural network
        for i, val in enumerate(input):
            if(i == 0):
                player_id = val
            else:
                in_buffer[i-1] = val

        print(player_id)
        print(in_buffer)

        # DMA send and receive channel transfer
        dma.sendchannel.transfer(in_buffer)
        dma.recvchannel.transfer(out_buffer)

        # Wait for transfer to finish
        dma.sendchannel.wait()
        dma.recvchannel.wait()

        # Print and return output buffer
        # Should be a list of 2 elements:
        #   1) Player ID
        #   2) Identified action (by neural network)
        output = int(out_buffer[0])
        print("Player: " + str(player_id))
        print("Predicted Action: " + str(output))
        print("Expected Action: " + str(xoutput))

    input = [1, -1.4491,0.5708,-0.337,-1.5568,-1.0423,-0.7411]
    xoutput = 3
    dma = load_overlay()
    confirm_action(input, xoutput, dma)
