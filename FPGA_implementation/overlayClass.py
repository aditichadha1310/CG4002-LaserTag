# ONLY CAN RUN PYNQ LIB ON FPGA
import pynq
from pynq import Overlay
import numpy as np

overlay = Overlay("design_3.bit")

dma = overlay.axi_dma_0


class OL():
    def overlay(input):
        # Allocate in and out buffer

        # In buffer of 6 floats
        in_buffer = pynq.allocate(shape=(6,), dtype=np.float32)

        # Out buffer of 1 integer
        out_buffer = pynq.allocate(shape=(1,), dtype=np.int32)

        # 1st number of the input is the Player ID (i.e. 1 or 2), 2nd to 7th numbers are to be fed into the neural network
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
        output = [player_id, out_buffer[0]]
        print(output)
        return output
