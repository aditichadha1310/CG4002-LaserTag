# ONLY CAN RUN PYNQ LIB ON FPGA
import pynq
from pynq import Overlay
import numpy as np

overlay = Overlay("design_3.bit")

dma = overlay.axi_dma_0


class OL():
    def overlay(input):
        # allocate in and out buffer
        in_buffer = pynq.allocate(shape=(6,), dtype=np.float32)

        # out buffer of 1 integer
        out_buffer = pynq.allocate(shape=(1,), dtype=np.int32)

        # fill in buffer with data populate with array of [0.01, 0.02, .., 0.24]
        # input_data = np.arange(0.01, 0.25, 0.01, dtype=np.double)

        for i, val in enumerate(input):
            if(i != 0):
                in_buffer[i-1] = val

        print(in_buffer)

        # dma send and receive channel transfer
        dma.sendchannel.transfer(in_buffer)
        dma.recvchannel.transfer(out_buffer)

        # wait for transfer to finish
        dma.sendchannel.wait()
        dma.recvchannel.wait()

        # print output buffer
        for output in out_buffer:
            print(output)
        return output
