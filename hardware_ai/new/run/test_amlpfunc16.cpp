/*
----------------------------------------------------------------------------------
--	(c) Rajesh C Panicker, NUS,
--  Description : Self-checking testbench for AXI Stream Coprocessor (HLS) implementing the sum of 4 numbers
--	License terms :
--	You are free to use this code as long as you
--		(i) DO NOT post a modified version of this on any public repository;
--		(ii) use it only for educational purposes;
--		(iii) accept the responsibility to ensure that your implementation does not violate any intellectual property of any entity.
--		(iv) accept that the program is provided "as is" without warranty of any kind or assurance regarding its suitability for any particular purpose;
--		(v) send an email to rajesh.panicker@ieee.org briefly mentioning its use (except when used for the course EE4218 at the National University of Singapore);
--		(vi) retain this notice in this file or any files derived from this.
----------------------------------------------------------------------------------
*/

#include <stdio.h>
#include "hls_stream.h"

/***************** AXIS with TLAST structure declaration *********************/

struct AXIS_wLAST
{
    int data;
    bool tlast;
};

/***************** Coprocessor function declaration *********************/

void amlpfunc16(hls::stream<AXIS_wLAST> &S_AXIS, hls::stream<AXIS_wLAST> &M_AXIS);

/***************** Macros *********************/
#define NUMBER_OF_INPUT_WORDS 100 // length of an input vector
#define NUMBER_OF_OUTPUT_WORDS 1  // length of an input vector
#define NUMBER_OF_TEST_VECTORS 1  // number of such test vectors (cases)

/************************** Variable Definitions *****************************/
float test_input_memory[NUMBER_OF_TEST_VECTORS * NUMBER_OF_INPUT_WORDS] = {5.51, -13.105250000000002, -5.789, -1.4289999999999998, -0.08574999999999995, 0.9092500000000001, 10.145655227731721, 12.873724982983752, 9.040936566528934, 2.687293992104325, 2.5076930508935895, 4.401890155092469, 7.781499999999999, 9.28735, 6.4732, 1.9505, 2.0024, 3.0829875, -16.45, -53.58, -29.86, -8.76, -7.53, -8.75, 31.72, 10.79, 16.33, 1.8, 3.58, 8.71, 48.17, 64.37, 46.19, 10.56, 11.11, 17.46, 3.4000000000000004, -9.254999999999999, -4.0649999999999995, -0.5549999999999999, 0.425, 0.38, 3.825, 2.044999999999999, 3.175, 0.7399999999999999, 1.685, 1.195, 11.1675, 8.487499999999997, 7.869999999999999, 1.5950000000000002, 3.48, 2.37, 7, 36, 35, 23, 17, 17, 33, 4, 5, 17, 23, 23, 15, 28, 26, 30, 24, 13, 7, 7, 8, 8, 8, 5, 0.18837716324219778, -1.1148495201898665, -0.5359489408515978, -1.6911688314449382, -0.8261015522668979, 0.018709260771407686, 0.22161418984668213, 1.5625359541701993, 1.105812835135608, 1.7848819461142842, 0.2937533408364894, 0.1970396628270099, 53.317767999999994, 134.99214899999998, 46.100422, 3.705436, 2.518351, 8.081349, 20.13226737596954, 4.539360752217203, 12.3203, 2.6886}; // 6 inputs
float test_result_expected_memory[NUMBER_OF_TEST_VECTORS * NUMBER_OF_OUTPUT_WORDS] = {1};                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             // 1 output
float result_memory[NUMBER_OF_TEST_VECTORS * NUMBER_OF_OUTPUT_WORDS];                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 // same size as test_result_expected_memory

/*****************************************************************************
 * Main function
 ******************************************************************************/
int main()
{
    int word_cnt, test_case_cnt = 0;
    int success;
    AXIS_wLAST read_output, write_input;
    hls::stream<AXIS_wLAST> S_AXIS;
    hls::stream<AXIS_wLAST> M_AXIS;

    for (test_case_cnt = 0; test_case_cnt < NUMBER_OF_TEST_VECTORS; test_case_cnt++)
    {

        /******************** Input to Coprocessor : Transmit the Data Stream ***********************/

        printf(" Transmitting Data for test case %d ... \r\n", test_case_cnt);

        for (word_cnt = 0; word_cnt < NUMBER_OF_INPUT_WORDS; word_cnt++)
        {

            write_input.data = test_input_memory[word_cnt + test_case_cnt * NUMBER_OF_INPUT_WORDS];
            write_input.tlast = 0;
            if (word_cnt == NUMBER_OF_INPUT_WORDS - 1)
            {
                write_input.tlast = 1;
                // S_AXIS_TLAST is asserted for the last word.
                // Actually, doesn't matter since we are not making using of S_AXIS_TLAST.
            }
            S_AXIS.write(write_input); // insert one word into the stream
        }

        /* Transmission Complete */

        /********************* Call the hardware function (invoke the co-processor / ip) ***************/

        amlpfunc16(S_AXIS, M_AXIS);

        /******************** Output from Coprocessor : Receive the Data Stream ***********************/

        printf(" Receiving data for test case %d ... \r\n", test_case_cnt);

        for (word_cnt = 0; word_cnt < NUMBER_OF_OUTPUT_WORDS; word_cnt++)
        {

            read_output = M_AXIS.read(); // extract one word from the stream
            result_memory[word_cnt + test_case_cnt * NUMBER_OF_OUTPUT_WORDS] = read_output.data;
        }

        /* Reception Complete */
    }

    /************************** Checking correctness of results *****************************/

    success = 1;

    /* Compare the data send with the data received */
    printf(" Comparing data ...\r\n");
    for (word_cnt = 0; word_cnt < NUMBER_OF_TEST_VECTORS * NUMBER_OF_OUTPUT_WORDS; word_cnt++)
    {
        success = success & (result_memory[word_cnt] == test_result_expected_memory[word_cnt]);
    }

    if (success != 1)
    {
        printf("Test Failed\r\n");
        return 1;
    }

    printf("Test Success\r\n");

    return 0;
}
