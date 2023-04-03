// C++ code to test accuracy of ML model
// Not to be directly implemented on FPGA

#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>

// #include "ap_axi_sdata.h" // ap_axis can also be used, but it will include all sideband signals which we don't need
//  #include "hls_stream.h"
//  #include "ap_int.h"

#define INPUT_SIZE 100  // number of inputs
#define LAYER_1_SIZE 16 // number of nodes in layer 1
#define LAYER_2_SIZE 16 // number of nodes in layer 2
#define OUTPUT_SIZE 4   // number of outputs

int main()
{
    // Define input test data
    int expected_result = 4;
    float input[INPUT_SIZE] = {0.24324999999999983, -10.536750000000001, -3.215, -0.8452500000000001, 0.03200000000000003, 0.28575000000000006, 8.410948040351931, 12.615768781073154, 7.669250941258865, 2.3939110128615892, 3.4773166378689186, 2.7875023654698485, 5.6360624999999995, 6.2422875000000015, 4.88925, 1.6348749999999999, 2.5549000000000004, 1.85875, -21.57, -78.7, -15.93, -8.76, -8.69, -8.75, 26.52, 9.51, 32.26, 1.39, 4.99, 5.48, 48.09, 88.21000000000001, 48.19, 10.15, 13.68, 14.23, -1.225, -9.175, -2.205, -0.03, 0.935, 0.215, 2.625, 2.690000000000001, 2.115, 0.55, 1.605, 1.23, 5.25, 4.437499999999999, 7.315, 1.2125000000000001, 2.9125, 2.6799999999999997, 25, 36, 34, 21, 14, 15, 15, 4, 6, 19, 26, 25, 15, 29, 25, 30, 26, 20, 12, 9, 12, 8, 7, 7, 0.8812531564304672, -3.8466423253204076, 2.186448792197021, -2.167383120076574, -1.2696155582326993, -1.2844012211508546, 2.648583681045414, 18.62571964475787, 9.103241547384865, 3.9595505919228717, 1.058302818046391, 3.3244869759846747, 28.321287, 108.072289, 27.661454000000003, 2.578103, 4.837102, 3.140729, 15.193860957261004, 3.871826054148567, 8.9658, 2.3164};
    // Define the weights and biases
    float w1[INPUT_SIZE][LAYER_1_SIZE] = {{-0.21111391, -0.17394981, -0.055259287, 0.22600919, -0.039178435, 0.046841275, 0.08613842, 0.11019336, 0.23498958, 0.18031937, -0.15806875, 0.058781207, 0.29583052, -0.07197704, 0.07076211, 0.10182482},
                                          {-0.20850466, -0.082502425, -0.07714081, 0.047337756, 0.110604614, -0.20832652, -0.117986545, -0.10008835, 0.15872358, -0.09508201, -0.14515994, -0.08256629, -0.15062118, -0.06192283, 0.05908247, 0.17868075},
                                          {-0.04524999, 0.18997532, 0.123591006, 0.19684269, -0.124871455, 0.06945229, -0.0634678, 0.06360028, 0.07001846, 0.13247707, 0.21959794, -0.07003066, 0.12811236, 0.14775693, 0.02425695, -0.105979174},
                                          {0.22372355, -0.17979482, -0.009538606, -0.06043869, 0.28722695, 0.18442379, -0.09945579, -0.28648946, 0.06407682, 0.07346946, 0.1449317, -0.07509534, -0.17333472, 0.2709286, -0.41934693, -0.28459576},
                                          {0.018575769, 0.17604423, -0.10430733, 0.1869799, 0.06870757, 0.107091345, -0.18925127, -0.3917064, -0.012309747, 0.06383185, 0.018788353, 0.06262526, 0.29066265, -0.33368188, 0.12064437, 0.21766098},
                                          {0.36670813, 0.14369792, 0.22231066, 0.27983397, -0.03149328, 0.19603886, 0.20209908, -0.2212847, 0.28019428, 0.03576366, -0.16161686, -0.19015908, 0.15551503, -0.22503978, -0.022752924, 0.28702217},
                                          {-0.23101446, -0.02269119, -0.11923346, 0.14656661, -0.1999112, 0.0408198, 0.14799336, 0.18381271, -0.009760248, 0.09567971, -0.19525672, 0.004593909, 0.016392978, -0.046977725, -0.19773859, 0.09301824},
                                          {0.19302693, 0.0033523142, -0.12455115, 0.1499191, -0.22369285, 0.11860202, -0.02847982, -0.066399, 0.003518595, -0.07954197, 0.04754594, 0.0048851967, -0.0047918595, 0.18362296, -0.064584725, 0.14723308},
                                          {0.047491975, -0.09856269, 9.353459e-05, 0.1485251, 0.03915818, -0.14295977, 0.044702157, -0.12532322, 0.16264671, 0.13996351, 0.030514598, 0.18646592, -0.06554232, 0.034564473, 0.14242934, -0.0996095},
                                          {-0.08850235, -0.13159856, 0.17973784, -0.13882262, -0.057459824, -0.038237225, -0.22843084, -0.15639508, 0.10193385, -0.066360615, -0.028103277, -0.07529256, 0.18087728, 0.1497969, 0.0958634, -0.049879067},
                                          {0.083210595, -0.042011634, -0.11907453, 0.027843868, -0.19261982, 0.13551353, 0.29242897, 0.20922945, 0.10950243, -0.12897572, -0.18408404, -0.026811674, -0.22404547, 0.1379936, -0.14916232, 0.21662913},
                                          {-0.18148422, -0.11725886, -0.049546793, 0.16639203, -0.23206168, 0.019185439, 0.0474647, -0.14292003, 0.15243761, -0.061669745, 0.052781314, -0.06347732, 0.003330149, -0.018868737, -0.10046243, 0.12486801},
                                          {-0.102742076, 0.21855175, -0.10196021, -0.1484262, -0.23395753, 0.010980432, -0.20248592, -0.14193432, -0.14202338, -0.08086079, 0.017758757, 0.089619905, 0.035235148, -0.09556725, 0.1786365, -0.14344835},
                                          {-0.031783696, 0.08566445, 0.18812472, -0.10032869, 0.21030706, -0.22563522, 0.0102486005, -0.15829502, -0.08517982, 0.16776988, 0.090795875, -0.20454422, 0.20527664, 0.007898707, 0.10730799, 0.039635595},
                                          {-0.19371164, -0.21869013, -0.07767101, 0.12968884, 0.009163679, 0.07592342, -0.1397448, -0.10286899, 0.10712115, 0.11152029, 0.033168197, 0.20320657, 0.15098645, 0.18910983, 0.04374415, 0.08590715},
                                          {-0.28727993, 0.20897275, 0.22554749, 0.10489236, -0.1541061, 0.14815167, -0.22788414, -0.035447665, 0.20008627, -0.06812345, 0.15471247, -0.21599816, -0.0080043385, 0.09503184, 0.15829226, -0.15110591},
                                          {0.16202568, -0.018266782, -0.13434048, -0.17548159, 0.047135536, 0.18884636, -0.08528441, -0.06659525, 0.059723016, -0.15689285, 0.13319945, 0.0378502, 0.039779615, -0.26020393, 0.22914085, -0.052037776},
                                          {-0.044556975, 0.08446801, -0.04243502, -0.25019425, 0.08211763, -0.08116413, -0.18965618, -0.13719644, -0.24723981, 0.042990144, -0.116225146, -0.18227983, -0.048705023, -0.19972661, 0.24723075, 0.19655055},
                                          {-0.22575366, -0.053284734, 0.097696096, -0.1145188, 0.12012314, 0.019504694, -0.19909209, -0.2804303, 0.08995311, -0.060779244, -0.063319206, -0.06528297, 0.11399475, 0.0805459, 0.13084435, 0.121908285},
                                          {-0.08772511, 0.074359745, 0.11056775, -0.14447968, 0.14370914, 0.10947338, -0.17916305, -0.15714322, -0.15267849, 0.16295126, 0.10324627, -0.1411781, -0.15320706, 0.0917566, 0.04983541, 0.003771008},
                                          {-0.20072326, -0.09142287, -0.18531437, -0.15191188, 0.04738774, 0.18674722, 0.1527111, -0.019771716, -0.07608504, 0.18649907, -0.07280196, 0.18086958, 0.08278364, -0.014085815, 0.06281642, 0.11360724},
                                          {0.09167022, 0.20301586, -0.18441182, 0.026403248, 0.058512583, 0.09548042, -0.12666383, 0.15434328, 0.0055855475, 0.17676863, -0.12133684, 0.06338218, 0.16835971, -0.112262055, -0.16691129, -0.02269399},
                                          {0.16505224, 0.17127663, 0.21008787, 0.08103944, 0.15217519, 0.1671894, 0.1558508, 0.11076877, -0.026002374, -0.050771542, 0.18002656, 0.1460971, 0.1507422, -0.08707895, 0.2266411, 0.17697516},
                                          {0.19829865, -0.09290166, 0.014839649, -0.12238698, -0.07375387, 0.06764891, 0.23536076, -0.065019116, -0.024178859, -0.06861287, -0.1899443, 0.047623932, 0.08547976, 0.12970957, 0.09818603, -0.20084538},
                                          {-0.16811465, -0.117187664, -0.07909486, 0.027991356, 0.0619055, -0.19668667, 0.0969747, -0.14015263, -0.1040658, 0.20043626, -0.10305927, 0.00078743696, -0.0008714101, 0.1895946, -0.15256152, -0.052553773},
                                          {0.24045163, 0.073620886, 0.096194535, -0.16112372, -0.16711286, -0.11444585, -0.009149654, 0.108069286, 0.053406067, -0.0112409, 0.14701629, 0.003423497, -0.04933727, -0.09337013, -0.01152941, -0.07084344},
                                          {0.16004032, -0.15421087, -0.19346046, -0.11422131, 0.0423002, -0.020765128, 0.27192095, -0.21891089, -0.19688523, -0.013026999, -0.13864072, 0.16498792, -0.20082065, -0.01932808, 0.09017356, -0.13009559},
                                          {0.20447028, 0.0361515, 0.07060292, -0.24868482, -0.114137, -0.013254764, -0.14166984, -0.10794948, -0.15484679, -0.27553585, -0.16435394, -0.13561945, 0.11361063, -0.06486922, -0.17422563, 0.16297393},
                                          {-0.09292402, -0.1270105, 0.117572874, -0.06415745, -0.08010649, -0.11945244, -0.021238972, -0.00031991728, 0.18052608, -0.04027073, 0.1364257, 0.18176144, 0.24759728, -0.19860968, 0.043281626, -0.14113036},
                                          {-0.13940583, -0.0005439222, -0.05609183, 0.116041265, 0.0037676808, -0.18558317, 0.12559673, 0.12480175, -0.07539006, 0.004629924, 0.15006834, -0.14176247, 0.18409295, -0.16084792, 0.031723797, -0.06058817},
                                          {-0.018043889, 0.07102749, -0.12293806, 0.112214014, -0.24299394, -0.2018249, 0.17344783, -0.018853255, -0.10406883, -0.0107524665, 0.037431926, -0.12298116, 0.06878624, 0.10502236, 0.11276507, -0.02586674},
                                          {0.09713784, 0.081486434, -0.16365668, -0.08163521, -0.16174701, -0.10566132, -0.0688482, -0.24226302, -0.107244976, 0.16467541, 0.016835064, -0.034947306, -0.11033372, -0.05205307, 0.07309152, -0.10094366},
                                          {-0.07959084, 0.21795094, 0.02150403, -0.097377114, 0.23075825, -0.02185401, -0.19033183, 0.00013605792, 0.12077214, -0.091783784, -0.19233766, -0.020747826, 0.020568447, -0.010256648, 0.14290497, -0.23376626},
                                          {0.097275995, -0.10673415, 0.17228758, -0.16472958, 0.07918096, 0.008553178, 0.05404414, 0.10732577, 0.13633445, -0.00953173, 0.054976076, -0.1593945, 0.026344666, -0.24463713, -0.08443607, -0.044266764},
                                          {-0.11372441, 0.11413938, 0.043281555, 0.01717358, -0.1290504, 0.17090458, 0.0745344, -0.048781294, -0.062361103, 0.11358945, 0.056527197, 0.012300745, 0.12835754, 0.08801558, 0.20375298, 0.06547677},
                                          {-0.1856755, 0.14178503, 0.19853821, 0.04989407, 0.01885397, -0.09855191, -0.037440717, -0.087605916, 0.05867965, 0.11844742, 0.14032158, 0.0039334744, -0.060031, 0.010197556, -0.04995981, -0.043264844},
                                          {-0.06447444, -0.1376693, 0.096820205, 0.106634684, -0.2099534, -0.015821034, -0.20460433, 0.20953675, 0.17066866, 0.12538904, -0.07215324, -0.010331944, 0.10204211, -0.069388494, 0.23676513, 0.17436576},
                                          {0.1482631, -0.1899527, 0.08266693, 0.17110117, -0.14794618, -0.09361081, 0.14506924, 0.13892186, -0.039188188, -0.120805286, 0.068835616, 0.21847963, 0.18591572, 0.1028502, 0.12832075, 0.09157155},
                                          {-0.24513917, -0.15050235, 0.044953704, 0.13971123, 0.12257362, -0.06814572, 0.20171338, -0.29135665, -0.05266339, 0.23784827, 0.22396025, 0.0731926, 0.08768762, 0.20597793, 0.15072493, -0.06633443},
                                          {-0.115377, -0.2220543, -0.10737626, -0.18867952, 0.2703615, 0.0037250724, -0.24233134, -0.2272671, 0.06324166, 0.12614244, -0.17010243, 0.1980983, -0.030249905, 0.1844857, 0.07755917, -0.15592855},
                                          {-0.16365306, -0.13796395, -0.041487023, 0.1602505, -0.23832206, -0.23130743, -0.078360125, -0.42525774, 0.3008402, -0.2683219, -0.019173726, -0.14211367, -0.07014755, -0.29288417, 0.010485972, 0.20322248},
                                          {0.100343995, -0.15710318, 0.22425768, 0.17522724, 0.056563403, -0.09558318, 0.1384658, -0.46129692, -0.060143575, -0.011697973, 0.05790469, 0.21005696, 0.00036516547, -0.12500522, 0.242664, 0.13210604},
                                          {0.15516256, 0.07017678, 0.0013550967, 0.13184498, -0.15631077, 0.03990785, 0.16780253, 0.034205023, 0.04115885, -0.054526884, 0.055137455, 0.09535113, -0.08916591, -0.20317763, 0.14964662, -0.12863754},
                                          {-0.17735882, 0.028634727, 0.21641791, -0.28723967, -0.051735975, 0.06570569, 0.13318412, 0.057352155, -0.16121724, 0.015940685, 0.07750824, 0.20569354, -0.09028162, 0.2919276, -0.015846418, 0.083508834},
                                          {0.06365926, 0.12517938, 0.0077103525, 0.0041412497, 0.11562613, 0.20736502, -0.17333186, 0.17963699, -0.07174516, 0.14303264, -0.055444837, -0.22237834, -0.12524728, -0.1337608, -0.11746324, -0.046786673},
                                          {0.0144484285, 0.0064338297, -0.05407396, 0.106212005, 0.10267911, 0.009380695, 0.20487496, -0.1399252, 0.057487104, 0.07940207, -0.17893337, -0.158151, 0.1501262, -0.0131540755, 0.17312543, 0.1511236},
                                          {-0.25312972, -0.10614149, 0.057371378, 0.11862889, -0.19222784, -0.032278527, 0.05185333, 0.11839105, -0.20351177, 0.13089125, -0.014063224, -0.012599841, -0.1587181, 0.0187779, 0.06847618, -0.010301402},
                                          {-0.24880664, -0.11609072, 0.12397784, -0.007652849, -0.11789023, 0.12495384, 0.059951965, 0.017465943, 0.11622828, -0.042256735, 0.13488749, -0.15539369, 0.024225758, -0.090432145, -0.017178167, -0.22037096},
                                          {-0.23525982, -0.18699335, 0.1988365, 0.024771746, 0.0383107, 0.06392676, -0.2016397, -0.13974993, 0.12673213, 0.14131227, -0.17931657, 0.06752929, 0.18662001, -0.07367086, 0.25674918, -0.18982361},
                                          {0.07488975, 0.044113815, 0.16689408, -0.1471079, -0.02925873, -0.18002582, 0.0048213773, -0.10389435, -0.17745475, -0.17621566, 0.081418455, 0.0060986727, -0.09412741, 0.103995934, -0.1851977, -0.08665675},
                                          {-0.043700524, -0.06897667, 0.14550745, -0.19598499, -0.0041154902, 0.079459414, -0.13466664, -0.21516538, -0.08831189, 0.13999923, -0.17749211, -0.18999456, 0.051853176, -0.15999752, 0.21557532, -0.11980706},
                                          {-0.015539434, 0.160604, 0.14455205, 0.1037703, -0.09308403, -0.09559142, -0.13013582, -0.08096819, -0.15491901, -0.04217386, -0.055310145, 0.17981923, 0.0114621585, 0.028999215, 0.30987433, 0.047170762},
                                          {-0.1374884, -0.21198405, 0.20269564, -0.0347126, -0.041345607, 0.16250119, -0.01257472, -0.061566744, -0.075910255, -0.278149, 0.21590272, 0.033043474, -0.013305163, 0.10403989, -0.07082724, -0.16891716},
                                          {-0.25533405, -0.05218476, 0.22712159, -0.064843416, -0.28273144, 0.05085609, 0.08930368, -0.043074656, 0.057772364, -0.017708234, 0.22320196, 0.11476734, 0.08528514, -0.16990475, -0.09488417, -0.111736596},
                                          {0.008540474, -0.038279817, -0.18597996, -0.24173461, 0.11882572, -0.20950748, 0.0024192648, 0.20848331, 0.14697848, 0.10179457, -0.10439924, -0.030770585, -0.0613844, 0.18333238, 0.1749895, -0.19719401},
                                          {0.20977241, -0.1532453, -0.059116676, -0.14608982, -0.093422465, 0.18898328, 0.12798779, -0.018703481, -0.107651316, 0.061912958, -0.047928005, 0.16440463, 0.12245453, -0.10576619, -0.09545767, 0.09257372},
                                          {0.033486933, -0.02831009, 0.026563823, -0.008179595, 0.09812467, -0.13922358, 0.045342885, -0.00981978, 0.066224225, 0.12107059, -0.1319008, -0.033481106, -0.01971574, -0.23678966, -0.11674063, 0.058173917},
                                          {-0.17559767, 0.03796485, 0.030917794, -0.18244037, 0.1020389, -0.053143263, -0.008184229, 0.0791096, 0.031750318, -0.21724336, -0.07734947, 0.16120654, 0.1958298, -0.13850754, 0.2512121, -0.19887109},
                                          {-0.14367145, 0.0026146024, -0.17798863, -0.043550868, 0.107895866, -0.1845445, 0.15073155, 0.13232899, 0.1585823, -0.05280355, 0.13373566, 0.14020392, -0.16380914, -0.147234, 0.11884488, -0.23227203},
                                          {0.1307209, 0.021855175, 0.03402403, -0.103502184, 0.033615474, -0.030172117, -0.22078821, 0.08855389, -0.04386739, -0.12860239, -0.030649558, -0.0884652, -0.17991573, 0.24655813, -0.24783318, 0.15250306},
                                          {0.05979879, -0.08951427, 0.0990521, 0.008821668, 0.1714005, -0.1560903, -0.1505835, -0.22007094, -0.09018886, -0.17187384, -0.08910596, 0.110488206, -0.048458662, -0.12440217, 0.1808182, 0.07657107},
                                          {-0.18860365, -0.13554841, -0.101995245, 0.10796087, -0.10189765, 0.14948215, 0.036189992, 0.11630514, -0.23943985, -0.09116335, -0.12661636, -0.15856451, -0.26789957, 0.031241715, 0.0790299, 0.19863334},
                                          {-0.03157685, -0.054508775, 0.18445104, -0.13190594, 0.08086658, -0.045804225, 0.23436114, -0.07826516, -0.12001398, -0.020545296, -0.076217815, -0.15412802, -0.095382765, -0.13665712, 0.17817, 0.15306875},
                                          {0.19386932, 0.021836683, 0.15381432, -0.017694931, 0.19956677, 0.094665, -0.05501641, 0.001954546, -0.20358469, 0.11993347, 0.025947034, 0.05552417, -0.1513693, -0.07975689, -0.1567672, 0.036678273},
                                          {0.087346084, 0.01402314, 0.12238005, 0.03313019, 0.06247143, -0.19975448, 0.079286925, -0.19491507, 0.1528035, 0.16659056, -0.1761055, 0.22511744, 0.032509208, -0.03611347, 0.2548142, 0.060308445},
                                          {0.013656307, -0.041879877, 0.22090358, 0.19216156, 0.13607492, -0.049451265, 0.055482253, -0.06702839, -0.12740985, -0.0837018, -0.03029868, -0.1549381, 0.027937973, 0.13825442, 0.038763795, 0.06068711},
                                          {-0.11221839, -0.21038657, 0.2156893, -0.2179673, -0.19733325, 0.1503751, -0.113746725, -0.06094941, 0.09409646, 0.1058029, -0.1246115, -0.016415164, -0.062188346, -0.03241405, -0.20840037, 0.16395286},
                                          {0.08673683, -0.15040308, -0.13498107, 0.12415147, 0.076698795, -0.042510428, 0.096435815, 0.14524373, -0.09321887, -0.029386614, 0.033837795, -0.10307288, -0.046764854, 0.1547223, 0.08050384, 0.104434095},
                                          {0.18608786, -0.17506236, -0.11972917, -0.074246675, -0.014404073, -0.014385056, -0.09053371, 0.10166986, -0.20553957, -0.15921986, -0.07182594, 0.012406915, 0.087471336, 0.123958744, -0.20410572, -0.007413443},
                                          {-0.20132, -0.13633117, -0.19287094, 0.16290107, 0.018047059, 0.1388492, -0.13154164, -0.099697664, -0.16326855, 0.11826323, 0.15728632, -0.19993761, -0.01618386, 0.04651138, -0.07496443, 0.07422974},
                                          {-0.07379929, -0.11125958, -0.08642429, 0.065657735, -0.03477079, -0.18183468, 0.10156316, 0.32514602, 0.18791577, 0.042282086, 0.21754819, 0.023379922, 0.02014061, -0.15597433, 0.02101283, -0.16705689},
                                          {-0.089322485, -0.09799686, -0.028335035, 0.053966228, 0.22222699, -0.021388084, -0.25547546, -0.13293004, -0.046557996, -0.14777721, 0.00584355, 0.04639706, -0.16322306, -0.015208829, -0.1778203, -0.15801068},
                                          {0.16331008, 0.11030075, -0.21477866, -0.13357905, 0.14026065, -0.09845313, 0.16473617, 0.14632767, -0.110818125, 0.1680783, -0.048536018, -0.031869262, 0.019856801, 0.08689651, -0.1851147, 0.113551185},
                                          {0.24200001, -0.003346458, 0.15252808, 0.16280176, 0.08265581, 0.11965313, 0.012701016, 0.13698533, 0.091807924, 0.026365697, -0.22537088, -0.008628026, -0.088972576, -0.18047126, -0.087369226, -0.1706768},
                                          {0.02766547, -0.020053387, -0.20145999, -0.26182365, 0.12901191, 0.035803318, -0.02885805, 0.12262477, 0.019103413, 0.13942535, -0.076875925, 0.15407893, -0.08250217, -0.040530313, 0.09800996, 0.21888098},
                                          {0.19195853, 0.035734355, 0.07380921, -0.25222215, 0.19265097, 0.20955105, -0.16368721, 0.025839979, -0.18200305, -0.14034584, 0.081563324, 0.044172764, 0.19667962, -0.094866335, 0.18222326, 0.10376413},
                                          {0.039906193, -0.20536624, -0.17776051, -0.15076874, 0.071994975, 0.16482133, -0.18741198, -0.092063025, 0.14532822, -0.14561749, -0.15458778, -0.17479596, 0.2159301, 0.176535, -0.016072504, -0.15256889},
                                          {-0.16381602, -0.06901695, -0.04073012, -0.2613058, -0.2652932, 0.047872692, -0.03481325, 0.19150552, -0.21283522, -0.023930768, 0.19552276, -0.093231544, -0.093116775, -0.08053554, -0.054826874, -0.20807892},
                                          {0.17020436, 0.22741011, 0.01506348, 0.14162302, -0.279912, 0.029815638, -0.21428865, -0.124257065, -0.12043368, -0.018463485, -0.026064917, -0.12401276, 0.07402749, -0.24325538, -0.016888881, -0.17007433},
                                          {-0.07143222, -0.1353344, -0.16020668, -0.107379906, -0.17889403, 0.018940967, -0.17691156, 0.3431818, -0.15287456, 0.25741655, 0.18026227, -0.033899114, -0.17842512, 0.15999353, 0.14315122, -0.19858128},
                                          {0.18757914, 0.18533304, 0.028185636, -0.108844854, -0.12689574, -0.119213045, 0.3274583, -0.06679649, -0.031507827, -0.13768363, 0.21839976, -0.1972862, -0.032061893, -0.23623936, -0.115716174, -0.20834097},
                                          {0.0034868605, -0.1274854, 0.08017385, 0.15350634, -0.18312697, -0.03069747, -0.041062508, 0.035885874, 0.1858624, 0.023703389, 0.1872603, -0.18124329, -0.15492943, 0.037219767, -0.024009865, 0.29882815},
                                          {0.13361414, -0.14098513, -0.2031915, 0.26318198, -0.013915265, -0.012847539, -0.06603843, -0.355618, -0.08973669, -0.18757556, 0.15119365, 0.012533411, 0.056321695, -0.23331788, -0.08565784, 0.08314823},
                                          {-0.2857058, 0.12955153, 0.034112543, -0.26290327, -0.2509171, -0.10668339, 0.006097805, 0.2949081, -0.11960735, -0.054566953, 0.098438084, -0.040096357, -0.090219446, -0.10568637, -0.08954906, -0.15799339},
                                          {0.3474499, -0.090911165, -0.13003242, 0.0038620525, 0.097712345, -0.078954674, 0.0056304177, -0.031796172, -0.13248911, -0.22745095, 0.085701436, -0.17136714, 0.13179511, -0.21310075, -0.025792167, 0.022248521},
                                          {-0.103411645, -0.1805559, -0.22506706, 0.020887919, -0.056208897, 0.044508375, 0.112059765, 0.0528369, -0.05674966, -0.090804085, -0.05594331, -0.19591706, 0.06216015, -0.18823351, -0.01698633, -0.04955761},
                                          {0.36694905, 0.19524041, -0.1972823, -0.07710368, 0.20258865, -0.030706175, 0.066125475, -0.14549187, 0.050291207, 0.05791057, 0.14820817, -0.21216679, 0.14465915, -0.06554338, -0.3720239, 0.023410846},
                                          {-0.02544957, -0.06528248, -0.053794876, 0.069871776, 0.19476078, -0.19673029, 0.026438903, -0.17044331, 0.03327086, -0.019216724, 0.025229812, 0.09600547, -0.017100979, 0.13617949, 0.09605425, 0.28660947},
                                          {0.13665576, -0.2223256, -0.21642445, 0.20878756, 0.31514835, 0.057631563, -0.14929098, -0.040172465, 0.0701823, -0.18668845, 0.03163886, 0.18535987, 0.2668832, 0.0854402, -0.16762693, -0.23104066},
                                          {0.03575865, 0.2081945, -0.03835492, -0.09796598, 0.1888828, -0.088255785, 0.2288433, -0.3512302, -0.16885142, 0.056544892, -0.03245385, 0.10249096, 0.2471789, 0.14895163, 0.1222818, 0.14053299},
                                          {-0.10314961, -0.18785621, -0.10423044, -0.014247816, 0.03672221, 0.047482617, -0.24579303, 0.14661606, 0.10566807, -0.2223477, 0.17915303, 0.092927694, 0.10415058, 0.13466862, -0.11955346, 0.10194847},
                                          {-0.14560163, -0.033564016, -0.15108731, 0.1332919, 0.046178937, 0.11172304, 0.042422652, -0.029927785, 0.11421372, -0.04520775, -0.12982264, -0.22106397, 0.011501417, -0.07393912, 0.09840751, 0.08668879},
                                          {0.039928686, -0.2015076, -0.1403352, 0.044038974, -0.034634322, -0.06668483, -0.0080919, 0.13359146, -0.08321132, 0.05610505, 0.18196571, 0.009440407, 0.02740763, 0.1312052, -0.034788057, 0.19800785},
                                          {0.12533481, 0.13187927, 0.13538429, -0.19226934, -0.13131723, -0.07821261, 0.03512691, -0.13767815, -0.08746762, 0.17155722, -0.021085203, 0.09356779, -0.18754466, -0.19616532, 0.23296575, 0.1993776},
                                          {-0.011983611, 0.12628749, -0.028009042, -0.07979585, -0.06777854, 0.10685723, -0.045575913, 0.23182908, 0.1725931, -0.06853971, 0.065399885, -0.09207627, -0.18494749, -0.11259835, -0.16020717, -0.12932412},
                                          {-0.07880896, 0.1514548, -0.17651056, -0.18929166, 0.16300744, 0.13484028, -0.093699045, -0.14225744, -0.11077885, -0.108852185, -0.07799797, 0.11118236, -0.13747679, 0.10563318, -0.041080732, 0.115215674},
                                          {0.16365924, -0.21065368, -0.178618, -0.21231954, 0.14200439, 0.13507384, 0.06853975, 0.09981624, 0.016947929, 0.0578303, -0.14326695, 0.047725767, -0.033518642, 0.096794784, -0.20092446, -0.14715385},
                                          {0.12382753, 0.2112861, 0.09372705, 0.029662993, -0.261142, -0.06985408, 0.065152295, -0.048029795, -0.1873127, 0.16243663, -0.086382106, 0.15257788, -0.1361149, 0.13858233, -0.1705692, 0.103568934},
                                          {0.0026549234, -0.13444096, -0.19154426, 0.11752253, 0.018406542, 0.059960753, -0.051811755, -0.0832015, 0.11247445, 0.18507642, -0.13751087, -0.08561279, 0.13644356, -0.22037345, -0.08901854, 0.060052168},
                                          {-0.18007378, 0.06790018, -0.026603028, 0.035293855, -0.028154623, 0.18681347, 0.21296419, -0.22213458, 0.010159629, 0.05448391, -0.20062593, -0.16076756, -0.039663117, -0.10787575, 0.028616501, 0.13111207}};
    float b1[LAYER_1_SIZE] = {0.009862435,
                              0.0,
                              0.0,
                              -0.050082687,
                              -0.04022722,
                              -0.012412175,
                              0.016657514,
                              0.023196291,
                              -0.044985514,
                              -0.037931915,
                              0.0,
                              0.0,
                              -0.008738017,
                              -0.023684327,
                              0.011506127,
                              -0.00405792};

    float w2[LAYER_1_SIZE][LAYER_2_SIZE] = {{0.105688676, -0.21746802, -0.07134576, 0.04856655, 0.3166988, 0.2726997, -0.27447724, -0.15492739, -0.39015338, -0.30940288, 0.21996294, 0.117555745, -0.37292627, 0.024302047, -0.33800712, -0.3361333},
                                            {-0.0143525, 0.4039403, 0.24061695, 0.36244252, -0.22460096, 0.3911561, 0.13788655, 0.013252497, -0.19215481, 0.38125077, 0.36713025, -0.38163027, -0.003359884, -0.31120384, -0.1576213, 0.32212105},
                                            {-0.41372955, -0.13891557, -0.3606782, -0.20700295, 0.18708912, 0.17457756, -0.06651434, -0.07730472, -0.40993336, 0.012092829, -0.22733366, -0.04128942, 0.22350755, -0.16207728, 0.10893008, -0.19709858},
                                            {-0.04205052, 0.06197931, -0.24716944, -0.38217553, -0.08813819, -0.293496, 0.27324066, 0.20739357, 0.29979575, -0.23982313, 0.010591811, -0.34156856, -0.04852437, -0.30416954, -0.3888065, 0.2771137},
                                            {0.08366113, 0.045958452, 0.12050004, -0.2909372, -0.121499516, -0.07283517, 0.14825986, -0.17414597, 0.39500654, -0.13429327, -0.2505412, 0.32926333, -0.1775105, -0.37996083, 0.028501239, -0.074120186},
                                            {0.3804441, 0.35987493, -0.1833168, -0.022575617, -0.075638846, 0.25453737, -0.29404497, 0.22126983, -0.15511295, 0.20945862, -0.40140495, -0.18357024, -0.066714644, -0.2806201, 0.17001651, -0.31016272},
                                            {0.09249767, 0.023024257, 0.4167177, 0.30594432, 0.4551409, -0.17819794, 0.43884668, 0.19786444, -0.25496995, 0.19610845, -0.1910317, -0.23626044, -0.3051752, -0.44426063, -0.33938286, -0.08998514},
                                            {-0.05289857, 0.22397429, 0.49702, 0.035607055, 0.18267262, 0.21020316, -0.2741177, -0.06310429, 0.08931338, 0.056176223, -0.07881685, -0.09022803, 0.32666367, -0.13336478, -0.14671685, 0.08738307},
                                            {0.034918822, -0.31067595, -0.2478561, -0.51554143, -0.28013682, 0.19530062, 0.33387944, 0.35387602, 0.3341119, -0.22932415, 0.119715266, 0.34994254, -0.36100638, 0.30359265, -0.093301885, 0.29786137},
                                            {-0.12673582, 0.40666008, 0.013311241, -0.2182119, 0.21482089, -0.4467616, 0.21006204, 0.12583926, 0.17133395, 0.39692134, 0.2925497, 0.25978658, 0.14066389, 0.37985992, 0.23203994, 0.0095612155},
                                            {-0.13766658, 0.054871142, 0.2704011, 0.1720691, -0.42228314, 0.4080052, -0.201097, -0.2978196, 0.16805682, -0.09799799, 0.25745448, -0.4304747, 0.087299734, 0.023326278, 0.38442, 0.28450033},
                                            {0.22341225, 0.35603496, 0.36103407, 0.22227064, 0.28396192, 0.052597016, 0.022625804, 0.35052946, 0.19352397, 0.29463252, -0.10566592, 0.3707302, -0.11948776, -0.041819453, -0.048838943, -0.4197346},
                                            {-0.17658053, -0.22626202, 0.057912648, 0.07333029, -0.05457122, -0.45154816, 0.48792043, 0.3114056, -0.2725631, -0.21781842, -0.057094015, 0.31523696, 0.18283221, 0.015140926, 0.200511, -0.31610507},
                                            {0.222782, 0.40800273, -0.26989084, -0.22045052, 0.23681639, -0.32258838, -0.31849667, -0.45548084, 0.22136237, 0.30002066, -0.37353405, -0.1239449, 0.21857356, 0.25524512, 0.17863354, 0.19869627},
                                            {0.12692094, 0.08487331, -0.19183911, 0.35518458, 0.008890993, -0.22839552, 0.35248145, 0.4108355, -0.31168216, 0.016869515, -0.19471942, -0.24101461, 0.061299056, 0.19278264, -0.12592243, -0.32150713},
                                            {-0.31087995, -0.03336612, 0.22098841, 0.2563783, 0.36900008, -0.36375487, -0.10371325, 0.36094958, -0.11522388, -0.28827998, -0.2759779, 0.19398195, -0.20393471, -0.28252652, -0.29400423, -0.1416768}};
    float b2[LAYER_2_SIZE] = {-0.22312787,
                              0.13417931,
                              0.016149422,
                              -0.1886903,
                              0.010811105,
                              -0.08368328,
                              -0.023244286,
                              0.00026643364,
                              0.049304746,
                              -0.21146263,
                              -0.021612018,
                              0.052320033,
                              0.087697044,
                              -0.0845259,
                              -0.26996404,
                              -0.14131488};

    float w3[LAYER_2_SIZE][OUTPUT_SIZE] = {{-0.1620996, 0.04024708, -0.4601195, 0.08948969},
                                           {-0.3684273, -0.44127744, 0.23931149, -0.39439255},
                                           {-0.57973397, 0.04375093, 0.13331343, 0.2554413},
                                           {-0.095333256, -0.15875722, -0.03836681, 0.27984983},
                                           {0.12006102, 0.24929234, 0.31983247, 0.41976482},
                                           {-0.3419157, 0.28669012, -0.2569518, -0.34410867},
                                           {0.3968187, 0.060394116, -0.39974236, 0.20480083},
                                           {0.6078617, 0.39442334, -0.29162455, 0.49926683},
                                           {0.10802403, 0.24325, 0.114978604, -0.54601145},
                                           {0.26313546, -0.014306934, -0.19082534, -0.3214355},
                                           {-0.08980425, 0.030706154, -0.12915407, -0.36917695},
                                           {0.029870331, 0.0547429, -0.12482797, -0.4379642},
                                           {-0.24616435, 0.15406223, 0.5946607, -0.30673224},
                                           {-0.10486213, -0.398684, 0.2317782, -0.39020526},
                                           {-0.06530856, -0.26429978, -0.41091397, 0.42080066},
                                           {-0.059960723, 0.14375149, 0.24381088, 0.49892682}};

    float b3[OUTPUT_SIZE] = {0.24000001,
                             -0.21577272,
                             0.16171904,
                             -0.04980183};

    // Placeholder variable for temporarily storage of matrix operation results
    float sum;

    // Layer 1 logic:
    std::cout << std::endl;
    std::cout << "Layer 1:" << std::endl;

    float result1[LAYER_1_SIZE];

    // Matrix multiplication: input * weight 1 = result1
    // Matrix addition: result1 + bias 1 = result1

    for (int i = 0; i < LAYER_1_SIZE; i++)
    {
        sum = 0.0;
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            sum += input[j] * w1[j][i];
        }
        result1[i] = sum + b1[i];

        // Apply elu activation function
        if (result1[i] < 0.0)
        {
            result1[i] = 0.0;
        }
        std::cout << result1[i] << ", ";
    }
    std::cout << std::endl;

    // Layer 2 logic:
    std::cout << std::endl;
    std::cout << "Layer 2:" << std::endl;

    float result2[LAYER_2_SIZE];

    // Matrix multiplication: result1 * weight 2 = result2
    // Matrix addition: result2 + bias 2 = result2

    for (int i = 0; i < LAYER_2_SIZE; i++)
    {
        sum = 0.0;
        for (int j = 0; j < LAYER_1_SIZE; j++)
        {
            sum += result1[j] * w2[j][i];
        }
        result2[i] = sum + b2[i];

        // Apply relu activation function
        if (result2[i] < 0.0)
        {
            result2[i] = 0.0;
        }
        std::cout << result2[i];
        std::cout << ", ";
    }
    std::cout << std::endl;

    // Layer 3 logic:
    std::cout << std::endl;
    std::cout << "Layer 3:" << std::endl;

    float result3[OUTPUT_SIZE];

    // Matrix multiplication: result2 * weight 3 = result3
    // Matrix addition: result3 + bias 3 = result3

    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        sum = 0.0;
        for (int j = 0; j < LAYER_2_SIZE; j++)
        {
            sum += result2[j] * w3[j][i];
        }
        result3[i] = sum + b3[i];
        std::cout << result3[i] << ", ";
    }
    std::cout << std::endl;

    // Apply softmax activation function
    sum = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        sum = sum + exp(result3[i]);
    }

    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        result3[i] = exp(result3[i]) / sum;
        std::cout << result3[i] << ", ";
    }
    std::cout << std::endl;

    // Output the classification result
    int max = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        if (result3[i] > result3[max])
        {
            max = i;
        }
    }
    if (result3[max] < 0.3)
    {
        max = 0;
    }
    else
    {
        max += 1;
    }
    std::cout << std::endl
              << "Expected result: " << expected_result << std::endl;
    std::cout << "Actual result: " << max << std::endl;
}