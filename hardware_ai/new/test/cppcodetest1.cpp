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
    int expected_result = 1;
    float input[INPUT_SIZE] = {5.51, -13.105250000000002, -5.789, -1.4289999999999998, -0.08574999999999995, 0.9092500000000001, 10.145655227731721, 12.873724982983752, 9.040936566528934, 2.687293992104325, 2.5076930508935895, 4.401890155092469, 7.781499999999999, 9.28735, 6.4732, 1.9505, 2.0024, 3.0829875, -16.45, -53.58, -29.86, -8.76, -7.53, -8.75, 31.72, 10.79, 16.33, 1.8, 3.58, 8.71, 48.17, 64.37, 46.19, 10.56, 11.11, 17.46, 3.4000000000000004, -9.254999999999999, -4.0649999999999995, -0.5549999999999999, 0.425, 0.38, 3.825, 2.044999999999999, 3.175, 0.7399999999999999, 1.685, 1.195, 11.1675, 8.487499999999997, 7.869999999999999, 1.5950000000000002, 3.48, 2.37, 7, 36, 35, 23, 17, 17, 33, 4, 5, 17, 23, 23, 15, 28, 26, 30, 24, 13, 7, 7, 8, 8, 8, 5, 0.18837716324219778, -1.1148495201898665, -0.5359489408515978, -1.6911688314449382, -0.8261015522668979, 0.018709260771407686, 0.22161418984668213, 1.5625359541701993, 1.105812835135608, 1.7848819461142842, 0.2937533408364894, 0.1970396628270099, 53.317767999999994, 134.99214899999998, 46.100422, 3.705436, 2.518351, 8.081349, 20.13226737596954, 4.539360752217203, 12.3203, 2.6886};
    // Define the weights and biases
    float w1[INPUT_SIZE][LAYER_1_SIZE] = {{-0.0074744415, 0.106033884, -0.026708389, -0.16673818, -0.1930939, 0.04508832, 0.10306441, 0.07224594, -0.4496125, 0.10884783, 0.089768544, 0.16728519, -0.18486665, -0.18465099, -0.19857691, -0.05316081},
                                          {0.17508066, 0.07687401, 0.22994107, 0.0507748, 0.16480125, -0.04468718, 0.08420018, 0.05751643, -0.10939631, -0.1808263, -0.1503384, 0.13952231, 0.15429834, -0.16298728, -0.060319044, 0.13869052},
                                          {-0.16557655, 0.22405127, 0.09277677, -0.25113037, -0.10722479, -0.01812759, -0.367979, 0.19932364, -0.46889898, -0.010996401, 0.16449612, -0.17140481, -0.086388126, -0.21141346, 0.31556585, 0.12868781},
                                          {-0.40292367, 0.055620126, -0.2516217, -0.02878004, -0.0050419876, 0.18729618, -0.001572117, -0.09086413, 0.14354628, -0.047461197, -0.13700216, -0.060505234, 0.027074993, 0.28906575, 0.06233416, -0.026179135},
                                          {0.39063752, -0.13053866, 0.045056682, 0.24245761, 0.0675995, -0.10680627, 0.22784823, -0.09045349, -0.14066197, 0.20364514, -0.0886005, -0.059702974, -0.19724694, 0.040970128, -0.10550332, 0.105706975},
                                          {0.062303353, 0.18361917, 0.27314016, 0.03381179, -0.0061255447, -0.21443284, 0.05061404, -0.05658002, -0.08206859, -0.13622366, 0.19484153, 0.17035782, -0.20570129, 0.03382428, -0.08902687, 0.037775826},
                                          {0.17581525, -0.115781724, 0.026476452, 0.05748559, -0.14813776, 0.1782628, -0.046721525, 0.09411787, -0.2000198, -0.22310884, -0.22437818, 0.16997364, 0.05744481, -0.18372706, 0.1401343, 0.1724352},
                                          {-0.11355302, -0.12161845, -0.12911762, 0.15199302, 0.00335661, 0.09667733, -0.1784, -0.085010424, 0.20972855, -0.12088424, 0.001460617, -0.08957977, -0.18125884, 0.15539296, 0.12442288, -0.059069853},
                                          {0.1280271, 0.04525671, -0.026679812, 0.19553167, 0.06087957, 0.045463055, 0.060155023, 0.13774702, 0.2305994, 0.02027966, -0.0016889554, -0.17041898, -0.010961264, 0.16246204, -0.2090415, 0.12055255},
                                          {-0.06629376, 0.034551427, 0.06985386, 0.23506689, 0.043326005, -0.065098226, 0.02808165, -0.14106059, 0.019403819, -0.09663461, 0.11611654, 0.140216, 0.206141, 0.23070294, 0.044151876, 0.015416688},
                                          {0.12656182, -0.1145041, -0.11322109, 0.0056017437, 0.21177113, 0.15065607, 0.099139504, 0.095939234, 0.09595664, -0.008475721, -0.19400413, 0.1854621, 0.17326063, 0.030164028, -0.23440377, 0.23848395},
                                          {-0.043835703, 0.11872935, 0.13695258, 0.14418283, -0.0033469049, -0.20051384, -0.2769569, -0.22920935, -0.05180994, -0.028576866, -0.21269198, -0.04160484, -0.105500296, 0.16413707, -0.1480335, 0.18606393},
                                          {-0.0836807, 0.120164104, -0.14589934, 0.025388468, -0.112947606, 0.052410066, 0.031604007, 0.12634419, 0.077615835, -0.031903476, 0.22430062, 0.20628218, -0.10361647, -0.058240652, 0.008993957, -0.21453929},
                                          {-0.21097638, -0.16356412, 0.10405508, 0.015764965, 0.13790634, 0.18871018, 0.0028803947, -0.25265, 0.18155831, 0.17121515, -0.29223606, -0.0071964967, -0.16707763, -0.09683414, -0.15426181, -0.1763438},
                                          {0.007557888, 0.051912673, 0.07731377, -0.10122948, 0.0505075, -0.12092051, 0.069964334, -0.023797652, -0.017550081, -0.044279426, -0.053108692, -0.11560801, -0.11506536, 0.21745189, -0.26659086, 0.08722048},
                                          {0.114820786, -0.06345787, -0.08395631, -0.006259905, -0.04965038, 0.2112683, -0.056967776, 0.054048046, -0.12422562, 0.17154434, -0.13171986, 0.10446541, -0.101135805, -0.016312499, -0.13957994, 0.16234566},
                                          {-0.049979653, -0.15350574, -0.17356928, 0.028925424, 0.12387362, 0.11120042, -0.14414486, -0.120839275, -0.26995942, 0.19546792, -0.13877596, 0.045928, -0.0221989, -0.13012795, -0.16151185, 0.23027182},
                                          {-0.18582751, -0.111929245, -0.15420692, -0.016434664, -0.11686955, -0.17849144, 0.070692666, -0.026119744, 0.10020093, -0.030718043, 0.0576926, 0.15139073, 0.21768299, -0.045822736, 0.241422, -0.19369052},
                                          {-0.10560743, 0.1648644, 0.1832068, -0.0007752846, 0.011258611, 0.081814975, -0.04260805, -0.063788824, -0.203431, 0.052607983, 0.108105965, 0.20601504, 0.15567917, -0.1365567, -0.2167259, 0.14699748},
                                          {-0.06639227, -0.20430581, 0.16908973, -0.08981215, -0.21868594, -0.01658228, 0.13329293, 0.15904564, 0.1706992, 0.1943115, -0.14271095, -0.02869945, 0.20256117, 0.13674304, -0.018090244, 0.12354774},
                                          {0.18153827, -0.14480188, 0.104744114, 0.0966072, -0.07937241, 0.117103994, -0.2053911, -0.08743129, 0.05928037, 0.14997602, -0.07617652, 0.09754406, -0.12051785, -0.05798086, 0.23161387, -0.12526734},
                                          {0.013573795, -0.12956274, 0.20220378, -0.083456025, -0.012527449, 0.086610705, 0.17074166, -0.028367974, -0.27608657, 0.121887416, 0.03189783, -0.02461202, -0.023249537, 0.08687655, 0.23154625, -0.021545995},
                                          {0.0195711, -0.13700289, -0.14580418, 0.23797075, 0.08456618, -0.06639846, 0.001605036, -0.041768383, 0.11846827, -0.1867891, -0.094969265, 0.23202373, 0.18174154, -0.0102498755, -0.14394984, -0.049192864},
                                          {-0.098884426, 0.11512355, 0.18767034, 0.06109176, -0.20933446, -0.011349171, -0.01286268, 0.073172145, 0.0868355, -0.17937225, 0.14288333, 0.14666627, -0.16163947, -0.1190183, -0.011429098, 0.02579507},
                                          {-0.0696191, 0.14090766, 0.038867947, -0.13447435, -0.14218822, -0.10405823, -0.30515435, -0.19575639, -0.10446055, -0.03297992, -0.17182326, -0.09302129, -0.04888457, -0.20630418, 0.26093087, -0.011229513},
                                          {-0.12965038, 0.12491662, -0.26567385, -0.1689025, 0.09640717, 0.1575824, 0.0010045471, 0.031509805, 0.048927065, 0.05311036, -0.01942503, 0.21055382, 0.21626878, 0.089280754, 0.13859688, -0.33717915},
                                          {0.02956263, -0.114003986, 0.040453013, 0.0155995, 0.07560952, 0.039485097, -0.24341054, -0.07104916, 0.13942097, 0.20511767, 0.028192673, -0.14659289, 0.079885215, -0.1687827, -0.043716364, 0.1785603},
                                          {-0.2862979, -0.17483248, 0.13511519, 0.15531334, -0.034718346, -0.00952661, -0.004400959, -0.13161944, 0.06459436, 0.067053914, -0.2159517, -0.12861761, -0.16475785, 0.03977821, 0.04891109, -0.21059884},
                                          {0.19304484, 0.15889937, -0.19673888, 0.1465022, 0.080982886, 0.016734421, 0.0576485, -0.16042437, -0.017298475, -0.07627724, -0.08071624, -0.11486536, -0.017091975, 0.1137517, -0.20671442, 0.36303604},
                                          {-0.09677946, -0.11651732, -0.00055289175, -0.114044346, -0.18928026, -0.22235286, 0.018885462, -0.060912948, -0.28645048, -0.16947192, -0.05414757, 0.20763832, 0.15106043, -0.1915318, -0.16527078, -0.17802478},
                                          {-0.08239516, -0.16467465, -0.06441121, 0.08870602, -0.18102905, 0.1400035, 0.07987968, 0.032918934, 0.018572077, -0.048598588, -0.010297153, -0.076448426, 0.06607729, -0.014148972, -0.091585405, 0.119053416},
                                          {-0.101320855, 0.15019095, 0.07528375, -0.21052897, -0.089650504, 0.07864031, 0.05375339, -0.1091929, 0.03726969, -0.21375808, 0.14358145, 0.080706544, 0.10389063, 0.071754724, 0.16012388, -0.008375489},
                                          {0.08451599, -0.16828889, 0.012247951, 0.16957487, -0.211274, -0.037912726, 0.21317092, 0.19334789, 0.08584435, -0.17072465, 0.11044179, -0.111952476, -0.21306331, -0.014270453, -0.11868255, -0.17158951},
                                          {-0.27410546, 0.038904145, -0.1259582, -0.022895765, -0.080205604, -0.2022869, 0.19020908, -0.070153706, 0.29812175, 0.1517778, 0.09693913, -0.21801786, 0.17215082, 0.15256551, -0.14989395, 0.13210563},
                                          {0.28448263, 0.13725509, -0.15424176, -0.07919793, 0.16889998, -0.011677712, 0.1331622, -0.14008167, 0.05050639, -0.023493692, -0.14342779, 0.084532864, 0.13165057, 0.13944758, -0.16925924, -0.05311975},
                                          {0.10532305, 0.006817034, -0.21192564, -0.23016255, 0.09253033, -0.18600291, 0.09248313, 0.1251203, -0.21312454, -0.17215478, -0.23317395, 0.17432222, 0.0794805, -0.15833624, -0.004366091, -0.13602933},
                                          {0.076414704, -0.20537172, -0.1577225, -0.15605141, 0.1944312, -0.12775803, 0.05228457, -0.18284602, -0.30330253, -0.04094316, 0.28861696, -0.11964968, -0.18153852, -0.17531632, -0.13559882, 0.14370367},
                                          {-0.06629404, -0.11278571, 0.17177203, 0.16458738, -0.03278103, -0.18346997, -0.036039438, 0.15175009, 0.012656183, 0.2074596, 0.11086807, 0.17636806, 0.022913992, -0.12178572, -0.06528586, 0.084479675},
                                          {-0.04138554, 0.10688618, 0.20290561, 0.049063116, 0.18428572, -0.116385266, -0.24444862, 0.15141329, -0.042291153, -0.16777907, 0.13449445, 0.07594932, 0.039836854, -0.16505958, 0.003301718, 0.18319365},
                                          {-0.30301395, 0.0325456, -0.044220917, 0.2749251, 0.017247459, -0.11751485, 0.28717357, -0.28938785, 0.24217665, 0.18345642, -0.112975486, 0.121043794, 0.16097307, -0.06246042, 0.008931646, -0.2877976},
                                          {0.16998239, -0.20605649, -0.09619106, 0.22430436, -0.117458045, 0.030550808, 0.07959536, 0.18309358, -0.053251814, 0.0243617, 0.38355613, 0.20421182, -0.22254601, 0.008140708, 0.03437791, 0.31768},
                                          {0.044624787, 0.22220686, 0.36667755, 0.13541147, -0.1748126, -0.069607764, 0.012712773, 0.095022246, 0.17949255, -0.12016486, 0.17484736, -0.15715957, -0.168499, 0.05323703, -0.23167437, 0.10382405},
                                          {-0.1525018, 0.03960367, -0.0063621714, 0.08088981, 0.11183361, -0.0073318183, -0.22304706, -0.07940019, -0.20076272, 0.20825234, 0.0054653403, 0.19774738, -0.0056916177, -0.052631237, -0.14593792, 0.03486654},
                                          {0.08394923, -0.08084694, 0.16788223, -0.15339856, 0.092382796, 0.10837728, 0.036198646, -0.14401847, -0.036855545, 0.12355572, 0.11192015, 0.17377332, 0.04194647, -0.017274087, 0.2767242, 0.20669994},
                                          {-0.04489444, 0.13926858, -0.14879191, 0.14554921, 0.16503559, 0.14307636, -0.11549941, -0.12344105, -0.015198107, -0.083001494, -0.096638784, -0.15981066, -0.071239516, -0.21025437, -0.25454062, -0.17008466},
                                          {-0.06550191, -0.18634307, 0.019614264, -0.23507276, 0.10765292, -0.08725272, 0.12903306, 0.0539868, -0.19682968, 0.021154985, 0.23654121, -0.1343158, 0.20851162, 0.11213413, 0.10871399, -0.13696884},
                                          {0.28078032, -0.236249, -0.22820294, -0.14667648, -0.1913239, 0.066361696, -0.038422175, 0.0009849574, -0.22454457, 0.06668073, -0.06558518, 0.13315518, -0.0715545, -0.34443408, -0.056594692, 0.28737625},
                                          {0.11330125, -0.13591711, 0.117824554, -0.242467, -0.19576879, -0.1259576, 0.053797547, 0.2148201, -0.23742084, -0.0870201, 0.06543731, -0.11382467, -0.016073003, -0.22844584, 0.08102526, 0.0062408154},
                                          {-0.10890871, 0.11003913, -0.075964436, 0.009109752, 0.19912699, -0.10697686, 0.10064163, 0.023239704, -0.20149028, 0.220063, 0.029320514, 0.09137677, -0.17136952, -0.08806105, -0.025526302, 0.098261714},
                                          {-0.07997138, 0.043312997, -0.1464986, -0.11708545, 0.21588372, 0.03408158, -0.076102674, 0.17716159, -0.17716081, -0.05337979, 0.02434353, -0.10882219, 0.21595755, 0.11150936, -0.037506178, 0.04256099},
                                          {-0.0874559, -0.17569979, 0.10825313, -0.04117568, -0.13022432, 0.19755161, -0.02737892, -0.13939431, -0.14314315, -0.20811835, 0.18763338, 0.107577756, 0.018102378, 0.08094613, -0.21661885, 0.22586016},
                                          {0.1884172, 0.02235227, 0.26210496, -0.05924055, 0.09439954, -0.02400449, 0.051711056, 0.004033688, 0.09758969, -0.07179558, 0.0027230706, -0.09158305, -0.06624098, -0.10289336, 0.06768994, 0.15740292},
                                          {0.22433366, 0.18351606, -0.16073404, -0.100986585, -0.10106252, -0.020983323, -0.14808661, 0.198782, -0.041720964, -0.16110814, -0.06842857, -0.05263322, -0.0073580593, -0.18095295, -0.011134143, 0.08587549},
                                          {-0.026553936, 0.104541466, 0.16731468, -0.2468072, 0.17426789, -0.15586239, 0.04508836, 0.116857715, -0.021998247, 0.19002575, -0.18220782, -0.21437202, 0.21062094, 0.051899813, 0.035820395, 0.112768136},
                                          {0.102302745, 0.14935254, -0.15672438, 0.069784276, -0.044848736, -0.117827065, 0.17563336, -0.12279147, 0.071198694, -0.18173812, 0.09996805, 0.029852804, -0.10515246, 0.20177442, 0.052043177, -0.2139706},
                                          {0.17617072, 0.19729291, 0.16979764, -0.12144502, -0.0902193, 0.08762622, 0.17371345, -0.0919947, -0.07271302, -0.15475202, 0.11771049, -0.15684806, 0.052181244, 0.15865783, 0.08881023, -0.12577578},
                                          {0.0019158572, -0.19474007, 0.16917883, 0.0822354, -0.13137352, -0.1498738, -0.15974669, -0.011966985, 0.22712761, 0.103824526, 0.028474445, 0.048565056, -0.15646315, 0.042455465, -0.23572499, 0.1277993},
                                          {-0.16038546, -0.09721789, -0.121431746, 0.03454946, -0.11730158, 0.12944236, 0.115997605, 0.22207499, 0.15119046, 0.096874654, -0.004202346, -0.10159312, -0.09575912, 0.16488896, -0.15364106, 0.18082388},
                                          {-0.22093785, -0.15312992, 0.037213918, 0.041960776, 0.18999186, -0.073025525, -0.17754865, -0.029003024, 0.04455145, -0.040761143, -0.038704954, 0.033483714, -0.18641669, -0.090456866, 0.25732112, -0.0105418805},
                                          {-0.03615029, -0.21023196, -0.11918764, -0.022197649, -0.2074232, -0.0661968, 0.1670394, -0.16410361, -0.2371458, 0.07341397, -0.06567582, 0.12925008, -0.18731387, -0.083170235, -0.085250415, -0.19736724},
                                          {0.18656468, -0.22400649, -0.014442349, -0.14652058, -0.037374694, -0.12871529, -0.05696281, -0.17781828, 0.09000139, -0.10213888, -0.15299647, -0.17144062, 0.13622713, -0.16226941, -0.15678105, -0.15367119},
                                          {-0.13573194, -0.15435278, -0.059233453, 0.16973451, -0.05021333, -0.13214828, 0.069941975, -0.1811794, 0.009020709, -0.032752067, -0.22633117, -0.11609598, 0.0038852692, -0.1987548, 0.011974044, -0.22265866},
                                          {-0.16133189, -0.02068804, -0.08871143, 0.12499272, -0.17758489, -0.121610396, 0.10096549, -0.08666028, -0.07836636, -0.0057331473, 0.1572142, -0.15222138, -0.16272718, 0.10366993, -0.12411687, -0.08254346},
                                          {0.047252968, -0.016376771, 0.027873415, -0.0821833, 0.20496285, -0.0680875, -0.08969593, -0.012440581, 0.05161116, -0.2025263, -0.22401065, -0.0651903, -0.15958303, -0.049890872, 0.03088999, 0.13687167},
                                          {0.19478415, -0.08764548, -0.05319284, -0.086723626, 0.013576532, -0.051864892, -0.18920265, 0.21827365, -0.14574796, 0.2112261, -0.12306626, -0.010226239, -0.21375488, -0.08085408, -0.022622447, 0.15005694},
                                          {0.15973744, 0.21068025, 0.1846775, -0.09948449, -0.232878, -0.055881277, 0.1775427, 0.20583607, -0.077749155, 0.08696002, 0.15124747, -0.0758743, 0.022401154, 0.04764498, 0.015285193, -0.08052602},
                                          {0.0701962, -0.1743454, -0.09651622, 0.026872251, 0.094202146, 0.01416342, 0.10533381, 0.016481997, 0.18908286, -0.10108766, -0.11606779, 0.08114836, 0.17775753, -0.14205834, 0.26870376, 0.08398069},
                                          {-0.03251318, 0.13670528, 0.10881132, 0.0018569448, 0.038647074, -0.12673143, -0.053166684, 0.15104932, 0.14752492, -0.019198552, 0.13707966, -0.025873715, 0.074118406, 0.19695844, 0.1741092, 0.20421106},
                                          {0.018827155, -0.06062241, 0.117699884, -0.069706134, 0.11218866, 0.09992081, -0.1699635, -0.029203977, 0.21428683, 0.1349256, -0.21247773, -0.024343941, 0.21411774, -0.16834237, -0.01860613, -0.10333715},
                                          {0.14410304, 0.024259795, 0.19666323, 0.16691956, 0.064833045, -0.13417509, -0.019065738, 0.0037185275, -0.13735078, 0.08746499, -0.22573693, 0.10269026, 0.08715695, -0.18298876, -0.0006246859, 0.07413004},
                                          {-0.14284936, -0.20331486, -0.12491484, -0.1700262, -0.09156605, -0.10811229, -0.080608934, -0.23286292, -0.11894799, 0.19136581, 0.07168295, -0.08233564, 0.21204033, 0.14778519, -0.16937622, 0.23058142},
                                          {-0.16855119, -0.14822839, 0.069418214, 0.10655419, -0.075192146, -0.17123955, 0.19077033, 0.060798537, 0.018398805, -0.032146826, -0.11305909, -0.21003136, 0.19349128, -0.14675565, -0.0671464, 0.07426071},
                                          {0.08719982, 0.06235383, -0.020025974, -0.0662692, 0.07483276, -0.12565666, -0.16367422, 0.077604175, -0.034854073, 0.17345843, -0.102374755, -0.21459565, -0.054161474, 0.20530942, 0.12110133, -0.080287695},
                                          {-0.1556493, 0.10927671, -0.06818259, 0.029900953, 0.18723544, -0.18557405, -0.09120517, -0.038286045, 0.23572192, -0.018768504, 0.11640027, -0.19702737, 0.114931494, 0.044854447, -0.17460814, -0.13933301},
                                          {-0.087791525, -0.113593355, -0.113219425, 0.19399282, -0.1503528, 0.035904616, -0.05827143, 0.16269623, 0.04973466, -0.17665409, -0.095144, 0.0037690646, 0.025622189, -0.07632774, 0.24274032, 0.2127128},
                                          {0.17906396, 0.13010505, -0.24952933, 0.012053887, -0.0020577686, 0.024546772, 0.15691464, 0.05877553, -0.051307306, 0.04037437, 0.03178623, 0.13712308, 0.06574148, -0.07800291, 0.21654767, 0.052864384},
                                          {-0.10074042, -0.053060893, 0.15527017, -0.114919305, -0.04937572, 0.15014887, -0.102768056, -0.0010566419, -0.081635036, -0.18478602, 0.13629803, 0.13656592, 0.08746135, -0.12720084, 0.0016423331, -0.09699949},
                                          {0.23503263, -0.171187, 0.052068703, 0.12271425, -0.065823205, 0.025829166, -0.10879777, 0.010525859, 0.13359816, 0.011831656, -0.07087397, -0.10397321, 0.14781418, 0.17375349, 0.20513962, 0.23124523},
                                          {0.09620662, 0.16092758, -0.12711474, -0.13203849, 0.09273559, -0.2231903, -0.20007224, 0.02430749, 0.12877038, 0.017962426, 0.0036199286, -0.22126986, 0.15516984, -0.103453346, 0.1392102, -0.048600223},
                                          {-0.00593167, 0.045017675, -0.13633825, -0.085851274, -0.11011825, -0.1008219, -0.09883843, -0.13894291, 0.07905762, -0.020295873, 0.082064316, -0.07278549, -0.0679705, -0.08822053, 0.04620612, -0.2506842},
                                          {0.075013354, 0.12455083, -0.20894594, -0.237573, 0.03850939, 0.04102108, -0.27270517, -0.019002376, -0.12724467, -0.1279496, -0.18890934, 0.027548246, 0.19988123, -0.115076266, 0.18964987, 0.16667782},
                                          {0.21657032, -0.13302553, 0.06018824, -0.019289324, -0.05639592, 0.06901577, -0.17858541, 0.0026058033, -0.14470938, 0.1880543, 0.109016255, -0.04802947, 0.21312079, 0.13813214, 0.026056085, -0.10189803},
                                          {0.07865955, -0.010687151, -0.0040483717, 0.07961197, -0.008409729, 0.22281489, -0.05765366, 0.090179, -0.10344769, -0.22299877, -0.04327231, -0.14304464, 0.21568555, -0.015932761, -0.114740685, 0.29704192},
                                          {-0.10625068, -0.15740816, 0.16178276, -0.2971748, -0.10022456, 0.028957903, 0.003608254, 0.15067808, -0.14923601, -0.2211593, -0.24867453, 0.21580096, 0.16503543, -0.2942211, 0.046439018, -0.047029562},
                                          {-0.07512172, 0.017450515, -0.05238587, -0.11159334, 0.21009742, 0.21662664, -0.028631654, -0.21056964, 0.10186036, -0.039516553, -0.15281701, -0.22616442, -0.1623095, 0.2699464, 0.2947855, 0.21124111},
                                          {0.23622736, 0.20686801, -0.042696793, 0.22549725, 0.12150914, -0.08036895, 0.19477572, -0.05459936, 0.12080607, -0.041728437, 0.07960481, 0.030763596, 0.0042952597, 0.17198758, -0.37004, 0.011702083},
                                          {-0.2542703, 0.0938982, -0.1506907, 0.1484482, -0.13386194, -0.12736985, 0.03265814, -0.085618466, 0.10559602, 0.2038154, -0.11624465, -0.18847547, 0.013911724, 0.38715118, 0.21032412, -0.19375274},
                                          {-0.13963735, -0.17494042, -0.0035377848, 0.010449307, -0.083195105, 0.18056262, -0.057262737, -0.08184818, -0.08777973, 0.0006928593, -0.13422778, -0.12664266, -0.098545775, -0.01548534, 0.2016444, -0.36736047},
                                          {0.14631647, -0.15155317, 0.10390614, 0.03718302, 0.13633415, 0.22189566, 0.08018887, -0.10339525, -0.0707307, -0.103488445, 0.18560898, -0.021560345, -0.021554396, -0.06646646, 0.051921368, -0.22317918},
                                          {-0.12537076, -0.092251524, -0.13896923, 0.06947374, 0.12101187, 0.02956134, -0.030694803, 0.004884837, 0.42570534, 0.17827618, -0.011281059, 0.21204887, -0.0020656437, 0.21860637, -0.26375374, 0.07280262},
                                          {0.06595043, -0.18242316, 0.17582545, 0.027113987, -0.0022585855, -0.21206208, 0.094939366, 0.022877049, -0.20774512, 0.11783287, -0.11116181, -0.13627051, -0.20117298, -0.123971574, -0.13340965, 0.06767061},
                                          {-0.045718364, -0.17621045, 0.20286627, 0.1354647, -0.08122406, -0.02295667, -0.02243143, 0.024462165, -0.0140479505, 0.10667634, 0.07234054, 0.11641493, -0.22524287, -0.012172159, 0.13413776, -0.12965763},
                                          {0.037929, 0.031082513, -0.09385836, -0.034854762, -0.1943029, -0.03967282, -0.17676054, 0.0881409, -0.09537003, -0.21214135, -0.02743656, -0.0061759953, 0.20896289, -0.039526764, 0.028976046, 0.098006524},
                                          {-0.010475648, 0.07148586, 0.00630785, 0.0958336, -0.09958284, 0.11488074, -0.15736136, -0.06300208, -0.13068646, -0.16580984, -0.019079624, 0.040636025, 0.04640183, -0.017833563, -0.14451948, 0.10273017},
                                          {-0.035470974, -0.13522369, -0.08248737, -0.1556296, 0.14240736, -0.18264788, 0.1715161, -0.10024676, 0.03959565, 0.21231428, 0.11320895, 0.09091229, -0.06953853, -0.20803833, -0.17243227, 0.29454917},
                                          {-0.340731, 0.14227015, -0.04299577, -0.09454011, 0.1771962, -0.10622158, -0.23493196, 0.1357974, -0.25190228, -0.21721992, -0.036294665, -0.113693215, -0.18771365, 0.04029677, 0.16275148, -0.15396154},
                                          {0.1738272, 0.017102594, 0.20288078, -0.22807916, 0.18485323, 0.013673574, -0.2032531, 0.028456124, 0.08599297, -0.119694084, 0.19091372, -0.000762644, 0.095799744, 0.10565582, -0.18106396, 0.2186017},
                                          {0.25958106, 0.1134092, 0.19576532, -0.22616869, 0.069568664, 0.04588172, -0.22328861, -0.20974456, -0.017422982, -0.0058229417, 0.14016844, 0.14859587, 0.08933571, -0.08593586, 0.089522384, 0.023446888},
                                          {-0.18566553, 0.20489644, -0.08066737, 0.10340041, 0.0034948597, -0.21319908, 0.06262394, 0.21155258, 0.07953283, 0.18998665, -0.22975136, -0.070754655, -0.1609005, 0.061073042, -0.2003879, -0.13572916},
                                          {0.23070212, 0.14411591, -0.18228498, -0.21181968, 0.12000929, -0.068687975, -0.031967606, -0.080805086, 0.17774782, -0.096472815, 0.037891652, 0.19055055, -0.09151468, -0.044807054, -0.21532667, -0.097372524}};
    float b1[LAYER_1_SIZE] = {0.027148787,
                              -0.013978068,
                              -0.02306449,
                              -0.017106675,
                              -0.007628223,
                              0.0,
                              -0.013779657,
                              -0.010468639,
                              -0.024337832,
                              0.0,
                              -0.04728805,
                              -0.017949358,
                              0.0,
                              0.005333029,
                              0.02227887,
                              0.03986374};

    float w2[LAYER_1_SIZE][LAYER_2_SIZE] = {{-0.290493, 0.23336919, 0.5216523, -0.20183715, -0.26257095, -0.3413794, 0.45338625, 0.5516953, 0.22715886, -0.07332323, 0.44800848, -0.29911473, -0.033168245, -0.22236213, 0.25559527, 0.058342613},
                                            {-0.35169834, 0.22999784, -0.19627412, -0.085003495, 0.31774417, 0.13988484, -0.0645552, -0.19519342, 0.3846351, -0.28702956, -0.22492152, -0.32717928, -0.39279097, 0.14402705, -0.3833004, 0.4143851},
                                            {0.20092732, -0.16007397, -0.088163294, 0.43646586, 0.18729772, 0.3369691, -0.25550893, -0.2247728, 0.1800111, 0.030304706, 0.053051405, 0.2251941, -0.08139625, 0.0024841803, 0.3293715, 0.22383678},
                                            {0.2752892, -0.37914366, -0.34671292, 0.46450415, -0.1986283, -0.06506222, 0.21808828, -0.2413113, 0.24969923, 0.34484994, 0.24059542, 0.2519771, 0.35576248, -0.029789079, -0.16835672, -0.35286286},
                                            {0.113393575, -0.008938462, -0.017343938, 0.09954646, -0.07970914, -0.034151055, -0.14980274, 0.045733303, -0.3057009, 0.083454, -0.3662931, -0.08060087, -0.030055717, 0.41766945, 0.36635575, -0.14854231},
                                            {-0.41979933, 0.22669831, 0.23348793, -0.10133064, -0.21366027, -0.15168428, 0.2271274, -0.2349128, -0.40397054, -0.05250728, -0.20907836, -0.42891982, -0.23068374, -0.10583976, -0.22499739, -0.26418602},
                                            {-0.13938826, -0.04352379, -0.028255211, 0.10541761, -0.14291428, -0.20607445, -0.33793935, -0.3571671, 0.18385676, -0.21721397, -0.25557363, 0.2040068, 0.25811967, 0.39006987, -0.07665007, 0.03940765},
                                            {0.44123304, -0.21855773, 0.09166994, -0.4159282, -0.07464492, 0.10298892, 0.17349786, 0.28654426, 0.2418441, 0.22184779, 0.13840008, 0.060286194, -0.38942426, 0.50170547, 0.09146002, 0.3480675},
                                            {0.3801778, 0.014443967, -0.036945105, -0.077899665, -0.08001419, 0.0011844988, 0.015446999, -0.2599514, -0.009190367, -0.11501578, 0.40076676, -0.19957396, 0.5028748, 0.21461977, -0.22497793, 0.10697482},
                                            {-0.2564059, -0.36279178, 0.36494026, -0.2679785, 0.29586634, 0.21250191, -0.41659015, -0.39555806, -0.31767768, -0.21809384, 0.36732516, -0.2895082, -0.4275456, -0.26268122, -0.043979496, 0.34022346},
                                            {0.3525486, -0.3755044, -0.03336902, -0.23578276, -0.1775323, 0.17189865, -0.3728316, -0.11981129, -0.45440158, 0.34260887, 0.39816803, -0.21308175, 0.08070498, -0.15722647, 0.20409665, 0.37415734},
                                            {0.29233003, 0.414491, 0.0984312, -0.3923442, -0.36871302, -0.047032088, 0.06820944, 0.42930883, 0.2597736, -0.00012884871, 0.14382419, 0.07315268, 0.15503863, 0.23916784, -0.2431437, -0.38353986},
                                            {0.23112074, -0.15707994, -0.010885239, 0.21814564, -0.4268148, 0.38632587, -0.25096986, -0.104023606, 0.35929695, 0.18435827, 0.29787174, -0.4088156, -0.105066836, -0.3772543, 0.077810496, 0.29931036},
                                            {0.47616133, -0.10293469, -0.16499442, 0.1046895, -0.22844198, -0.4579308, -0.3733521, -0.27854756, -0.12803373, -0.50543916, -0.09968734, 0.5186329, -0.05312087, 0.13594314, -0.009324435, -0.45641473},
                                            {-0.046843227, 0.4383116, -0.113399446, 0.16628623, -0.47945687, 0.17002097, 0.0024268997, 0.5021783, 0.0758999, -0.3245717, -0.03879106, -0.20987299, 0.2823674, -0.21980214, -0.16553262, -0.06276125},
                                            {0.47895736, -0.32830766, 0.03171298, 0.028930502, -0.3167476, 0.19029438, -0.2632185, 0.23111096, 0.498232, 0.016804578, -0.044439472, 0.14527884, -0.23575026, 0.19066583, 0.018537322, 0.34377235}};
    float b2[LAYER_2_SIZE] = {-0.09300699,
                              -0.001522365,
                              0.013893657,
                              0.07346713,
                              0.082291946,
                              0.10660192,
                              0.058674987,
                              0.22099395,
                              -0.027197002,
                              0.05195217,
                              -0.047633708,
                              -0.102807425,
                              -0.12027226,
                              -0.025867274,
                              0.04059894,
                              -0.057174478};

    float w3[LAYER_2_SIZE][OUTPUT_SIZE] = {{0.16538608, 0.33040625, 0.054056447, 0.20515883},
                                           {-0.48914382, -0.18632169, 0.30466, -0.29564542},
                                           {-0.41868058, -0.5299284, -0.31750253, 0.21150893},
                                           {0.26625094, -0.2361063, 0.1875127, -0.5444843},
                                           {-0.24816947, -0.30203873, 0.21061991, -0.19342008},
                                           {0.5550761, -0.2678891, 0.5071391, 0.05185537},
                                           {0.504216, 0.018808288, 0.039834596, 0.47030485},
                                           {0.021526983, -0.5698145, 0.2890704, 0.36273468},
                                           {0.06464504, 0.10387023, 0.3034834, 0.5412283},
                                           {0.33994442, 0.08602458, -0.2862473, -0.41275597},
                                           {-0.26463932, -0.08248844, -0.45493463, -0.26736444},
                                           {-0.0623757, 0.31524917, 0.05355641, 0.10106367},
                                           {-0.06542428, 0.40053564, 0.009903532, -0.38463938},
                                           {-0.14006364, 0.56431955, -0.44876784, 0.3394511},
                                           {0.27672058, -0.4872197, -0.13196808, 0.30280915},
                                           {0.02595778, -0.41166544, 0.26492056, 0.29734963}};

    float b3[OUTPUT_SIZE] = {0.059982374,
                             -0.028793443,
                             0.038082995,
                             -0.12811448};

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