from ai_interface import OL

data = [1.9115000000000002,-7.461,0.4914999999999997,-1.7954999999999999,-0.9164999999999999,0.48150000000000015,5.097055301838504,10.801549842499455,7.246225413965536,3.612189744462492,1.8300034836032417,3.3470991544918416,4.5095,8.402800000000003,6.58605,2.95515,1.4877999999999998,2.44735,-7.72,-28.74,-12.96,-8.76,-5.98,-6.82,10.71,11.4,8.36,2.37,1.86,6.86,18.43,40.14,21.32,11.129999999999999,7.840000000000001,13.68,0.935,-10.655,4.295,-0.71,-0.31,0.815,3.5199999999999996,3.5050000000000003,2.9,1.94,1.1,1.19,7.85,10.0775,13.3925,4.5225,2.285,2.2325,10,15,7,12,12,8,10,5,13,8,8,12,10,8,13,13,12,11,7,4,6,5,7,4,0.10923658920611898,0.006123064316109655,-0.6698011636773273,-0.8947177887388064,-0.8887391083578259,-0.11782114766905882,-1.0305975650450216,-0.27395891164450115,-1.2182079198404148,-0.5161646380695504,0.72656307638605,0.06782663364040076,5.926761,34.468,10.549871000000003,3.254347000000001,0.837777,2.286983,14.627902077504405,4.49440426778172,4.550599999999999,1.3639000000000001]
xresult = 1

dma = OL.load_overlay()
action = OL.feed_overlay(data,dma)
print(action)