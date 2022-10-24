from FilamentAnalysis import FilamentExp
import pickle

datapath = "C:/Kim_Min_Jung/2022년 2학기/핵융합 플라즈마 실험/2. 필라멘트/실험데이터/22.10.24(월)/1024/"
nametype = "1024_shot{0}.dat"

dataholder = []
for i in range(-8,20):
    #print(i)
    temp = FilamentExp(datapath+nametype.format(i), "거리: {0}".format(26-i), -54.51, -57.27, -51.75, -30)
    temp.calculate()
    print(temp.nIon)
    #temp.breif()
    #dataholder.append(temp)

"""
with open("10_24_analysis", "wb") as fp:   #Pickling
    pickle.dump(dataholder, fp)
"""