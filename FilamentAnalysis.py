import pandas as pd
import numpy as np
from scipy import stats
from linearFitting import linearFitting

class FilamentExp:
    def __init__(self, datapath, filamentV, ionCut, peCutL, peCutR, cutBK_L = 0, cutBK_R = 2, cutRM_L = 6, cutRM_R = 8) -> None:
        """ parameter """
        self.filamentV = filamentV
        self.cutIon = ionCut
        self.cutPE_L = peCutL
        self.cutPE_R = peCutR
        self.cutBK_L = cutBK_L
        self.cutBK_R = cutBK_R
        self.cutRM_L = cutRM_L
        self.cutRM_R = cutRM_R

        """ make pandas df """
        colName = ['Vb','I']
        self.data = pd.read_csv(datapath, delimiter='\t', names=colName, header=None)
        self.data = self.data.dropna().reset_index(drop=True)

        """ 사용할 축들 """
        self.dataV = data['Vb'].to_numpy()
        self.dataI_original = data['I'].to_numpy()
        # Ion 제거
        self.dataI_del_Ion = None
        # Ion, Primary electron 제거
        self.dataI_del_Ion_PE = None
        # Ion, Primary electron 제거하고 ln씌운거
        self.dataI_del_Ion_PE_ln = None

        """ 얻게될 결과 및 람다함수들 """
        # result dictionary 구성
        # dict{'polynomial': array([기울기, y절편]), 'determination': 결정계수}

        # Ion curve
        self.resultIon = None
        self.curveIon = None
        # Primary Electron curve
        self.resultPE = None
        self.curvePE = None
        # Bulk Electron curve
        self.resultBE = None
        self.curveBE = None
        # rightmost curve (플라즈마 전위구할때 씀)
        self.resultRM = None
        self.curveRM = None

        """ characteristic values """
        self.plasmaV = None
        self.tempBE = None
        self.nIon = None
        self.nPE = None
        self.nBE = None

    
    def calculate(self):
        self.linearFit_Ion()
        self.linearFit_PE()
        self.linearFit_BE()
        self.linearFit_RM()
        self.calcCharacteristics()

    
    def linearFit_Ion(self):
        self.resultIon = linearFitting(self.dataV,self.dataI_original,xTo=self.cutIon)
        self.curveIon = np.poly1d(self.resultIon['polynomial'])
        self.dataI_del_Ion = self.dataI_original - self.curveIon(self.dataV)

    def linearFit_PE(self):
        self.resultPE = linearFitting(self.dataV,self.dataI_del_Ion,xFrom=self.cutPE_L,xTo=self.cutPE_R)
        self.curvePE = np.poly1d(self.resultPE['polynomial'])
        self.dataI_del_Ion_PE = self.dataI_del_Ion - self.curvePE(self.dataV)

    def linearFit_BE(self):
        """ BE 전자의 linear fitting은 전류에(y축에) ln을 적용하고 이뤄짐 """
        # abs를 취하는 이유는 ln 오류방지하기 위함. 큰 상관없음
        self.dataI_del_Ion_PE_ln = np.log(np.abs(self.dataI_del_Ion_PE))
        self.resultBE = linearFitting(self.dataV,self.dataI_del_Ion_PE_ln,xFrom=self.cutBK_L,xTo=self.cutBK_R)
        self.curveBE = np.poly1d(self.resultBE['polynomial'])
    
    def linearFit_RM(self):
        self.resultRM = linearFitting(self.dataV,self.dataI_del_Ion_PE_ln,xFrom=self.cutRM_L,xTo=self.cutRM_R)
        self.curveRM = np.poly1d(self.resultRM['polynomial'])

    def calcCharacteristics(self):
        """ 상수 """
        e = 1.6 * 10^(-19)
        k = 8.617 * 10^(-5)
        me = 9.109 * 10^(-31)
        # 타겟기체 질량, 여기선 Ar
        M_ion = 40 * 1.66 * 10^(-27)
        Cs = np.sqrt(self.tempBE/M_ion)
        # 탐침 면적
        Area = np.pi * (0.005) ^ 2

        """ 플라즈마 전위 """
        coeffBE = self.resultBE['polynomial']
        coeffRM = self.resultRM['polynomial'] 
        self.plasmaV = - (coeffBE[1] - coeffRM[1])/(coeffBE[0] - coeffRM[0])

        """ 전자 온도 """
        # Bulk Electron ln 상에서 linearfit 한거 기울기 역수가 온도(eV)
        self.tempBE = 1/coeffBE[0]

        """ 이온 밀도 """
        IonIsat = self.curveIon(self.plasmaV)
        self.nIon = IonIsat/(0.6 * e * Area * Cs)

        """ 주 전자 밀도 """
        PEIsat = self.curvePE(self.plasmaV)
        vp = np.sqrt(2*e*(self.plasmaV - self.filamentV) / me)
        self.nPE = 4*PEIsat/(e * vp * Area)

        """ 벌크 전자 밀도 """
        BEIsat = np.exp(self.curveBE(self.plasmaV))
        vth = np.sqrt(8*self.tempBE / np.pi * me)
        self.nBE = 4*BEIsat/(e*vth*Area)
        


defaultPath = 'C:/Kim_Min_Jung/2022년 2학기/핵융합 플라즈마 실험/2. 필라멘트/실험데이터/22.10.19(수)/1019'

"pandas 객체 데이터 구성"
colName = ['Vb','I']
data = pd.read_csv(defaultPath+'/1019_shot5.dat', delimiter='\t', names=colName, header=None)
data = data.dropna().reset_index(drop=True)

"Ion linear fitting"
#parameter
Vb = -57.4
# x val
dataV = data['Vb'].to_numpy()
# y val
dataI_original = data['I'].to_numpy()

# linear fitting result
ionFittngResult = linearFitting(dataV,dataI_original,xTo=Vb)
print('ion fit result\n',ionFittngResult)
IonCurve = np.poly1d(ionFittngResult['polynomial'])
dataI_delIon = dataI_original - IonCurve(dataV)

"Electron linear fitting"
#parameter
VFrom = -57.4
VTo = -40
primaryElectronFittingResult = linearFitting(dataV,dataI_delIon,xFrom=VFrom,xTo=VTo)
print('primary electron fit result\n',primaryElectronFittingResult)
PrimaryElectronCurve = np.poly1d(primaryElectronFittingResult['polynomial'])
dataI_delIonElec = dataI_delIon - PrimaryElectronCurve(dataV)


"BE Electron fitting"
# get Vp(plasma voltage), kTe(electron energy)
dataI_delIonElec_ln = np.log(np.abs(dataI_delIonElec))

# Vp
BEResult = linearFitting(dataV,dataI_delIonElec_ln,xFrom=0,xTo=2)
print(BEResult)
otherResult = linearFitting(dataV,dataI_delIonElec_ln,xFrom=6,xTo=8)
print(otherResult)
Vp = - (BEResult['polynomial'][1] - otherResult['polynomial'][1])/(BEResult['polynomial'][0] - otherResult['polynomial'][0])
print(Vp)

# Te
# 역수
