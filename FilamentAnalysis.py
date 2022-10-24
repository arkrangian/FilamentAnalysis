import pandas as pd
import numpy as np
from scipy import stats
from linearFitting import linearFitting

class FilamentExp:
    def __init__(self, datapath, memo, filamentV, ionCut, peCutL, peCutR, cutBK_L = 0, cutBK_R = 2, cutRM_L = 6, cutRM_R = 8) -> None:
        self.memo = memo
        
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
        self.dataV = self.data['Vb'].to_numpy()
        self.dataI_original = self.data['I'].to_numpy()
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
        """ 플라즈마 전위 """
        coeffBE = self.resultBE['polynomial']
        coeffRM = self.resultRM['polynomial'] 
        self.plasmaV = - (coeffBE[1] - coeffRM[1])/(coeffBE[0] - coeffRM[0])

        """ 전자 온도 """
        # Bulk Electron ln 상에서 linearfit 한거 기울기 역수가 온도(eV)
        self.tempBE = 1/coeffBE[0]

        """ 상수 """
        e = 1.6 * 10**(-19)
        k = 8.617 * 10**(-5)
        me = 9.109 * 10**(-31)
        # 타겟기체 질량, 여기선 Ar
        M_ion = 40 * 1.66 * 10**(-27)
        Cs = np.sqrt(self.tempBE/M_ion)
        # 탐침 면적
        Area = np.pi * (0.005) ** 2

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

    def breif(self):
        print(self.memo)
        # Ion curve
        print("이온 커브")
        print(self.resultIon)
        # Primary Electron curve
        print("PE 커브")
        print(self.resultPE)
        # Bulk Electron curve
        print("bulk전자 커브(ln 씌운거)")
        print(self.resultBE)
        # rightmost curve (플라즈마 전위구할때 씀)
        print("Rm 커브(ln 씌운거)")
        print(self.resultRM)

        """ characteristic values """
        print("플라즈마 전위")
        print(self.plasmaV)
        print("벌크전자 온도")
        print(self.tempBE)
        print("이온 밀도")
        print(self.nIon) 
        print("주전자 밀도")
        print(self.nPE)
        print("벌크전자 밀도")
        print(self.nBE)
        print("")
