# --------------- HFGR functions  -------------------
#    Calculating T_rot from NH3 (1,1) and (2,2) lines
#    Created on Fri Jan 5 14:02:20 2016
#    Latest editing: June 10, 2020
#    @author: Zhiyuan Ren, Shen Wang


import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
from scipy import stats
from astropy.modeling import custom_model
from astropy.modeling.fitting import LevMarLSQFitter
# from matplotlib.pyplot.figure as figure

def ap1_fit(Trot=30, dv=0.5):
    # ------------fitting functions-------------
    h1 = np.array([0.71485162, -1.16045709, 0.26500927, 0.22447124, -0.07664547])
    h2 = np.array([1.00365352, -2.89365732, -0.62763706, 2.18515929, 0.21931311])
    h1s = np.array([1.18921497, -4.42581709, 0.00975187, 9.47213247, 0.01528113])
    h2s = np.array([-3.79926585, 35.70032889, 1.06861243, -94.07470039, -0.54213132])
    # Rsm0=0.555511; cf0=1.10205; # (Rsm/R12 includes mg+isg)
    # Rsm0 = 1.00003; cf0 = 0.952406;  # (Rsm/R12 includes mg+isg+osg)
    if Trot > 14:
        return h1[0] + h1[1] * (Trot / 60.0) + h1[2] * (dv / 1.0) + h1[3] * (Trot / 60.0) ** 2 + h1[4] * (dv / 1.0) ** 2
    else:
        return h1s[0] + h1s[1] * (Trot / 60.0) + h1s[2] * (dv / 1.0) + h1s[3] * (Trot / 60.0) ** 2 + h1s[4] * (
                    dv / 1.0) ** 2


def ap2_fit(Trot=30, dv=0.5):
    # ------------fitting functions-------------
    h1 = np.array([0.71485162, -1.16045709, 0.26500927, 0.22447124, -0.07664547])
    h2 = np.array([1.00365352, -2.89365732, -0.62763706, 2.18515929, 0.21931311])
    h1s = np.array([1.18921497, -4.42581709, 0.00975187, 9.47213247, 0.01528113])
    h2s = np.array([-3.79926585, 35.70032889, 1.06861243, -94.07470039, -0.54213132])
    # Rsm0=0.555511; cf0=1.10205; # (Rsm/R12 includes mg+isg)
    Rsm0 = 1.00003; cf0 = 0.952406;  # (Rsm/R12 includes mg+isg+osg)
    if Trot > 14:
        return h2[0] + h2[1] * (Trot / 60.0) + h2[2] * (dv / 1.0) + h2[3] * (Trot / 60.0) ** 2 + h2[4] * (dv / 1.0) ** 2
    else:
        return h2s[0] + h2s[1] * (Trot / 60.0) + h2s[2] * (dv / 1.0) + h2s[3] * (Trot / 60.0) ** 2 + h2s[4] * (
                    dv / 1.0) ** 2


def cf_fit(Rsm1, ap1=1.0, ap2=0.3):
    Rsm0 = 1.00003
    cf0 = 0.952406
    return ap1 * (Rsm1 - Rsm0) + ap2 * (Rsm1 - Rsm0) ** 2 + cf0


def Trot_fit(R12=1.5, cf=0.4):
    return 40.99 / np.log(cf * R12)

def trot_fun(v_ax11, Tb_11, v_ax22, Tb_22):
    wm = 4.0
    n_it = 3
    i0_11 = np.argmax(Tb_11)
    i0_22 = np.argmax(Tb_22)
    vsys_11 = v_ax11[i0_11]
    vsys_22 = v_ax11[i0_22]
    v_ax11 -= vsys_11
    v_ax22 -= vsys_22
    cw11 = np.abs(np.mean(v_ax11[1:]-v_ax11[0:-1]))
    cw22 = np.abs(np.mean(v_ax22[1:] - v_ax22[0:-1]))
    idx_mg1 = np.argwhere((v_ax11 < (0.00136 + wm)) & (v_ax11 > (0.00136 - wm)))
    idx_isg1a = np.argwhere((v_ax11 < (-7.4903 + wm)) & (v_ax11 > (-7.4903 - wm)))
    idx_isg1b = np.argwhere((v_ax11 < (7.49233 + wm)) & (v_ax11 > (7.49233 - wm)))
    idx_osg1a = np.argwhere((v_ax11 < (19.5022 + wm)) & (v_ax11 > (19.5022 - wm)))
    idx_osg1b = np.argwhere((v_ax11 < (-19.5048 + wm)) & (v_ax11 > (-19.5048 - wm)))
    idx_mg2 = np.argwhere((v_ax22 < (0.00068 + wm)) & (v_ax22 > (0.00068 - wm)))
    idx_isg2a = np.argwhere((v_ax22 < (-16.4 + wm)) & (v_ax22 > (-16.4 - wm)))
    idx_isg2b = np.argwhere((v_ax22 < (16.4 + wm)) & (v_ax22 > (16.4 - wm)))
    idx_osg2a = np.argwhere((v_ax22 < (-26.03 + wm)) & (v_ax22 > (-26.03 - wm)))
    idx_osg2b = np.argwhere((v_ax22 < (26.03 + wm)) & (v_ax22 > (26.3 - wm)))
    # roughly estimate Delta V
    dv_fw = np.sum(Tb_11[idx_mg1] > Tb_11[idx_mg1].max()*0.5) * cw11 - 0.026
    # correction for the channel-width contribution to Delta_V:
    dv_fw = np.sqrt(dv_fw**2 - cw11**2)
    # -------(1,1) intensities------
    S_mg1 = np.sum(Tb_11[idx_mg1]) * cw11
    S_isg1 = np.sum(Tb_11[idx_isg1a]) * cw11 + np.sum(Tb_11[idx_isg1b]) * cw11
    S_osg1 = np.sum(Tb_11[idx_osg1a]) * cw11 + np.sum(Tb_11[idx_osg1b]) * cw11
    # -------(2,2) intensities------
    S_mg2 = np.sum(Tb_22[idx_mg2]) * cw22
    S_isg2 = np.sum(Tb_22[idx_isg2a]) * cw22 + np.sum(Tb_22[idx_isg2b]) * cw22
    S_osg2 = np.sum(Tb_22[idx_osg2a]) * cw22 + np.sum(Tb_22[idx_osg2b]) * cw22
    # ------intensity ratios-----
    Rsm1 = (S_isg1 + S_osg1) / S_mg1
    R12 = (S_mg1 + S_isg1 + S_osg1) / (S_mg2 + S_isg2 + S_osg2)
    Trot0 = 40.99 / np.log(R12)
    for j in np.arange(n_it):
        ap1f = ap1_fit(Trot=Trot0, dv=dv_fw)
        ap2f = ap2_fit(Trot=Trot0, dv=dv_fw)
        cf1 = cf_fit(Rsm1=np.abs(Rsm1), ap1=ap1f, ap2=ap2f)
        Trot0 = Trot_fit(R12=np.abs(R12), cf=cf1)
    return Trot0


# ------------ intensity ratio ----------------
def tau_11m(s11_mg, s11_isg):
    tau_m = np.arange(0.001, 30, 0.02)
    ratio_ms = (1 - np.exp(-tau_m)) / (1 - np.exp(-tau_m/1.8))
    i_tau = np.argmin(np.abs(s11_mg/s11_isg) - ratio_ms)
    return tau_m[i_tau]

def Trot_intr(s11_mg, s11_isg, s22_mg):
    tau_11 = tau_11m(s11_mg, s11_isg)
    Tex_int0 = -41.5 / np.log(-0.42 / tau_11 * np.log(1 - np.abs(s22_mg/s11_mg) * (1 - np.exp(-tau_11))))
    return Tex_int0


def T11_theory(vs, dv, tau_0=0., v0=0., Tex=0., osg1=0., isg1=0., mg=0., isg2=0., osg2=0.):
    v_0, v_1 = -19.84514, -19.31960
    v_2, v_3, v_4 = -7.88632, -7.46920, -7.35005
    v_5, v_6, v_7, v_8, v_9, v_10, v_11, v_12 = -0.46227, -0.32312, -0.30864, -0.18950, 0.07399, 0.13304, 0.21316, 0.25219
    v_13, v_14, v_15 = 7.23455, 7.37370, 7.81539
    v_16, v_17 = 19.40943, 19.54859
    a0, a1 = 1.0 / 27, 2.0 / 27
    a2, a3, a4 = 5.0 / 108, 1.0 / 12, 1.0 / 108
    a5, a6, a7, a8, a9, a10, a11, a12 = 1.0 / 54, 1.0 / 108, 1.0 / 60, 3.0 / 20, 1.0 / 108, 7.0 / 30, 5.0 / 108, 1.0 / 60
    a13, a14, a15 = 5.0 / 108, 1.0 / 108, 1.0 / 12
    a16, a17 = 1.0 / 27, 2.0 / 27
    profile1 = a0 * np.exp(-4 * np.log(2) * ((vs - v0 - v_0) / dv) ** 2) + \
               a1 * np.exp(-4 * np.log(2) * ((vs - v0 - v_1) / dv) ** 2)  # osg_1
    profile2 = a2 * np.exp(-4 * np.log(2) * ((vs - v0 - v_2) / dv) ** 2) + \
               a3 * np.exp(-4 * np.log(2) * ((vs - v0 - v_3) / dv) ** 2) + \
               a4 * np.exp(-4 * np.log(2) * ((vs - v0 - v_4) / dv) ** 2)  # isg_1
    profile3 = a5 * np.exp(-4 * np.log(2) * ((vs - v0 - v_5) / dv) ** 2) + \
               a6 * np.exp(-4 * np.log(2) * ((vs - v0 - v_6) / dv) ** 2) + \
               a7 * np.exp(-4 * np.log(2) * ((vs - v0 - v_7) / dv) ** 2) + \
               a8 * np.exp(-4 * np.log(2) * ((vs - v0 - v_8) / dv) ** 2) + \
               a9 * np.exp(-4 * np.log(2) * ((vs - v0 - v_9) / dv) ** 2) + \
               a10 * np.exp(-4 * np.log(2) * ((vs - v0 - v_10) / dv) ** 2) + \
               a11 * np.exp(-4 * np.log(2) * ((vs - v0 - v_11) / dv) ** 2) + \
               a12 * np.exp(-4 * np.log(2) * ((vs - v0 - v_12) / dv) ** 2)  # mg
    profile4 = a13 * np.exp(-4 * np.log(2) * ((vs - v0 - v_13) / dv) ** 2) + \
               a14 * np.exp(-4 * np.log(2) * ((vs - v0 - v_14) / dv) ** 2) + \
               a15 * np.exp(-4 * np.log(2) * ((vs - v0 - v_15) / dv) ** 2)  # isg_2
    profile5 = a16 * np.exp(-4 * np.log(2) * ((vs - v0 - v_16) / dv) ** 2) + \
               a17 * np.exp(-4 * np.log(2) * ((vs - v0 - v_17) / dv) ** 2)  # osg_2
    # (a13+a14+a15+a2+a3+a4)/(a5+a6+a7+a8+a9+a10+a11+a12) ;   a13+a14+a15+a2+a3+a4

    tau_v = tau_0 * (osg1 * profile1 + isg1 * profile2 + mg * profile3 + isg2 * profile4 + osg2 * profile5)

    T11_theory = (1.137 / (-1 + np.exp(1.137 / Tex)) - 2.201) * (1 - np.exp(-tau_v))
    return T11_theory


def T22_theory(vs, dv, tau_0=0.0, v0=0., Tex=0., osg1=0., isg1=0., mg=0., isg2=0., osg2=0.):
    v_0, v_1, v_2 = -26.52625, -26.01112, -25.95045
    v_3, v_4, v_5 = -16.39171, -16.37929, -15.86417
    v_6, v_7, v_8, v_9, v_10, v_11, v_12, v_13, v_14, v_15, v_16, v_17 = -0.56250, -0.52841, -0.52374, -0.01328, -0.01328, 0.00390, \
                                                                         0.00390, 0.01332, 0.01332, 0.50183, 0.53134, 0.58908
    v_18, v_19, v_20 = 15.85468, 16.36980, 16.38222
    v_21, v_22, v_23 = 25.95045, 26.01112, 26.52625
    a0, a1, a2 = 1. / 300, 3. / 100, 1. / 60
    a3, a4, a5 = 4. / 135, 14. / 675, 1. / 675
    a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17 = 1. / 60, 1. / 108, 8. / 945, 7. / 54, 1. / 12, 8. / 35, 32. / 189, 1. / 12, 1. / 30, 1. / 108, \
                                                             8. / 945, 1. / 60
    a18, a19, a20 = 1. / 675, 14. / 675, 4. / 135
    a21, a22, a23 = 1. / 60, 3. / 100, 1. / 300
    profile1 = a0 * np.exp(-4 * np.log(2) * ((vs - v0 - v_0) / dv) ** 2) + \
               a1 * np.exp(-4 * np.log(2) * ((vs - v0 - v_1) / dv) ** 2) + \
               a2 * np.exp(-4 * np.log(2) * ((vs - v0 - v_2) / dv) ** 2)  # osg_1
    profile2 = a3 * np.exp(-4 * np.log(2) * ((vs - v0 - v_3) / dv) ** 2) + \
               a4 * np.exp(-4 * np.log(2) * ((vs - v0 - v_4) / dv) ** 2) + \
               a5 * np.exp(-4 * np.log(2) * ((vs - v0 - v_5) / dv) ** 2)  # isg_1
    profile3 = a6 * np.exp(-4 * np.log(2) * ((vs - v0 - v_6) / dv) ** 2) + \
               a7 * np.exp(-4 * np.log(2) * ((vs - v0 - v_7) / dv) ** 2) + \
               a8 * np.exp(-4 * np.log(2) * ((vs - v0 - v_8) / dv) ** 2) + \
               a9 * np.exp(-4 * np.log(2) * ((vs - v0 - v_9) / dv) ** 2) + \
               a10 * np.exp(-4 * np.log(2) * ((vs - v0 - v_10) / dv) ** 2) + \
               a11 * np.exp(-4 * np.log(2) * ((vs - v0 - v_11) / dv) ** 2) + \
               a12 * np.exp(-4 * np.log(2) * ((vs - v0 - v_12) / dv) ** 2) + \
               a13 * np.exp(-4 * np.log(2) * ((vs - v0 - v_13) / dv) ** 2) + \
               a14 * np.exp(-4 * np.log(2) * ((vs - v0 - v_14) / dv) ** 2) + \
               a15 * np.exp(-4 * np.log(2) * ((vs - v0 - v_15) / dv) ** 2) + \
               a16 * np.exp(-4 * np.log(2) * ((vs - v0 - v_16) / dv) ** 2) + \
               a17 * np.exp(-4 * np.log(2) * ((vs - v0 - v_17) / dv) ** 2)  # mg
    profile4 = a18 * np.exp(-4 * np.log(2) * ((vs - v0 - v_18) / dv) ** 2) + \
               a19 * np.exp(-4 * np.log(2) * ((vs - v0 - v_19) / dv) ** 2) + \
               a20 * np.exp(-4 * np.log(2) * ((vs - v0 - v_20) / dv) ** 2)  # isg_2
    profile5 = a21 * np.exp(-4 * np.log(2) * ((vs - v0 - v_21) / dv) ** 2) + \
               a22 * np.exp(-4 * np.log(2) * ((vs - v0 - v_22) / dv) ** 2) + \
               a23 * np.exp(-4 * np.log(2) * ((vs - v0 - v_23) / dv) ** 2)  # osg_2
    tau_v = tau_0 * (osg1 * profile1 + isg1 * profile2 + mg * profile3 + isg2 * profile4 + osg2 * profile5)
    T22_theory = (1.138 / (-1 + np.exp(1.138 / Tex)) - 2.2) * (1 - np.exp(-tau_v))
    return T22_theory
    # intrinsic Rsm=(a3+a4+a5+a18+a19+a20)/(a6+a7+a8+a9+a10+a11+a12+a13+a14+a15+a16+a17)


def funC11(dv, x, aa0, aa1, aa2, aa3, aa4):
    # y=aa0+aa1*x+aa2*x**2+aa3*(np.exp(aa4*x))
    aa0, aa1, aa2 = 0.53, -0.26, 0.63
    y = aa0 + aa1 * x + aa2 * x ** 2 + aa3 * (np.exp(aa4 * x))
    return y


def funC22(dv, x, aa5, aa6):
    y = aa5 * (x ** aa6)
    return y
