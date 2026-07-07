import numpy as np

class Ideal_JonesMatrix:
    """偏振光学琼斯矩阵工具类，所有方法返回 2×2 numpy 数组"""
    
    @staticmethod
    def polarizer(angle_deg):
        """理想线偏振片，透振轴与x轴夹角 angle_deg (度)"""
        theta = np.deg2rad(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c*c, c*s],
                         [s*c, s*s]])

    @staticmethod
    def wave_plate(retardance_deg, fast_axis_deg=0.0):
        """
        通用波片
        retardance_deg: 相位延迟量（度），例如 180 为半波片，90 为四分之一波片
        fast_axis_deg : 快轴与x轴夹角（度）
        """
        delta = np.deg2rad(retardance_deg) / 2.0
        theta = np.deg2rad(fast_axis_deg)
        return np.array([
            [np.cos(delta) + 1j*np.sin(delta)*np.cos(2*theta),
             1j*np.sin(delta)*np.sin(2*theta)],
            [1j*np.sin(delta)*np.sin(2*theta),
             np.cos(delta) - 1j*np.sin(delta)*np.cos(2*theta)]
        ])

    @staticmethod
    def hwp(fast_axis_deg=0.0):
        """半波片 (180° 延迟)"""
        return Ideal_JonesMatrix.wave_plate(180.0, fast_axis_deg)

    @staticmethod
    def qwp(fast_axis_deg=0.0):
        """四分之一波片 (90° 延迟)"""
        return Ideal_JonesMatrix.wave_plate(90.0, fast_axis_deg)

    @staticmethod
    def pbs(mode='transmit_P'):
        """
        偏振分束器
        mode='transmit_P' : 透射端口 (P光通过)
        mode='reflect_S'  : 反射端口 (S光通过)
        """
        if mode == 'transmit_P':
            return np.array([[1, 0], [0, 0]])
        elif mode == 'reflect_S':
            return np.array([[0, 0], [0, 1]])
        else:
            raise ValueError("mode must be 'transmit_P' or 'reflect_S'")

    @staticmethod
    def birefringent_sample(phase_e, phase_o, fast_axis_deg=0.0):
        """
        双折射样品，快慢轴分别引入相位延迟 phase_e, phase_o (弧度)
        fast_axis_deg : e光主轴与x轴夹角（度）
        """
        phi_e, phi_o = phase_e, phase_o
        theta = np.deg2rad(fast_axis_deg)
        # 主轴坐标系下的对角矩阵
        J_diag = np.array([[np.exp(1j*phi_e), 0],
                           [0, np.exp(1j*phi_o)]])
        # 旋转到实验室坐标系
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return R @ J_diag @ R.T

    @staticmethod
    def cascade(*matrices):
        """将多个琼斯矩阵按顺序级联（左乘），即 matrices[-1] @ ... @ matrices[0]"""
        if not matrices:
            raise ValueError("At least one matrix required")
        result = matrices[0]
        for M in matrices[1:]:
            result = M @ result
        return result