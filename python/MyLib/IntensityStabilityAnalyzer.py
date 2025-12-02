"""
光强稳定性分析器类
用于分析光功率数据的稳定性指标
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List, Optional
import os


class IntensityStabilityAnalyzer:
    """光强稳定性分析器类"""
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        初始化分析器
        
        参数:
        data: 包含光强数据的DataFrame，可选
        """
        self.data = data
        self.time_data = None
        self.intensity_data = None
        self.results = {}
    
    def load_excel(self, file_path: str, skiprows: int = 4, usecols: str = "A:B") -> None:
        """
        从Excel文件加载数据
        
        参数:
        file_path: Excel文件路径
        skiprows: 跳过的行数
        usecols: 使用的列范围
        """
        try:
            self.data = pd.read_excel(file_path, skiprows=skiprows, usecols=usecols)
            self._process_data()
            print(f"成功加载文件: {os.path.basename(file_path)}")
        except Exception as e:
            raise Exception(f"加载Excel文件失败: {e}")
    
    def load_csv(self, file_path: str, **kwargs) -> None:
        """
        从CSV文件加载数据
        
        参数:
        file_path: CSV文件路径
        **kwargs: 传递给pd.read_csv的其他参数
        """
        try:
            self.data = pd.read_csv(file_path, **kwargs)
            self._process_data()
            print(f"成功加载文件: {os.path.basename(file_path)}")
        except Exception as e:
            raise Exception(f"加载CSV文件失败: {e}")
    
    def _process_data(self) -> None:
        """处理加载的数据"""
        if self.data is None or len(self.data.columns) < 2:
            raise ValueError("数据格式不正确，需要至少2列数据")
        
        # 假设第一列是光强数据，第二列是时间数据
        self.intensity_data = 1e6 * self.data[self.data.columns[0]]  # 转换为μW
        self.time_data = 0.001 * (self.data[self.data.columns[1]] - self.data[self.data.columns[1]].iloc[0])
    
    def set_data(self, time_data: np.ndarray, intensity_data: np.ndarray) -> None:
        """
        直接设置时间数据和光强数据
        
        参数:
        time_data: 时间数据数组
        intensity_data: 光强数据数组
        """
        self.time_data = time_data
        self.intensity_data = intensity_data
    
    def analyze_stability(self) -> dict:
        """
        分析光强稳定性
        
        返回:
        dict: 包含所有稳定性指标的结果字典
        """
        if self.intensity_data is None:
            raise ValueError("请先加载数据或设置数据")
        
        # 计算各项指标
        self.results = {
            'mean_intensity': np.mean(self.intensity_data),
            'peak_to_peak': self._peak_to_peak_fluctuation(),
            'rms_fluctuation': self._rms_fluctuation(),
            'power_stability': self._power_stability(),
            'signal_to_noise': self._signal_to_noise_ratio(),
            'total_time': self._get_total_time(),
            'data_points': len(self.intensity_data)
        }
        
        return self.results
    
    def _peak_to_peak_fluctuation(self) -> Tuple[float, float]:
        """计算峰峰值波动"""
        I_max = np.max(self.intensity_data)
        I_min = np.min(self.intensity_data)
        I_avg = np.mean(self.intensity_data)
        
        absolute_pp = I_max - I_min
        relative_pp = (absolute_pp / I_avg) * 100
        
        return absolute_pp, relative_pp
    
    def _rms_fluctuation(self) -> Tuple[float, float]:
        """计算RMS起伏"""
        I_avg = np.mean(self.intensity_data)
        I_std = np.std(self.intensity_data)
        
        absolute_rms = I_std
        relative_rms = (I_std / I_avg) * 100
        
        return absolute_rms, relative_rms
    
    def _power_stability(self) -> float:
        """计算功率稳定度"""
        I_avg = np.mean(self.intensity_data)
        I_std = np.std(self.intensity_data)
        
        stability = (1 - I_std / I_avg) * 100
        return stability
    
    def _signal_to_noise_ratio(self) -> float:
        """计算信噪比"""
        I_avg = np.mean(self.intensity_data)
        I_std = np.std(self.intensity_data)
        
        snr_db = 20 * np.log10(I_avg / I_std)
        return snr_db
    
    def _get_total_time(self) -> Tuple[int, float]:
        """获取总时间（分:秒格式）"""
        if self.time_data is None:
            return 0, 0.0
        
        total_seconds = self.time_data.iloc[-1] if hasattr(self.time_data, 'iloc') else self.time_data[-1]
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        
        return minutes, seconds
    
    def plot_intensity_vs_time(self, title: str = "光功率随时间变化曲线", 
                              save_path: Optional[str] = None) -> None:
        """
        绘制光功率随时间变化曲线
        
        参数:
        title: 图表标题
        save_path: 保存路径，如果提供则保存图表
        """
        if self.time_data is None or self.intensity_data is None:
            raise ValueError("没有可用的数据用于绘图")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_data / 60, self.intensity_data, linewidth=1)
        plt.xlabel("时间 (分钟)")
        plt.ylabel("光功率 (μW)")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def print_report(self, title: str = "光强稳定性分析报告") -> None:
        """打印分析报告"""
        if not self.results:
            print("请先运行analyze_stability()方法")
            return
        
        print("=" * 60)
        print(title)
        print("=" * 60)
        print(f"数据点数: {self.results['data_points']}")
        minutes, seconds = self.results['total_time']
        print(f"总时间: {minutes}分{seconds:.1f}秒")
        print(f"光功率均值: {self.results['mean_intensity']:.4f} μW")
        
        pp_abs, pp_rel = self.results['peak_to_peak']
        print(f"峰峰值波动: {pp_abs:.4f} μW, 相对峰峰值波动: {pp_rel:.4f} %")
        
        rms_abs, rms_rel = self.results['rms_fluctuation']
        print(f"RMS起伏: {rms_abs:.4f} μW, 相对RMS起伏: {rms_rel:.4f} %")
        
        print(f"功率稳定度: {self.results['power_stability']:.4f} %")
        print(f"信噪比: {self.results['signal_to_noise']:.4f} dB")
        print("=" * 60)

