import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from typing import Dict, Tuple, Optional, List, Callable, Any

class FunctionFit:
    """
    曲线拟合器类
    提供多种函数拟合功能
    """
    
    def __init__(self):
        """初始化拟合器"""
        self.x_data = None
        self.y_data = None
        self.fit_results = {}
        self.best_fit = None
        
        # 定义所有可用的拟合函数
        self.functions = {
            'linear': {
                'func': self.linear_func,
                'name': '线性函数',
                'params': ['a', 'b'],
                'formula': 'y = a*x + b'
            },
            'quadratic': {
                'func': self.quadratic_func,
                'name': '二次函数',
                'params': ['a', 'b', 'c'],
                'formula': 'y = a*x² + b*x + c'
            },
            'exponential': {
                'func': self.exponential_func,
                'name': '指数函数',
                'params': ['a', 'b', 'c'],
                'formula': 'y = a * exp(b*x) + c'
            },
            'sine': {
                'func': self.sine_func,
                'name': '正弦函数',
                'params': ['振幅', '频率', '相位', '偏移'],
                'formula': 'y = a * sin(b*(x + c)) + d'
            },
            'cosine': {
                'func': self.cosine_func,
                'name': '余弦函数',
                'params': ['振幅', '频率', '相位', '偏移'],
                'formula': 'y = a * cos(b*(x + c)) + d'
            },
            'damped_sine': {
                'func': self.damped_sine_func,
                'name': '阻尼正弦函数',
                'params': ['初始振幅', '衰减系数', '频率', '相位', '偏移'],
                'formula': 'y = a * exp(-b*x) * sin(c*(x + d)) + e'
            },
            'power': {
                'func': self.power_func,
                'name': '幂函数',
                'params': ['a', 'b'],
                'formula': 'y = a * x^b'
            },
            'gaussian': {
                'func': self.gaussian_func,
                'name': '高斯函数',
                'params': ['振幅', '均值', '标准差'],
                'formula': 'y = a * exp(-(x-b)²/(2*c²))'
            }
        }
    
    # 拟合函数定义
    @staticmethod
    def linear_func(x, a, b):
        """线性函数: y = a*x + b"""
        return a * x + b
    
    @staticmethod
    def quadratic_func(x, a, b, c):
        """二次函数: y = a*x² + b*x + c"""
        return a * x**2 + b * x + c
    
    @staticmethod
    def exponential_func(x, a, b, c):
        """指数函数: y = a * exp(b*x) + c"""
        return a * np.exp(b * x) + c
    
    @staticmethod
    def sine_func(x, a, b, c, d):
        """正弦函数: y = a * sin(b*(x + c)) + d"""
        return a * np.sin(b * (x + c)) + d
    
    @staticmethod
    def cosine_func(x, a, b, c, d):
        """余弦函数: y = a * cos(b*(x + c)) + d"""
        return a * np.cos((b * x + c)) + d
    
    @staticmethod
    def damped_sine_func(x, a, b, c, d, e):
        """阻尼正弦函数: y = a * exp(-b*x) * sin(c*(x + d)) + e"""
        return a * np.exp(-b * x) * np.sin(c * (x + d)) + e
    
    @staticmethod
    def power_func(x, a, b):
        """幂函数: y = a * x^b"""
        return a * x**b
    
    @staticmethod
    def gaussian_func(x, a, b, c):
        """高斯函数: y = a * exp(-(x-b)²/(2*c²))"""
        return a * np.exp(-(x - b)**2 / (2 * c**2))
    
    def set_data(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """
        设置拟合数据
        
        参数:
        x_data: x数据数组
        y_data: y数据数组
        """
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.fit_results = {}
        self.best_fit = None
    
    def _get_initial_guesses(self, func_name: str) -> Tuple[List, Tuple, Tuple]:
        """
        获取初始参数猜测和边界
        
        参数:
        func_name: 函数名称
        
        返回:
        p0: 初始参数猜测
        bounds_lower: 参数下界
        bounds_upper: 参数上界
        """
        x = self.x_data
        y = self.y_data
        
        amplitude_guess = (np.max(y) - np.min(y)) / 2
        frequency_guess = 2 * np.pi / (x[-1] - x[0]) if len(x) > 1 else 1
        
        if func_name == 'linear':
            p0 = [1, 0]
            bounds = (-np.inf, np.inf)
        elif func_name == 'quadratic':
            p0 = [1, 1, 0]
            bounds = (-np.inf, np.inf)
        elif func_name == 'exponential':
            p0 = [np.max(y), 0.1, np.min(y)]
            bounds = ([0, -10, -np.inf], [np.inf, 10, np.inf])
        elif func_name == 'sine':
            p0 = [amplitude_guess, frequency_guess, 0, np.mean(y)]
            bounds = ([0, 0, -np.pi, -np.inf], [np.inf, np.inf, np.pi, np.inf])
        elif func_name == 'cosine':
            p0 = [amplitude_guess, frequency_guess, 0, np.mean(y)]
            bounds = ([0, 0, -np.pi, -np.inf], [np.inf, np.inf, np.pi, np.inf])
        elif func_name == 'damped_sine':
            p0 = [amplitude_guess, 0.1, frequency_guess, 0, np.mean(y)]
            bounds = ([0, 0, 0, -np.pi, -np.inf], [np.inf, np.inf, np.inf, np.pi, np.inf])
        elif func_name == 'power':
            p0 = [1, 1]
            bounds = ([0, -10], [np.inf, 10])
        elif func_name == 'gaussian':
            p0 = [amplitude_guess, np.mean(x), np.std(x)]
            bounds = ([0, -np.inf, 0], [np.inf, np.inf, np.inf])
        else:
            p0 = None
            bounds = (-np.inf, np.inf)
        
        return p0, bounds
    
    def fit_all(self) -> Dict[str, Dict]:
        """
        进行所有可用函数的拟合
        
        返回:
        Dict: 包含所有拟合结果的字典
        """
        if self.x_data is None or self.y_data is None:
            raise ValueError("请先使用set_data()方法设置数据")
        
        self.fit_results = {}
        best_r2 = -np.inf
        
        for func_name, func_info in self.functions.items():
            try:
                func = func_info['func']
                p0, bounds = self._get_initial_guesses(func_name)
                
                # 进行拟合
                popt, pcov, r2 = self._fit_curve(func, p0, bounds)
                
                if popt is not None:
                    self.fit_results[func_name] = {
                        'parameters': popt,
                        'covariance': pcov,
                        'r_squared': r2,
                        'function_info': func_info
                    }
                    
                    # 更新最佳拟合
                    if r2 > best_r2:
                        best_r2 = r2
                        self.best_fit = func_name
                        
            except Exception as e:
                print(f"拟合 {func_info['name']} 时出错: {e}")
                continue
        
        return self.fit_results
    
    def fit_specific(self, func_name: str) -> Optional[Dict]:
        """
        拟合特定函数
        
        参数:
        func_name: 函数名称
        
        返回:
        Dict: 拟合结果，失败返回None
        """
        if func_name not in self.functions:
            raise ValueError(f"未知的函数: {func_name}")
        
        func_info = self.functions[func_name]
        func = func_info['func']
        p0, bounds = self._get_initial_guesses(func_name)
        
        try:
            popt, pcov, r2 = self._fit_curve(func, p0, bounds)
            
            if popt is not None:
                result = {
                    'parameters': popt,
                    'covariance': pcov,
                    'r_squared': r2,
                    'function_info': func_info
                }
                
                # 更新总体结果
                self.fit_results[func_name] = result
                
                # 更新最佳拟合
                if self.best_fit is None or r2 > self.fit_results.get(self.best_fit, {}).get('r_squared', -np.inf):
                    self.best_fit = func_name
                
                return result
                
        except Exception as e:
            print(f"拟合 {func_info['name']} 时出错: {e}")
        
        return None
    
    def _fit_curve(self, func: Callable, p0: List, bounds: Tuple) -> Tuple:
        """
        执行曲线拟合
        
        参数:
        func: 拟合函数
        p0: 初始参数猜测
        bounds: 参数边界
        
        返回:
        popt: 最优参数
        pcov: 协方差矩阵
        r_squared: R²值
        """
        try:
            popt, pcov = curve_fit(func, self.x_data, self.y_data, p0=p0, bounds=bounds)
            
            # 计算R²
            y_pred = func(self.x_data, *popt)
            ss_res = np.sum((self.y_data - y_pred) ** 2)
            ss_tot = np.sum((self.y_data - np.mean(self.y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return popt, pcov, r_squared
            
        except Exception as e:
            return None, None, 0
    
    def get_best_fit(self) -> Optional[Dict]:
        """
        获取最佳拟合结果
        
        返回:
        Dict: 最佳拟合结果，无结果返回None
        """
        if self.best_fit and self.best_fit in self.fit_results:
            return self.fit_results[self.best_fit]
        return None
    
    def predict(self, x_values: np.ndarray, func_name: str = None) -> np.ndarray:
        """
        使用拟合函数进行预测
        
        参数:
        x_values: 要预测的x值
        func_name: 函数名称，None表示使用最佳拟合
        
        返回:
        np.ndarray: 预测的y值
        """
        if func_name is None:
            func_name = self.best_fit
        
        if func_name not in self.fit_results:
            raise ValueError(f"没有找到 {func_name} 的拟合结果")
        
        result = self.fit_results[func_name]
        func = result['function_info']['func']
        params = result['parameters']
        
        return func(x_values, *params)
    
    def print_report(self) -> None:
        """打印拟合报告"""
        if not self.fit_results:
            print("没有可用的拟合结果")
            return
        
        print("=" * 70)
        print("曲线拟合分析报告")
        print("=" * 70)
        
        for func_name, result in self.fit_results.items():
            func_info = result['function_info']
            params = result['parameters']
            r2 = result['r_squared']
            
            print(f"\n{func_info['name']} ({func_info['formula']}):")
            print(f"  R² = {r2:.6f}")
            print(f"  参数:")
            for i, (param_name, param_value) in enumerate(zip(func_info['params'], params)):
                print(f"    {param_name} = {param_value:.6f}")
        
        if self.best_fit:
            best_result = self.fit_results[self.best_fit]
            best_info = best_result['function_info']
            print(f"\n最佳拟合: {best_info['name']} (R² = {best_result['r_squared']:.6f})")
        
        print("=" * 70)
    
    def plot_fit(self, func_name: str = None, title: str = "函数拟合结果") -> None:
        """
        绘制拟合结果
        
        参数:
        func_name: 函数名称，None表示使用最佳拟合
        title: 图表标题
        """
        if func_name is None:
            func_name = self.best_fit
        
        if func_name not in self.fit_results:
            raise ValueError(f"没有找到 {func_name} 的拟合结果")
        
        result = self.fit_results[func_name]
        func_info = result['function_info']
        params = result['parameters']
        
        plt.figure(figsize=(10, 6))
        
        # 绘制原始数据
        plt.scatter(self.x_data, self.y_data, alpha=0.7, label='原始数据', color='blue', s=30)
        
        # 绘制拟合曲线
        x_fit = np.linspace(self.x_data.min(), self.x_data.max(), 1000)
        y_fit = func_info['func'](x_fit, *params)
        plt.plot(x_fit, y_fit, 'r-', label='拟合曲线', linewidth=2)
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def fit_custom_function(self, custom_func: Callable, param_names: List[str] = None, 
                           p0: List = None, bounds: Tuple = (-np.inf, np.inf)) -> Optional[Dict]:
        """
        拟合自定义函数

        参数:
        custom_func: 自定义函数，格式为 func(x, *params)
        param_names: 参数名称列表，可选
        p0: 初始参数猜测，可选
        bounds: 参数边界，可选

        返回:
        Dict: 拟合结果，失败返回None
        """
        if self.x_data is None or self.y_data is None:
            raise ValueError("请先使用set_data()方法设置数据")

        if param_names is None:
            # 如果没有提供参数名，自动生成
            import inspect
            sig = inspect.signature(custom_func)
            num_params = len(sig.parameters) - 1  # 减去x参数
            param_names = [f'param_{i}' for i in range(num_params)]

        if p0 is None:
            # 如果没有提供初始猜测，使用默认值
            p0 = [1.0] * len(param_names)

        try:
            # 进行拟合
            popt, pcov, r2 = self._fit_curve(custom_func, p0, bounds)

            if popt is not None:
                # 创建函数信息
                func_info = {
                    'func': custom_func,
                    'name': '自定义函数',
                    'params': param_names,
                    'formula': '用户自定义函数'
                }

                result = {
                    'parameters': popt,
                    'covariance': pcov,
                    'r_squared': r2,
                    'function_info': func_info
                }

                # 添加到结果中
                func_key = f'custom_{len([k for k in self.fit_results.keys() if k.startswith("custom_")])}'
                self.fit_results[func_key] = result

                # 更新最佳拟合
                if self.best_fit is None or r2 > self.fit_results.get(self.best_fit, {}).get('r_squared', -np.inf):
                    self.best_fit = func_key

                return result

        except Exception as e:
            print(f"拟合自定义函数时出错: {e}")

        return None
