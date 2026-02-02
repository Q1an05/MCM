#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Unified Visualization Configuration
统一的学术风格可视化配置 - 渐变莫兰迪色系

Author: MCM Team
Date: 2026-02-02
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns
import numpy as np

# =============================================================================
# 学术风格配色方案 (Academic Color Palette)
# 提高饱和度以增强辨识度，符合科研标准
# =============================================================================

# 主色板 - 8色学术配色 (适用于分类数据)
MORANDI_COLORS = [
    '#8B7355',  # 深咖啡棕 - Deep Coffee
    '#5B9BD5',  # 学术蓝 - Academic Blue
    '#ED7D31',  # 橙红 - Coral Orange
    '#70AD47',  # 绿色 - Forest Green
    '#A5A5A5',  # 中性灰 - Neutral Gray
    '#4472C4',  # 宝石蓝 - Sapphire Blue
    '#C55A11',  # 深橙 - Deep Orange
    '#548235',  # 深绿 - Deep Green
]

# 强调色 (用于突出重要数据)
MORANDI_ACCENT = [
    '#C00000',  # 深红 - Deep Red
    '#0070C0',  # 钴蓝 - Cobalt Blue
    '#00B050',  # 鲜绿 - Bright Green
    '#7030A0',  # 紫色 - Purple
]

# 连续渐变色板 (用于热图、梯度等)
MORANDI_GRADIENT_COOL = ['#F2F2F2', '#DEEBF7', '#9ECAE1', '#4292C6', '#08519C']  # 冷色调
MORANDI_GRADIENT_WARM = ['#FFF5EB', '#FDD0A2', '#FD8D3C', '#E6550D', '#A63603']  # 暖色调
MORANDI_GRADIENT_NEUTRAL = ['#F7F7F7', '#CCCCCC', '#969696', '#636363', '#252525']  # 中性色调

# 分歧色板 (用于对比数据，如正负值)
MORANDI_DIVERGING = ['#0070C0', '#9ECAE1', '#F7F7F7', '#FD8D3C', '#C00000']

# 序列色板 (单色渐变)
MORANDI_SEQ_BLUE = ['#F7FBFF', '#DEEBF7', '#9ECAE1', '#4292C6', '#08519C']
MORANDI_SEQ_GREEN = ['#F7FCF5', '#C7E9C0', '#74C476', '#31A354', '#006D2C']
MORANDI_SEQ_PINK = ['#FFF5F0', '#FCBBA1', '#FC9272', '#EF3B2C', '#A50F15']

# =============================================================================
# Matplotlib配置
# =============================================================================

def setup_academic_style():
    """
    设置学术风格的全局matplotlib配置
    """
    # 字体配置 - 使用DejaVu Sans以支持中英文
    plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 13
    
    # 线条与边框
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['grid.linewidth'] = 0.6
    plt.rcParams['lines.linewidth'] = 1.8
    plt.rcParams['patch.linewidth'] = 0.8
    
    # 网格与背景
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.25
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['axes.facecolor'] = '#FAFAFA'
    plt.rcParams['figure.facecolor'] = 'white'
    
    # 边距与布局
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['axes.edgecolor'] = '#666666'
    plt.rcParams['axes.labelcolor'] = '#333333'
    
    # 刻度
    plt.rcParams['xtick.color'] = '#666666'
    plt.rcParams['ytick.color'] = '#666666'
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    
    # 图例
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['legend.edgecolor'] = '#CCCCCC'
    plt.rcParams['legend.fancybox'] = True
    
    # DPI设置
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    
    # 设置默认配色为莫兰迪色系
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=MORANDI_COLORS)


# =============================================================================
# 自定义ColorMaps
# =============================================================================

def create_morandi_cmaps():
    """
    创建莫兰迪风格的colormap
    """
    cmaps = {}
    
    # 冷色调渐变
    cmaps['morandi_cool'] = LinearSegmentedColormap.from_list(
        'morandi_cool', MORANDI_GRADIENT_COOL
    )
    
    # 暖色调渐变
    cmaps['morandi_warm'] = LinearSegmentedColormap.from_list(
        'morandi_warm', MORANDI_GRADIENT_WARM
    )
    
    # 中性色调渐变
    cmaps['morandi_neutral'] = LinearSegmentedColormap.from_list(
        'morandi_neutral', MORANDI_GRADIENT_NEUTRAL
    )
    
    # 分歧色板
    cmaps['morandi_diverging'] = LinearSegmentedColormap.from_list(
        'morandi_diverging', MORANDI_DIVERGING
    )
    
    # 序列色板
    cmaps['morandi_seq_blue'] = LinearSegmentedColormap.from_list(
        'morandi_seq_blue', MORANDI_SEQ_BLUE
    )
    cmaps['morandi_seq_green'] = LinearSegmentedColormap.from_list(
        'morandi_seq_green', MORANDI_SEQ_GREEN
    )
    cmaps['morandi_seq_pink'] = LinearSegmentedColormap.from_list(
        'morandi_seq_pink', MORANDI_SEQ_PINK
    )
    
    # 注册到matplotlib
    for name, cmap in cmaps.items():
        try:
            mpl.colormaps.register(cmap, name=name, force=True)
        except:
            plt.register_cmap(name=name, cmap=cmap)
    
    return cmaps


# =============================================================================
# 便捷函数
# =============================================================================

def get_color(index: int) -> str:
    """
    获取指定索引的莫兰迪颜色
    
    Args:
        index: 颜色索引
        
    Returns:
        颜色hex代码
    """
    return MORANDI_COLORS[index % len(MORANDI_COLORS)]


def get_palette(n_colors: int, palette_type: str = 'main') -> list:
    """
    获取指定数量的颜色列表
    
    Args:
        n_colors: 需要的颜色数量
        palette_type: 色板类型 ('main', 'accent', 'cool', 'warm')
        
    Returns:
        颜色列表
    """
    if palette_type == 'main':
        base = MORANDI_COLORS
    elif palette_type == 'accent':
        base = MORANDI_ACCENT
    elif palette_type == 'cool':
        base = MORANDI_GRADIENT_COOL
    elif palette_type == 'warm':
        base = MORANDI_GRADIENT_WARM
    else:
        base = MORANDI_COLORS
    
    # 如果需要更多颜色，循环使用
    if n_colors <= len(base):
        return base[:n_colors]
    else:
        return (base * ((n_colors // len(base)) + 1))[:n_colors]


def style_axes(ax, title: str = None, xlabel: str = None, ylabel: str = None,
               grid: bool = True, spine_visible: dict = None):
    """
    统一设置坐标轴样式
    
    Args:
        ax: matplotlib axes对象
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        grid: 是否显示网格
        spine_visible: 控制边框显示的字典，如 {'top': False, 'right': False}
    """
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10, color='#333333')
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color='#333333')
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color='#333333')
    
    if grid:
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6, color='#888888')
        ax.set_axisbelow(True)
    
    # 设置边框可见性
    if spine_visible is None:
        spine_visible = {'top': False, 'right': False}
    
    for spine, visible in spine_visible.items():
        ax.spines[spine].set_visible(visible)
    
    # 设置刻度颜色
    ax.tick_params(colors='#666666', which='both')


def save_figure(fig, filepath: str, dpi: int = 300, tight: bool = True):
    """
    统一保存图表
    
    Args:
        fig: matplotlib figure对象
        filepath: 保存路径
        dpi: 分辨率
        tight: 是否使用tight_layout
    """
    if tight:
        fig.tight_layout()
    
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[INFO] Figure saved: {filepath}")


# =============================================================================
# Era颜色映射 (兼容性)
# =============================================================================

# 为不同时代定义专用颜色 (莫兰迪版本)
ERA_COLORS = {
    'Rank': MORANDI_COLORS[0],          # 灰咖啡
    'Percent': MORANDI_COLORS[1],       # 雾霾蓝
    'Rank_With_Save': MORANDI_COLORS[3] # 橄榄绿
}

# 对比色映射 (用于对立关系)
CONTRAST_COLORS = {
    'positive': MORANDI_ACCENT[2],   # 薄荷绿
    'negative': MORANDI_ACCENT[0],   # 玫瑰棕
    'neutral': MORANDI_COLORS[6]     # 暖沙色
}

# =============================================================================
# 初始化
# =============================================================================

# 自动设置学术风格
setup_academic_style()

# 创建并注册自定义colormaps
MORANDI_CMAPS = create_morandi_cmaps()

# Seaborn集成
sns.set_palette(MORANDI_COLORS)
sns.set_style("whitegrid", {
    'axes.facecolor': '#FAFAFA',
    'grid.color': '#DDDDDD',
    'grid.linestyle': '--',
    'axes.edgecolor': '#666666'
})

print("[INFO] Morandi academic visualization style loaded successfully!")
print(f"[INFO] Available colormaps: {list(MORANDI_CMAPS.keys())}")
