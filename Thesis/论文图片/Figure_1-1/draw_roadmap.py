"""
Figure 1-1 技术路线图 — Nature 风格
- Helvetica/Arial, 7pt 正文 / 8pt 标题
- 极细线条 (0.6pt), 无彩色填充
- 白底 + 浅灰线框 + 单色色带标题栏
- 紧凑、克制、一目了然
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial', 'Helvetica'],
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,       # Nature 要求
    'ps.fonttype': 42,
})

# ── 色彩体系：极低饱和度，仅章节色带用彩色 ───────────
CH3  = '#4878A8'   # 深蓝灰
CH4  = '#5A8F6E'   # 深绿灰
CH5  = '#7868A6'   # 深紫灰
CH6  = '#B85450'   # 深红灰
GOLD = '#C4960C'   # 结果高亮
BDR  = '#A0A0A0'   # 边框灰
TXT  = '#1A1A1A'   # 正文黑
TXT2 = '#555555'   # 次要文字
BG   = '#F5F5F5'   # 极浅灰填充

fig, ax = plt.subplots(figsize=(11, 15.5), dpi=300)
ax.set_xlim(0, 11)
ax.set_ylim(0, 20)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── 工具函数 ─────────────────────────────────────────
def band(y, color, label, w=9.6, fs=8.5):
    """章节色带标题栏 — 窄条"""
    rect = FancyBboxPatch((0.7, y - 0.22), w, 0.44,
                           boxstyle="round,pad=0.06", fc=color, ec='none',
                           lw=0, zorder=3, alpha=0.88)
    ax.add_patch(rect)
    ax.text(0.7 + w / 2, y, label, ha='center', va='center',
            fontsize=fs, fontweight='bold', color='white', zorder=4)

def rbox(x, y, w, h, text, fs=7, bold=False, fc='white', ec=BDR, lw=0.6):
    """圆角矩形"""
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle="round,pad=0.10", fc=fc, ec=ec,
                        lw=lw, zorder=3)
    ax.add_patch(b)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', color=TXT,
            zorder=4, linespacing=1.35)

def arr(x1, y1, x2, y2, c='#888888', lw=0.7):
    """细箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                               connectionstyle='arc3,rad=0'), zorder=2)

def tag(x, y, text, color):
    """章节标签"""
    ax.text(x, y, text, ha='center', va='center', fontsize=6.5,
            color='white', fontweight='bold', zorder=5,
            bbox=dict(boxstyle='round,pad=0.2', fc=color, ec='none', alpha=0.85))

# ── 标题 ─────────────────────────────────────────────
ax.text(5.5, 19.5, '基于多源光谱融合的水稻种质资源抗旱性快速评价方法',
        ha='center', va='center', fontsize=10.5, fontweight='bold', color=TXT)

# ═══════════════ 第3章 ═══════════════════════════════
Y = 18.6
tag(0.35, Y, '第3章', CH3)
band(Y, CH3, '实验设计与数据采集')

# 第一层
Y1 = 17.7
rbox(2.5, Y1, 3.0, 0.50, '13个籼稻品种\n干旱胁迫 + 复水处理', fc=BG)
rbox(5.5, Y1, 2.5, 0.50, '生理指标测定\n（5项破坏性指标）', fc=BG)
rbox(8.5, Y1, 2.5, 0.50, '多源光谱采集\n（3种模态同步）', fc=BG)
for xp in [2.5, 5.5, 8.5]:
    arr(5.5, Y - 0.26, xp, Y1 + 0.30)

# 第二层：三模态
Y2 = 16.6
rbox(2.5, Y2, 2.8, 0.70, '多光谱反射(Multi)\n11波段+8植被指数\n共19个特征', fs=6.5, fc=BG)
rbox(5.5, Y2, 2.8, 0.70, '稳态荧光(Static)\n4波段+6荧光比值\n共10个特征', fs=6.5, fc=BG)
rbox(8.5, Y2, 2.8, 0.70, 'OJIP荧光动力学\n8个JIP-test参数\n共8个特征', fs=6.5, fc=BG)
arr(8.5, Y1 - 0.30, 2.5, Y2 + 0.40)
arr(8.5, Y1 - 0.30, 5.5, Y2 + 0.40)
arr(8.5, Y1 - 0.30, 8.5, Y2 + 0.40)

# 汇聚
Y3 = 15.5
rbox(5.5, Y3, 5.0, 0.45, '多源融合特征体系（37个特征）', fs=8, bold=True, fc=BG, ec=CH3, lw=1.0)
for xp in [2.5, 5.5, 8.5]:
    arr(xp, Y2 - 0.40, 5.5, Y3 + 0.28)

# D_conv 分支
rbox(9.8, Y3, 2.0, 0.45, 'PCA-隶属函数法\n→ D_conv → 品种分类', fs=5.5, fc=BG)
arr(5.5, Y1 - 0.30, 9.8, Y3 + 0.28, c=CH3)

# ═══════════════ 第4章 ═══════════════════════════════
Y4 = 14.3
tag(0.35, Y4, '第4章', CH4)
band(Y4, CH4, '模型构建与消融分析')
arr(5.5, Y3 - 0.28, 5.5, Y4 + 0.26)

Y5 = 13.3
rbox(3.5, Y5, 4.0, 0.70, '六种模型系统比较\nPLSR | SVR | Ridge\nRF | CatBoost | TabPFN', fs=6.5, fc=BG)
rbox(7.8, Y5, 3.2, 0.70, '全组合消融实验\n单模态→双模态→三模态\n融合增益量化', fs=6.5, fc=BG)
arr(5.5, Y4 - 0.26, 3.5, Y5 + 0.40)
arr(5.5, Y4 - 0.26, 7.8, Y5 + 0.40)

# 最优结果
Y6 = 12.2
rbox(5.5, Y6, 6.0, 0.42, '最优配置：TabPFN + 三模态融合 (Spearman ρ = 1.000)',
     fs=7.5, bold=True, fc='#FFF8E7', ec=GOLD, lw=1.0)
arr(3.5, Y5 - 0.40, 5.5, Y6 + 0.26)
arr(7.8, Y5 - 0.40, 5.5, Y6 + 0.26)

# ═══════════════ 第5章 ═══════════════════════════════
Y7 = 11.0
tag(0.35, Y7, '第5章', CH5)
band(Y7, CH5, '可解释性与协同机制分析')
arr(5.5, Y6 - 0.26, 5.5, Y7 + 0.26)

Y8 = 9.9
rbox(2.5, Y8, 2.8, 0.70, 'CatBoost白盒代理\n保真度验证\n(ρ = 0.978)', fs=6.5, fc=BG)
rbox(5.5, Y8, 2.8, 0.70, 'TreeSHAP精确算法\n特征重要性排名\n模态贡献度分析', fs=6.5, fc=BG)
rbox(8.5, Y8, 2.8, 0.70, 'SHAP交互值分析\n跨模态协同效应\n生理学机制解读', fs=6.5, fc=BG)
for xp in [2.5, 5.5, 8.5]:
    arr(5.5, Y7 - 0.26, xp, Y8 + 0.40)

# 核心发现
Y9 = 8.8
rbox(5.5, Y9, 7.0, 0.42, '核心发现：BF(F440) × OJIP_Vi — "光合-代谢"协同防御机制',
     fs=7.5, bold=True, fc='#FFF8E7', ec=CH5, lw=1.0)
for xp in [2.5, 5.5, 8.5]:
    arr(xp, Y8 - 0.40, 5.5, Y9 + 0.26)

# ═══════════════ 第6章 ═══════════════════════════════
Y10 = 7.6
tag(0.35, Y10, '第6章', CH6)
band(Y10, CH6, '结论与应用展望')
arr(5.5, Y9 - 0.26, 5.5, Y10 + 0.26)

Y11 = 6.7
rbox(3.5, Y11, 3.8, 0.55, '13个品种抗旱性排名\n抗旱型(3) | 中间型(5) | 敏感型(5)', fs=6.5, fc=BG)
rbox(7.8, Y11, 3.2, 0.55, '育种应用与技术推广\n种质资源快速筛选方案', fs=6.5, fc=BG)
arr(5.5, Y10 - 0.26, 3.5, Y11 + 0.32)
arr(5.5, Y10 - 0.26, 7.8, Y11 + 0.32)

# ── 左侧逻辑线 ───────────────────────────────────────
ax.plot([0.2, 0.2], [Y11 - 0.32, Y + 0.26], color='#C0C0C0', lw=1.5, solid_capstyle='round', zorder=1)
ax.annotate('', xy=(0.2, Y11 - 0.32), xytext=(0.2, Y11 - 0.10),
            arrowprops=dict(arrowstyle='->', color='#C0C0C0', lw=1.5))
for i, ch in enumerate('研究逻辑主线'):
    ax.text(0.2, 13.8 - i * 0.45, ch, ha='center', va='center',
            fontsize=6.5, color='#AAAAAA', fontweight='bold')

# ── 图号 ─────────────────────────────────────────────
ax.text(5.5, 5.9, '图 1-1  技术路线图',
        ha='center', va='center', fontsize=9, fontweight='bold', color=TXT)

# ── 保存 ─────────────────────────────────────────────
plt.tight_layout(pad=0.3)
out = 'F:/all_exp/Thesis/论文图片/Figure_1-1/Figure_1-1_技术路线图.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f"SAVED → {out}")
