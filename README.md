# LucidChaos (清醒混沌)

> **在狂乱混沌的梦境中，保持绝对清醒的执掌者。**  
> 《卡厄思梦境》（Chaos Zero Nightmare）PC端原生自动化助手。

<p align="center">
  <b>纯原生 Python 轻量架构 · 零黑盒挂机框架依赖 · 多模态视觉场景路由 · 现代化 Fluent Design 客户端</b>
</p>

---

## 💡 设计理念：原生轻量自研架构（Zero-Framework）

本项目**完全没有引入任何第三方自动化黑盒框架**（无 OneDragon / ok-framework / MAA / Airtest 等沉重依赖），所有核心链路均为原生纯 Python 精准实现：

- **底层后台捕获**：纯 Windows 原生 Win32 API (`win32gui`, `win32ui`, `ctypes`, GDI `PrintWindow` / `BitBlt`)，毫秒级后台显卡位图流，不干扰前台工作。
- **底层后台输入**：纯 Win32 `PostMessage` / `SendMessage` 后台键鼠驱动，严格贴合游戏官方快捷键（F1/F2/F3、数字选牌、ENTER确认、E回合结束、SPACE对白）。
- **多模态视觉引擎**：轻量级 `RapidOCR (ONNXRuntime)` + `OpenCV` 原生 HSV 颜色通道与特征金字塔，毫秒级快速推理。
- **场景路由与决断**：自研轻量 FSM + 场景池动态打分路由 (`ScenePoolClassifier`) + 组合条件文字锁定，从底层杜绝场景跳戏与误触。
- **动作容错阶梯**：自研 `ActionLadder`（同一场景停留时智能递进选牌与交互）+ `StuckGuard` 看门狗三级防卡死机制。
- **离线回归验证体系**：纯原生无头打桩回归套件（内置 28 套冻结真实真机场景夹具），无需开游戏即可随时毫秒级自检。

---

## 🚀 快速开始

### 1. 环境准备
确保已安装 Python 3.10+，执行以下命令安装依赖：
```bash
pip install -r requirements.txt
```

### 2. 启动方式

#### 方式一：双击一键启动（推荐）
直接双击根目录下的 **`start_gui.bat`**。

#### 方式二：命令行启动
```bash
python run_gui.py
```

---

## 🎨 功能特性

1. **主控看板 (Home Dashboard)**
   - **一键启停**：全局快捷键 **`F8` 启动** / **`F9` 停止**。
   - **实时状态感知**：动态显示场景识别结果（肉鸽选路、神闪选牌、角色配卡、战斗进行、营地休息等）、运行时间与执行步数。
   - **富文本日志流**：分类高亮输出各场景判定证据、点击坐标与决策细节。

2. **核心业务场景全覆盖**
   - **肉鸽选路**：分支识别、青色地面与发光门智能导航，非战斗/问号/营地路线智能优选。
   - **灵光一闪 / 神闪卡牌**：
     - 独家算法：通过金色包边 HSV 光谱密度精准识别顶级强化**「神闪」**卡牌并最高优先级选定；
     - 普通灵光一闪：智能 1 -> 2 -> 3 阶梯递进选牌与自适应确认。
   - **卡牌交付与中立卡过滤**：
     - 自动识别战斗员槽位并先点击角色、后敲击确认；
     - 智能识别中立卡职业限制标签（`無法獲得`），按规则跳过中立卡（F1）。
   - **局内常驻 2X 加速**：自动监控右上角 ⏩ 状态并补刀点亮。
   - **BOSS 战与局内战斗**：原生 AUTO 自动激活，锁定回合时接管自动出牌与按 E 结束回合。
   - **系统零号深层创伤与治疗中心**：自动定位治疗槽位，智能执行角色消除记忆重置流程。
   - **奇遇互动与宝箱开启**：精准识别宝箱开启交互，无缝推进事件对话与选项选择。

3. **视觉调试工具箱 (Toolbox)**
   - **窗口连接诊断**：实时读取游戏窗口句柄 (HWND)、客户区尺寸与后台渲染画面。
   - **模板截取工具**：支持直接在运行画面上框选保存新场景与图标模板。

---

## 📁 目录结构

```text
├── lucid_chaos/
│   ├── core/              # 视觉底座 (VisionService, Features, Win32 WindowCapture)
│   ├── fsm/               # 状态机引擎与全局上下文 (FSMContext, Engine)
│   ├── gui/               # Fluent UI 前端界面 (QFluentWidgets)
│   ├── logic/             # 场景业务引擎 (NodeDecisionEngine, BattleManager, PathPlanner)
│   ├── router/            # 场景池评估器与文字锁定路由 (SceneRouter, ScenePool, Evaluators)
│   ├── state/             # 画面观察者与状态定义 (ScreenObserver, GameState)
│   └── tests/             # 离线回归测试套件与真实夹具库 (28 组真实场景回归)
├── run_gui.py             # GUI 主启动入口
├── start_gui.bat          # Windows 一键启动脚本
├── requirements.txt       # 依赖清单
└── README.md              # 项目说明文档
```

---

## 🛡️ 离线自测套件

在修改任何判定逻辑或增加新场景后，无需打开游戏即可通过离线回归测试验证全场景完整性：

```bash
# 跑全量 28 组真实场景回归测试
python -m lucid_chaos.tests.regress

# 跑神闪卡牌专项算法测试
python lucid_chaos/tests/test_god_epiphany.py

# 跑中立卡分配专项测试
python lucid_chaos/tests/test_neutral_card_assign.py
```
